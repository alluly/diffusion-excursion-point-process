import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.distributions import Normal
from scipy.special import erf,gammaincc, gamma

from utils import brownian_bridge, brownian_bridge_nd, excursion

from nets import HistoryMLP

import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy.interpolate import griddata, NearestNDInterpolator, interp1d, interpn

class LocalTimePP(nn.Module):

    def __init__(self, 
            mu : nn.Module, 
            sigma : nn.Module, 
            t  : torch.tensor,
            t0 : torch.tensor,
            eps: nn.Parameter, 
            xt = None,
            a  = None,
            b  = None,
            d  = 1, 
            n_e = 100,
            use_bridge=False,
            use_time = False,
            prior = None, 
            param_model : str = None,
            marks = None):

        '''
        Creates an instance of the LocalTimePP class which defines
        --- a point process on the local time of a diffusion. 

        mu : nn.Module    -- module that defines the drift 
        sigma : nn.Module -- module that defines the diffusion

        t  : torch.tensor -- tensor for all times over horizon
        t0 : torch.tensor -- tensor for all zero times 

        xt : torch.tensor  -- state space of diffusion
        a  : torch.tensor  -- values for upper curve (optional)
        b  : torch.tensor  -- values for lower curve (optional)

        n_e : int -- number of samples between each excursion

        marks : torch.tensor -- provides mark for each excursion
        '''
        super(LocalTimePP, self).__init__()

        self.param_model = param_model # for saving

        self.mu    = mu    # drift net
        self.sigma = sigma # diffusion net

        self.eps = eps # epsilon

        self.t  = t  # time horizon

        if len(self.t.shape) > 2:
            self.t = t[:,:,0]

        self.t0 = t0 # zero times

        self.use_time = use_time # condition the drift on time

        all_int = []

        self.use_bridge = use_bridge

        self.prior = prior

        # In the n-d case we must compute different excursion times for each dim

        if d > 1:

            assert d == len(t0)


            max_len = max([len(ti) for ti in t0]) + 1
            tensor_e_times = torch.zeros(d, max_len, n_e)

            self.time_grid = torch.linspace(t.min().item(), t.max().item(), int(1e4))

            # if t0 is a list then we have multi dimensions
            for ind_d, t0i in enumerate(t0):

                # for each set of excursion times, create a line grid
                e_times = []

                for it0 in range(-1,t0i.shape[0] - 1):
                    if it0 == -1:
                        e_time = torch.linspace(0, t0i[it0+1].item(), n_e)
                    elif it0 == t0i.shape[0]-1:
                        e_time = torch.linspace(t0i[it0+1].item(), t.max().item(), n_e)
                    else:
                        e_time = torch.linspace(t0i[it0].item(), t0i[it0+1].item(), n_e)

                    e_time_grid = e_time 
                    tensor_e_times[ind_d, it0+1, :] = e_time_grid.clone()
                    e_times.append(e_time_grid)

                tensor_e_times[ind_d, it0+1 : max_len, :] =  t.max().clone().item()

                all_int.append(e_times)

            bucket_inds = torch.bucketize(tensor_e_times, self.time_grid)
            self.e_times = [tensor_e_times]
            self.inds = bucket_inds.argsort(-1)

            self.ijk = torch.meshgrid(torch.arange(self.inds.shape[0]), torch.arange(self.inds.shape[1]), torch.arange(self.inds.shape[2]))

        else:
            if marks is not None:
                print(len(marks))
                print(len(t0))
                assert len(marks) == len(t0)
                self.marks = marks
            else:
                self.marks = 0
            # In the 1-d case, first check if we have a list of paths or one sample path
            if type(t0) == list or type(t0) == tuple:
                all_times = []
                for t00 in t0:
                    all_int = []

                    for it0 in range(t00.shape[0] - 1):
                        e_time = torch.linspace(t00[it0].item(), t00[it0+1].item(), n_e)
                        all_int.append(e_time)
                    all_times.append(torch.stack(all_int))
                self.e_times = all_times # get all the excursion times
            else:
                # With one sample path, compute the excursion times between each interarrival time 
                for it0 in range(t0.shape[0] - 1):
                    e_time = torch.linspace(t0[it0].item(), t0[it0+1].item(), n_e)
                    all_int.append(e_time)
                self.e_times = [torch.stack(all_int)] # get all the excursion times

        if xt is not None:
            self.xt = xt
        else:
            xt = torch.linspace(-2, 2, t.shape[0]).unsqueeze(1)

        # if there are marks (a, b) then use them
        # if not, just use zeros
        
        if a is not None:
            assert b is not None, 'must provide two curves'
            self.a = a
            self.b = b
            self.use_curve = True
        else:
            self.use_curve = False

        # fin

    def train_step(self):
        '''
        Calculates one step of the training routine.
        '''
        loss = -self.girsanov()

        return loss

    def girsanov(self, prior_type=None, neg=False):
        '''
        prior_type : string, type of prior to impose on the SDE
        neg : boolean, whether to use negative excursions or not
        '''

        N = len(self.e_times)

        loss = 0 
        expect = 0
        p = 0 

        eps = self.eps.abs()

        if N > 2:
            s_idx = torch.randint(N-1, (1,1))[0]
        else:
            s_idx = 0 

        for e_idx in range(s_idx, s_idx + 1):
        #for e_idx in range(N):

            e_time = self.e_times[e_idx]

            if len(e_time.shape) > 2:
                # headache for the nd case ...
                d  = e_time.shape[0]
                expect = 0
                K = 200
                t = self.time_grid.unsqueeze(-1).repeat(1,d)
                for _ in range(K):
                    if self.use_bridge:
                        x, _ = brownian_bridge_nd(e_time) # simulate bridges
                    else:
                        x, _ = excursion(e_time, neg=True, N=d) # simulate excursions

                    x1 = x
                    x1[x1.isnan()] = 0
                    et = e_time

                    x1 = x1[:,1:-1,:].permute(1,2,0).reshape(-1,d)
                    et = et[:,1:-1,:].permute(1,2,0).reshape(-1,d)

                    interp = x1
                    mu_t = self.mu(interp)

                    a = (mu_t[:-1,:] * (interp[1:,:] - interp[:-1,:])).sum()
                    #b = (mu_t[:-1,:] ** 2 * (self.time_grid[1] - self.time_grid[0])).sum()
                    b = (mu_t[:-1,:] ** 2 * (et[1:,:] - et[:-1,:])).sum()
                    s = a - 0.5 * b 
                    expect += s.exp() / K
                    #expect += s / K 
                dt = t[1:] - t[:-1]
                p = torch.zeros(1)

            else:
                # The 1d case is easy :)
                K = 200
                et = e_time.unsqueeze(0).repeat(K,1,1).unsqueeze(-1)
                if self.marks != 0:
                    x1, _ = excursion(e_time, neg=False, N=K) # simulate excursions
                    x1 = x1 * (self.marks[:-1]*2 - 1).unsqueeze(0).unsqueeze(-1)
                else:
                    sigma = self.sigma_t(et) # sample sigma

                    if len(sigma.shape) == 4: # reshape if it's a problem
                        sigma_in = sigma.squeeze(-1)
                    else:
                        sigma_in = sigma 

                    x1, _ = excursion(e_time, neg=neg, N=K, sigma=sigma_in.abs()) # simulate excursions
                indices = x1.abs().max(-1)[0] < eps

                x1 = x1.unsqueeze(-1) 

                mu_t = self.mu_t(x1, et, eps=eps)

                if len(sigma.shape) == 4:
                    sigma_gir = sigma[:,:,:-1,:]
                else:
                    sigma_gir = sigma

                a = (mu_t[:,:,:-1,:] * (x1[:,:,1:,:] - x1[:,:,:-1,:]) / sigma_gir ** 2).sum(-1).sum(-1)
                b = (mu_t[:,:,:-1,:]** 2 * (et[:,:,1:,:] - et[:,:,:-1,:]) / sigma_gir ** 2).sum(-1).sum(-1)
                s = a - 0.5*b

                expect = torch.exp(s).mean(0).log()
                #expect = s.mean(0)

                if type(self.t0) == list or type(self.t0) == tuple:
                    dt = self.t0[e_idx][1:] - self.t0[e_idx][:-1] 
                else:
                    dt = self.t0[1:] - self.t0[:-1] 


                if self.prior  == 'jeff':
                    p = torch.log(eps * gamma(1e-5) * torch.special.gammaincc(lower, 2*eps**2 / self.t)) - torch.tensor(np.log(np.sqrt(2*np.pi))) - (self.t**(3/2)).log()
                elif self.prior == 'arcsin':
                    #p = -np.pi*eps*torch.sqrt(dt * (eps - dt)).log()
                    p = 0.5 * (-(eps - dt).log() - dt.log() - np.log(2*np.pi))
                elif self.prior == 'diffusion':
                    p = 0 
                elif self.prior == None:
                    #p += ((eps + 1e-7).log() + torch.tensor(np.sqrt(2/np.pi)).log() - 2 * eps ** 2 / dt / self.sigma ** 2 - (dt**(3/2) * self.sigma.abs() + 1e-7).log())
                    p += ((eps + 1e-7).log() + torch.tensor(np.sqrt(2/np.pi)).log() - 2 * eps ** 2 / dt  - (dt**(3/2)  + 1e-7).log())
                    if len(p.shape) > 1:
                        p = p.squeeze()

        if len(self.e_times[0].shape) > 2:
            return expect.log()
        else:
            return (p + expect).mean() 

    def init_prior(self,  prior_type=None):
        eps = self.eps.abs()

        p = 0 
        lower = torch.tensor(1e-5)

        if type(self.t0) == list or type(self.t0) == tuple:
            N = len(self.e_times)
            for e_idx in range(N):
                dt = self.t0[e_idx][1:] - self.t0[e_idx][:-1] 

                if prior_type == 'jeff':
                    p += (torch.log(eps * gamma(1e-5) * torch.special.gammaincc(lower, 2*eps**2 / dt)) - torch.tensor(np.log(np.sqrt(2*np.pi))) - (dt**(3/2)).log()).mean()
                elif self.prior== 'arcsin':
                    p += 0.5 * (-(eps - dt).log() - dt.log() - np.log(2*np.pi)).mean()
                elif prior_type == None:
                    p += ((eps + 1e-7).log() + torch.tensor(np.sqrt(2/np.pi)).log() - 2 * eps **2 / dt - (dt**(3/2) + 1e-7).log()).mean()
        else:
            dt = self.t0[1:] - self.t0[:-1] 
            if self.prior == 'jeff':
                p += (torch.log(eps * gamma(1e-5) * torch.special.gammaincc(lower, 2*eps**2 / dt)) - torch.tensor(np.log(np.sqrt(2*np.pi))) - (dt**(3/2)).log()).mean()
            elif self.prior== 'arcsin':
                p = 0.5 * (-(eps - dt).log() - dt.log() - np.log(2*np.pi)).mean()
            elif self.prior == None:
                p += ((eps + 1e-7).log() + torch.tensor(np.sqrt(2/np.pi)).log() - 2 * eps **2 / dt - (dt**(3/2) + 1e-7).log()).mean()

        return p

    def logp(self, t, t0=1e-7, prior_type=None, noise=None, neg=False, print_prior=False):
        '''

        Function to compute logp(t). t is assumed to be the first hitting time.

        '''

        # define some parameters on tolerances
        lower = 1e-6*torch.ones(1)

        eps  = self.eps.abs()

        if noise is None:
            n_e = 100
            K = 1000
        else:
            dt = 1e-2
            K = noise.shape[0]
            n_e = noise.shape[-1]

        # get the excursion time
        e_time = torch.linspace(t0, t.item(), n_e).unsqueeze(0)

        # simulate excursions
        x1, _ = excursion(e_time, neg=neg, N=K, noise=noise) 
        #x1, _ = brownian_bridge(e_time, noise=noise, N=K) 

        indices = x1.max(-1)[0] > eps
        x1_ = x1[indices]
        if x1_.shape[0] > 0:
            K = x1_.shape[0]
            #x1 = x1_.unsqueeze(1)[:,:,:].reshape(x1_.shape[0],-1,1)
            x1_ = x1_.reshape(-1, x1.shape[1], x1.shape[2])
            x1  = x1_.unsqueeze(-1)
        else:
            #x1 = x1.reshape(K, -1, 1)
            x1 = x1.unsqueeze(-1)

        et = e_time.unsqueeze(0).repeat(K,1,1).unsqueeze(-1)
        #et = e_time.unsqueeze(0).repeat(K,1,1)
        #et = et[:,:,:].reshape(K,-1,1)

        # sample mu
        mu_t = self.mu_t(x1, et, eps=eps)
        #mu_t = self.mu_t(x1, et)

        # girsanov
        a = (mu_t[:,:,:-1,:] * (x1[:,:,1:,:] - x1[:,:,:-1,:])).sum(-1).sum(-1)
        b = (mu_t[:,:,:-1,:] ** 2 * (et[:,:,1:,:] - et[:,:,:-1,:])).sum(-1).sum(-1)
        s = a - 0.5*b
        expect = torch.exp(s).mean(0)

        if self.prior == 'jeff':
            prior = torch.log(eps * gamma(1e-5) * torch.special.gammaincc(lower, 2*eps**2 / t)) - torch.tensor(np.log(np.sqrt(2*np.pi))) - torch.log(t**(3/2))
        elif self.prior== 'arcsin':
            prior = 0.5 * (-(eps - t).log() - t.log() - np.log(2*np.pi))
        elif self.prior == 'diffusion':
            prior = 0
        elif self.prior == None:
            prior = (eps).log() + torch.tensor(np.sqrt(2/np.pi)).log() - 2 * eps **2 / t - torch.log(t**(3/2))

        if print_prior:
            print('Prior val {}'.format(prior))
            print('mu_t {}'.format(mu_t.max()))
            print('x1 {}'.format(x1.max()))
            print('et {}'.format(et.max()))
            print('expect {}'.format(expect))
            print('Log expect {}'.format(expect.log()))

        return expect.log() + prior, e_time 

    def mu_t(self, x, t=None, eps=None):

        if self.use_time:

            if t is not None:
                inx = torch.cat((x,t),-1)
            else:
                inx = torch.cat((x, torch.zeros_like(x)),-1)
        else:
            inx = x

        if isinstance(self.mu, HistoryMLP):
            return self.mu(x, t)
            
        if eps is not None:
            return self.mu(inx) #* (x > eps)

        return self.mu(inx)

    def sigma_t(self, t):

        if isinstance(self.sigma, HistoryMLP):
            return self.sigma(None, t)
        else:
            return self.sigma

    def sample(self, K, N, T, eps, min_t = 0, use_first=True, use_abs=False, take_mean=False):
        '''
        K number of samples
        N number of time steps
        T final times
        eps min height
        '''
        t = torch.linspace(0,T,N)
        dt = t[1] - t[0]
        X = torch.zeros(K, N, 1)

        with torch.no_grad():

            for idx in range(N - 1):
                X[:, idx+1] = X[:, idx] + dt * self.mu_t(X[:,idx], t[idx] * torch.ones(K, 1)) + dt.sqrt() * torch.randn(K, 1) * self.sigma_t(t[idx]*torch.ones(K,1)).abs() 

        if take_mean:
            X = X.mean(0,keepdim=True)

        zeros  = X[:,1:] * X[:,:-1] < 0 # where it hits zero

        if use_abs:
            thresh = (X[:,1:].abs() > eps).cumsum(1) # how many times it reached at least epsilon (use absolute value)
        else:
            thresh = (X[:,1:] > eps).cumsum(1) # how many times it reached at least epsilon

        height_reached = (zeros * thresh).cummax(1)[0] # find where the zeros occur and see if the height was reached
        valid_inds = height_reached[:,1:] > height_reached[:,:-1] # choose the indices where there is a difference

        if use_first:
            first_time  = torch.argmax(valid_inds.int(),1).unsqueeze(-1)
            valid_inds_ = torch.zeros_like(valid_inds)
            valid_inds_.scatter_(1,first_time, torch.ones_like(first_time, dtype=valid_inds_.dtype))
            valid_inds_[:,0,:] = 0 
            #valid_inds_[first_time] = 1
            valid_inds = valid_inds_
        
        t0 = t.unsqueeze(0).unsqueeze(-1).repeat(X.shape[0],1,1)[:,2:][valid_inds] # choose the times

        num_zeros = valid_inds.sum(1).squeeze(-1)
        dt0 = []
        base_iter = 0 
        for idx, n in enumerate(num_zeros):
            if n > 0:
                times = torch.zeros(n)
                times[0]  = t0[base_iter].clone()
                times[1:] = (t0[base_iter+1:base_iter+n] - t0[base_iter:base_iter+n-1]).clone()
                dt0.append(times)
            base_iter += n

        cmap = plt.cm.get_cmap('Dark2')
        if  X.shape[0] >= 3:
            plt.plot(t,  eps * torch.ones_like(t), 'g--', label=r'$\epsilon$')
            for idx in range(3):
                plt.plot(t, X[idx,:], linewidth=0.55, alpha=0.75, color=cmap(idx))
                plt.scatter(t[2:][valid_inds[idx,:].squeeze()], torch.zeros_like(t[2:][valid_inds[idx,:].squeeze()]), color=plt.gca().lines[-1].get_color(), s=1.2)
                #plt.scatter(dt0[idx].cumsum(0), torch.zeros_like(dt0[idx]), color='red', s=0.4)
            #plt.plot(t, -eps * torch.ones_like(t), 'g--')
            plt.xlabel(r'$t$')
            plt.ylabel(r'$Z_t$')
            plt.title('Reconstructed Paths')
            plt.tight_layout()
            plt.savefig('figs/{}/reached.pdf'.format(self.param_model))
            plt.close('all')
        if len(dt0) > 0:
            dt0 = torch.cat(dt0) + min_t
        else:
            return None, None, None, None
        return t0, dt0, X, t


    def intensity(self, t0, tn, dt=0.01):
        '''
        Returns the intensity function
        from t0 to tn
        '''

        t = torch.linspace(1e-2, tn.item(), int((tn)/dt))
        probs = torch.zeros_like(t)

        for idx, tn in enumerate(t):

            prob = (self.logp(tn)[0]).exp()
            probs[idx] = prob.clone()

        surv = 1 - probs.cumsum(-1) * dt

        return probs / surv, t

    def forward(self, x): 
        t,xt = x
        return self.logp(t).exp()
    
    def bessel_bridge(self):
        #x1 = (torch.sqrt(e_time).unsqueeze(1) * brownian_bridge_nd(e_time.unsqueeze(1).repeat(1,3,1))[0]).norm(p=2, dim=2)
        #x1 = x1[:,1:,:].reshape(K,-1,1)
        #a = (mu_t[:,:-1] * (x1[:,1:,:] - x1[:,:-1,:])).sum(-1).sum(-1)
        #b = (mu_t[:,:-1] ** 2 * (et[:,1:,:] - et[:,:-1,:])).sum(-1).sum(-1)
        #print((x1[:,1:,:] - x1[:,:-1,:]).mean())
        #s = -F.relu(-a - 0.5* b)
        pass

