import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.distributions import Normal
from scipy.special import erf,gammaincc, gamma

from utils import brownian_bridge, brownian_bridge_nd, excursion

import numpy as np

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy.interpolate import griddata, NearestNDInterpolator, interp1d, interpn

class LocalTimePP(nn.Module):

    def __init__(self, 
            mu : nn.Module, 
            sigma : nn.Module, 
            t  : torch.tensor,
            t0 : torch.tensor,
            xt = None,
            a  = None,
            b  = None,
            history = None,
            stim = None,
            d  = 1, 
            n_e = 100,
            use_bridge=False,
            use_time = False,
            mu_type='state'):
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
        '''
        super(LocalTimePP, self).__init__()

        self.mu    = mu    # drift net
        self.sigma = sigma # diffusion net

        self.t  = t  # time horizon

        if len(self.t.shape) > 2:
            self.t = t[:,:,0]

        self.t0 = t0 # zero times
        self.history = history
        self.stim = stim

        self.use_time = use_time # condition the drift on time
        self.mu_type = mu_type

        all_int = []

        self.use_bridge = use_bridge

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

            self.ijk = torch.meshgrid((torch.arange(self.inds.shape[0]), torch.arange(self.inds.shape[1]), torch.arange(self.inds.shape[2])), 'ij')

        else:
            # In the 1-d case, first check if we have a list of paths or one sample path
            if type(t0) == list or type(t0) == tuple:
                all_times = []
                for t00 in t0:
                    all_int = []

                    for it0 in range(t00.shape[0] - 1):
                        
                        e_time = torch.linspace(t00[it0].item(), t00[it0+1].item(), n_e)
                        all_int.append(e_time)

                    all_times.append(torch.stack(all_int))
                self.e_times = all_times# get all the excursion times
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


    def girsanov(self, eps=torch.Tensor([1e-2]), prior_type=None, neg=False):
        '''
        eps : float, minimum value that the excursions should be
        prior_type : string, type of prior to impose on the SDE
        neg : boolean, whether to use negative excursions or not
        '''

        N = len(self.e_times)

        loss = 0 
        expect = 0
        p = 0 

        eps = eps.abs()

        if N > 2:
            s_idx = torch.randint(N-2, (1,1))[0]
        else:
            s_idx = 0 

        for e_idx in range(s_idx, s_idx + 1):

            e_time = self.e_times[e_idx]

            if len(e_time.shape) > 2:
                # headache for the nd case ...
                d  = e_time.shape[0]
                expect = 0
                K = 1000
                t = self.time_grid.unsqueeze(-1).repeat(1,d)
                for _ in range(K):
                    if self.use_bridge:
                        x, _ = brownian_bridge_nd(e_time) # simulate bridges
                    else:
                        x, _ = excursion(e_time, neg=neg, N=d) # simulate excursions

                    x1 = x
                    x1[x1.isnan()] = 0
                    et = e_time

                    x1 = x1[:,1:-1,:].permute(1,2,0).reshape(-1,d)
                    et = et[:,1:-1,:].permute(1,2,0).reshape(-1,d)

                    interp = x1
                    mu_t = self.mu(interp)

                    a = (mu_t[:-1,:] * (interp[1:,:] - interp[:-1,:])).sum()
                    b = (mu_t[:-1,:] ** 2 * (et[1:,:] - et[:-1,:])).sum()
                    s = a - 0.5 * b 
                    expect += s.exp() / K
                dt = t[1:] - t[:-1]
                p = torch.zeros(1)

            else:
                # The 1d case is easy :)
                K = 200
                x1, _ = excursion(e_time, neg=neg, N=K) # simulate excursions
                x1 = x1.unsqueeze(-1)
                et = e_time.unsqueeze(0).repeat(K,1,1).unsqueeze(-1)

                if self.use_time:
                    mu_t = self.mu_t(x1, et)
                else:
                    mu_t = self.mu(x1)

                a = (mu_t[:,:,:-1,:] * (x1[:,:,1:,:] - x1[:,:,:-1,:])).sum(-1).sum(-1)
                b = (mu_t[:,:,:-1,:]** 2 * (et[:,:,1:,:] - et[:,:,:-1,:])).sum(-1).sum(-1)
                s = a - 0.5*b

                expect = torch.exp(s).mean(0).log()
                #expect = s.mean(0)

                if type(self.t0) == list or type(self.t0) == tuple:
                    dt = self.t0[e_idx][1:] - self.t0[e_idx][:-1] 
                else:
                    dt = self.t0[1:] - self.t0[:-1] 


                if prior_type == 'jeff':
                    p = torch.log(eps * gamma(1e-5) * torch.special.gammaincc(lower, 2*eps**2 / self.t)) - torch.tensor(np.log(np.sqrt(2*np.pi))) - (self.t**(3/2)).log()
                elif prior_type == None:
                    p += ((eps + 1e-7).log() + torch.tensor(np.sqrt(2/np.pi)).log() - 2 * eps ** 2 / dt - (dt**(3/2) + 1e-7).log())
                    if len(p.shape) > 1:
                        p = p.squeeze()

        return (p + expect).mean()


    def girsanov_history(self, eps=torch.Tensor([1e-2]), prior_type=None, neg=False, K=400):
        '''
        eps : float, minimum value that the excursions should be
        prior_type : string, type of prior to impose on the SDE
        neg : boolean, whether to use negative excursions or not
        '''

        N = len(self.e_times)

        loss = 0 
        expect = 0
        p = 0 

        eps = eps.abs()
        e_time = self.e_times[0]

        if self.use_bridge:
            x1, _ = brownian_bridge(e_time, N=K)
        else:
            x1, _ = excursion(e_time, neg=neg, N=K) # simulate excursions
        x1 = x1.unsqueeze(-1)
        et = e_time.unsqueeze(0).repeat(K,1,1).unsqueeze(-1)
        
        mu_t = self.mu(x1, et, self.history)
        
        a = (mu_t[:,:,:-1,:] * (x1[:,:,1:,:] - x1[:,:,:-1,:])).sum(-1).sum(-1)
        b = (mu_t[:,:,:-1,:]** 2 * (et[:,:,1:,:] - et[:,:,:-1,:])).sum(-1).sum(-1)
        s = a - 0.5*b

        #expect = torch.exp(s).mean(0).log()
        expect = (a - 0.5 * b).mean(0)

        if type(self.t0) == list or type(self.t0) == tuple:
            dt = self.t0[e_idx][1:] - self.t0[e_idx][:-1] 
        else:
            dt = self.t0[1:] - self.t0[:-1] 

        return (expect).mean()


    def girsanov_stim(self, eps=torch.Tensor([1e-2]), prior_type=None, neg=False, K=400):
        '''
        eps : float, minimum value that the excursions should be
        prior_type : string, type of prior to impose on the SDE
        neg : boolean, whether to use negative excursions or not
        '''
        N = len(self.e_times)

        loss = 0 
        expect = 0
        p = 0 

        eps = eps.abs()
        e_time = self.e_times[0]

        # K = 200
        if self.use_bridge:
            x1, _ = brownian_bridge(e_time, N=K) # simulate excursions
        else:
            x1, _ = excursion(e_time, neg=neg, N=K) # simulate excursions
        x1 = x1.unsqueeze(-1)
        et = e_time.unsqueeze(0).repeat(K,1,1).unsqueeze(-1)

        # if self.use_time and self.mu_type != 'state_stim':
        #     mu_t = self.mu_t(x1, et)
        # elif not self.use_time and self.mu_type != 'state_stim':
        #     mu_t = self.mu(x1)
        # elif self.mu_type == 'state_stim':
        mu_t = self.mu(x1, et, self.stim)

        a = (mu_t[:,:,:-1,:] * (x1[:,:,1:,:] - x1[:,:,:-1,:])).sum(-1).sum(-1)
        b = (mu_t[:,:,:-1,:]** 2 * (et[:,:,1:,:] - et[:,:,:-1,:])).sum(-1).sum(-1)
        s = a - 0.5*b

        expect = torch.exp(s).mean(0).log()
        #expect = s.mean(0)

        if type(self.t0) == list or type(self.t0) == tuple:
            dt = self.t0[e_idx][1:] - self.t0[e_idx][:-1] 
        else:
            dt = self.t0[1:] - self.t0[:-1] 

        if prior_type == 'jeff':
            p = torch.log(eps * gamma(1e-5) * torch.special.gammaincc(lower, 2*eps**2 / self.t)) - torch.tensor(np.log(np.sqrt(2*np.pi))) - (self.t**(3/2)).log()
        elif prior_type == None:
            p += ((eps + 1e-7).log() + torch.tensor(np.sqrt(2/np.pi)).log() - 2 * eps ** 2 / dt - (dt**(3/2) + 1e-7).log())
            if len(p.shape) > 1:
                p = p.squeeze()

        return (p+expect).mean()


    def prior(self, eps, prior_type=None):
        eps = eps.abs()

        p = 0 
        lower = torch.tensor(1e-5)

        if type(self.t0) == list or type(self.t0) == tuple:
            N = len(self.e_times)
            for e_idx in range(N):
                dt = self.t0[e_idx][1:] - self.t0[e_idx][:-1] 

                if prior_type == 'jeff':
                    p += (torch.log(eps * gamma(1e-5) * torch.special.gammaincc(lower, 2*eps**2 / dt)) - torch.tensor(np.log(np.sqrt(2*np.pi))) - (dt**(3/2)).log()).mean()
                elif prior_type == None:
                    p += ((eps + 1e-7).log() + torch.tensor(np.sqrt(2/np.pi)).log() - 2 * eps **2 / dt - (dt**(3/2) + 1e-7).log()).mean()
        else:
            dt = self.t0[1:] - self.t0[:-1] 
            if prior_type == 'jeff':
                p += (torch.log(eps * gamma(1e-5) * torch.special.gammaincc(lower, 2*eps**2 / dt)) - torch.tensor(np.log(np.sqrt(2*np.pi))) - (dt**(3/2)).log()).mean()
            elif prior_type == None:
                p += ((eps + 1e-7).log() + torch.tensor(np.sqrt(2/np.pi)).log() - 2 * eps **2 / dt - (dt**(3/2) + 1e-7).log()).mean()

        return p

    def logp(self, t, t0=0.000001,  eps=torch.Tensor([1e-2]), prior_type=None, noise=None):
        '''

        Function to compute logp(t). t is assumed to be the first hitting time.

        '''

        # define some parameters on tolerances
        lower = 1e-5*torch.ones(1)
        neg = False 

        eps  = eps.abs()

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
        '''
        if x1_.shape[0] > 0:
            K = x1_.shape[0]
            x1 = x1_.unsqueeze(1)[:,:,:].reshape(x1_.shape[0],-1,1)
        else:
            x1 = x1.reshape(K, -1, 1)
        '''

        x1 = x1.unsqueeze(-1)

        et = e_time.unsqueeze(0).repeat(K,1,1).unsqueeze(-1)
        #et = e_time.unsqueeze(0).repeat(K,1,1)
        #et = et[:,:,:].reshape(K,-1,1)

        # sample mu
        if self.use_time:
            mu_t = self.mu(x1, et, self.stim)
        else:
            mu_t = self.mu(x1, et, self.stim)
                        
        # girsanov
        a = (mu_t[:,:,:-1,:] * (x1[:,:,1:,:] - x1[:,:,:-1,:])).sum(-1).sum(-1)
        b = (mu_t[:,:,:-1,:] ** 2 * (et[:,:,1:,:] - et[:,:,:-1,:])).sum(-1).sum(-1)
        s = a - 0.5*b
        expect = torch.exp(s).mean(0)

        if prior_type == 'jeff':
            prior = torch.log(eps * gamma(1e-5) * torch.special.gammaincc(lower, 2*eps**2 / t)) - torch.tensor(np.log(np.sqrt(2*np.pi))) - torch.log(t**(3/2))
        elif prior_type == None:
            prior = (eps).log() + torch.tensor(np.sqrt(2/np.pi)).log() - 2 * eps **2 / t - torch.log(t**(3/2))

        return expect.log() #+ prior, e_time 

    def mu_t(self, x, t=None):

        if self.use_time:

            if t is not None:
                inx = torch.cat((x,t),-1)
            else:
                inx = torch.cat((x, torch.zeros_like(x)),-1)
        else:
            inx = x

        return self.mu(inx)

    def sample(self, K, N, T, eps):
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
                X[:, idx+1] = X[:, idx] + dt * self.mu(X[:,idx], t[idx] * torch.ones(K, 1), self.stim[idx]) + dt.sqrt() * torch.randn(K, 1)

        zeros  = X[:,1:] * X[:,:-1] < 0 # where it hits zero
        thresh = (X[:,1:] > eps).cumsum(1) # how many times it reached at least epsilon
        height_reached = (zeros * thresh).cummax(1)[0] # find where the zeros occur and see if the height was reached
        valid_inds = height_reached[:,1:] > height_reached[:,:-1] # choose the indices where there is a difference
        
        t0 = t.unsqueeze(0).unsqueeze(-1).repeat(K,1,1)[:,2:][valid_inds] # choose the times

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
        dt0 = torch.cat(dt0)

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

