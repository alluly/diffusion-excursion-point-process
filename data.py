import sim

import numpy as np

import torch

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def get_data(mu, sigma, use_max=False, tau=0, dt=0.01, N=1000, saveplot=True, offset=0, x_init=0):
    '''
    Returns a sample path of the diffusion process associated with mu and sigma.

    mu : drift function
    sigma : diffusion function

    use_max : boolean on whether to use running maximum as excursions
    tau : delay window (not used)
    dt  : time resolution
    N   : number of time steps
    saveplot : boolean to save the plot of the data

    offset : hitting time of constant level (usually 0, but GBM would be 1, for example)
    '''
    xt, t, snell_max, snell_min = sim.ito_diffusion(mu, sigma, x_init=x_init, tau=tau, dt=dt, N=N)
    xt = xt - offset

    idxmaxes = xt == snell_max
    idxmins  = xt == snell_min

    maxes = xt[idxmaxes][1:]
    mins  = xt[idxmins][1:]

    tmax = t[idxmaxes][1:]
    tmin = t[idxmins][1:]

    max_points = torch.stack((tmax, maxes), 1)
    min_points = torch.stack((tmin, mins), 1)

    pp_ = torch.cat((max_points, min_points))

    inds = pp_.sort(0)[1][:,0]

    pp = pp_[inds,:]

    if use_max:
        sx = np.maximum.accumulate(xt)
        xt = sx - xt

    if saveplot:
        run_path = 'figs/run_path.pdf'

        plt.plot(t.detach().numpy(), xt.detach().numpy(), linewidth=0.5, label=r'$Z_t$')
        plt.plot(t.detach().numpy(), snell_max.detach().numpy(), linewidth=0.5, label=r'$\mathbb{E}[f^{-1}(Ask)]$')
        plt.plot(t.detach().numpy(), snell_min.detach().numpy(), linewidth=0.5, label=r'$\mathbb{E}[f^{-1}(Bid)]$')
        plt.scatter(tmin, mins, color='tab:red',s=2, alpha=0.3, label=r'$\ell_{\max}$')
        plt.scatter(tmax, maxes,color='tab:purple',s=2, alpha=0.3, label=r'$\ell_{\min}$')
        plt.legend()
        plt.savefig(run_path)
        plt.close('all')

        plt.scatter(tmax, torch.zeros_like(tmax), s=4, color='tab:purple', label='Maxes')
        plt.scatter(tmin, torch.zeros_like(tmin), s=4, color='tab:red', label='Mins')
        plt.legend()
        plt.savefig('figs/points.pdf')
        plt.close('all')

    return xt, t, tau, pp 

def get_data_nd(mu, sigma, d, use_max=False, tau=0, dt=0.01, N=1000, saveplot=True, offset=0):
    '''
    Returns a sample path of the diffusion process associated with mu and sigma.
    '''
    x_init = torch.randn(d)

    xt, t, snell_max, snell_min = sim.ito_diffusion_nd(mu, sigma, x_init=x_init, tau=tau, dt=dt, N=N)
    t = t.unsqueeze(1).repeat(1,d)
    xt = xt - offset

    return xt, t, None, None

def setup_data(data, use_curve=False, ex_name='test', min_height=0, marks=False):
    '''
    Formats the data in an appropriate way. 
    
    data : expects a tuple of X_t (stochastic process) t (time steps) tau (delay times, not used) and running max/min process

    use_curve : boolean that checks whether or not to consider the hitting times of the expectations at running max/min

    ex_name : name of the experiment

    min_height : minimum height of the excursion

    marks : boolean on if should return the mark of the excursion (positive or negative)
    '''

    xt, t, tau, pp = data
    if len(t.shape) < 2:
        xt = xt.unsqueeze(1)
        t  =  t.unsqueeze(1)

    dx0 = (xt[:-1] * xt[1:]) < 0 # get the zero crossings

    if xt.shape[1] == 1:

        min_reached_mask = torch.zeros_like(dx0)

        run_max  = 0 # track the running maximum
        zero_idx = 0 # the index where the starts

        for idx in range(dx0.shape[0]):
            if dx0[idx] == 1:
                if run_max > min_height:
                    min_reached_mask[zero_idx] = 1
                zero_idx = idx
                run_max = 0
            else:
                if torch.abs(xt[idx]) > run_max:
                    run_max = torch.abs(xt[idx])
        dx0_orig = dx0.clone()

        dx0 = dx0_orig * min_reached_mask

    if dx0.sum() == 0: 
        # if there are no zeros, use the running maximum
        dx0 = xt == 0
        dx0 = dx0[1:]

    if dx0.shape[1] == 1:

        t0 = (t[:-1][ dx0]).unsqueeze(1)
        t0_orig = (t[:-1][ dx0_orig]).unsqueeze(1)
        t1 = (t[:-1][~dx0]).unsqueeze(1)

        t0a = []
        t1a = []

        plt.style.use('seaborn-darkgrid')
        zerotimes = t0
        plt.scatter(zerotimes,torch.ones_like(zerotimes))
        plt.scatter(t0_orig,torch.zeros_like(t0_orig))
        plt.xlabel(r'$t$')
        plt.title('Arrivals of Point Process')
        plt.tight_layout()
        plt.savefig('figs/{}_points_d.pdf'.format(ex_name))
        plt.close('all')

    else:

        t0 = (t[:-1] * dx0).split(1,1)
        t1 = (t[:-1] * (~dx0)).split(1,1)
        t0a = []
        t1a = []

        for ind in range(dx0.shape[1]):
            zerotimes = t0[ind][t0[ind] != 0]
            t0a.append(zerotimes)
            t1a.append(t1[ind][t1[ind] != 0])
            plt.style.use('seaborn-darkgrid')
            plt.scatter(zerotimes, ind*torch.ones_like(zerotimes))
        plt.xlabel(r'$t$')
        plt.ylabel('Class Index')
        plt.title('Arrivals of Point Process')
        plt.tight_layout()
        plt.savefig('figs/{}_multi_d.pdf'.format(ex_name))
        plt.close('all')
        t0 = t0a 
        t1 = t1a


    if type(t0) == list:
        a = [torch.zeros_like(t0i) for t0i in t0]
        b = [torch.zeros_like(t0i) for t0i in t0]
    else:
        a = torch.zeros_like(t0)
        b = torch.zeros_like(t0)

    if use_curve:
        t0 = pp[:,0]
        a  = pp[:-1, 1]
        b  = pp[1:,1]

    if marks:

        mark = xt[:-1][dx0] > 0 

        return t, t0, xt, pp, a, b, mark


    return t, t0, xt, pp, a, b
