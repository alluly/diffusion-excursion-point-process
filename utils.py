import torch
import torch.distributions as D
import torch.nn as nn

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def brownian_bridge_ab(t, a, b):
    '''
    Samples a Brownian Bridge from a to b.
    '''

    dt = t[:,1] - t[:,0]
    t = (t - t[:,0].unsqueeze(1)) / (t[:,-1] - t[:,0]).unsqueeze(1)

    dW = torch.randn_like(t) * dt.sqrt().unsqueeze(1)
    W = dW.cumsum(1)
    W[:,0] = 0
    W = W + a.unsqueeze(1)

    BB = W - t * (W[:,-1] - b).unsqueeze(1)

    return BB, t

def brownian_bridge(t, N=1, noise=None, sigma=1):
    '''
    Samples a Brownian Bridge from 0 to 0.
    '''

    dt = t[:,1] - t[:,0]
    t = (t - t[:,0].unsqueeze(1)) / (t[:,-1] - t[:,0]).unsqueeze(1)

    if noise is None:
        noise = torch.randn((N, t.shape[0], t.shape[1]))

    dW = noise * dt.sqrt().unsqueeze(1) * sigma
    W = dW.cumsum(-1)
    W[:,:,0] = 0

    BB = W - t * (W[:,:,-1]).unsqueeze(-1)
    
    return BB, t

def brownian_bridge_nd(t, N = 100, noise=None):
    '''
    Samples a Brownian sheet from 0 to 0.
    '''

    dt = t[:,:,1] - t[:,:,0]
    t = (t - t[:,:,0].unsqueeze(-1)) / (t[:,:,-1] - t[:,:,0]).unsqueeze(-1)

    if noise is None:
        noise = torch.randn((N, t.shape[0], t.shape[1], t.shape[2]))

    dW = noise * dt.sqrt().unsqueeze(-1)
    W = dW.cumsum(-1)
    W[:,:,:,0] = 0

    BB = W - t.unsqueeze(0) * (W[:,:,:,-1]).unsqueeze(-1)
    
    return BB, t

def brownian_bridge_nd(t, noise=None):
    '''
    Samples a Brownian Bridge from 0 to 0.
    '''

    dta = (t[:,:,1] - t[:,:,0])
    dt = (t[:,:,1] - t[:,:,0]).clone()
    t = (t - t[:,:,0].unsqueeze(-1)) / (t[:,:,-1] - t[:,:,0]).unsqueeze(-1)

    if noise is None:
        noise = torch.randn_like(t)

    dW = noise * dt.sqrt().unsqueeze(-1)
    W = dW
    W[:,:,0] = 0
    W = W.cumsum(-1)

    BB = W - t * (W[:,:,-1]).unsqueeze(-1)
    
    return BB, t

def excursion(t, neg=True, N=2, noise=None, sigma=1):
    '''
    Simulates excursions from a brownian bridge.
    '''
    if len(t.shape) > 2:
        bb, t  = brownian_bridge_nd(t, noise=noise)
    else:
        bb, t  = brownian_bridge(t, N=N, noise=noise, sigma=sigma)
    m, idx = bb.min(-1)
    if len(t.shape) > 2:
        t_rep = t
    else:
        t_rep = t.unsqueeze(0).repeat(N, 1, 1)
    ini = torch.arange(bb.shape[0])
    inj = torch.arange(bb.shape[1])
    ij  = torch.meshgrid(ini, inj, indexing='ij')

    nt = ( t_rep[ij[0], ij[1], idx].unsqueeze(-1) + t_rep ) % 1
    j  = torch.floor(nt * t.shape[-1]).long()

    j[j<0] = 0 

    BE = (bb.gather(-1,j) - m.unsqueeze(-1))

    if neg:
        if len(t.shape) > 2:
            bernoulli = torch.randint(2, (t.shape[0],t.shape[1])) * 2 - 1
            BE = BE * bernoulli.unsqueeze(-1)
        else:
            bernoulli = torch.randint(2, (t.shape[0],1)) * 2 - 1
            BE = BE * bernoulli

    return BE, bb

def get_log_mixture(N):

    mix_param = nn.Parameter(torch.ones(N,))
    loc_param = nn.Parameter(torch.rand(N,))
    scale_param = nn.Parameter(torch.rand(N,))

    mix  = D.categorical.Categorical(mix_param)
    comp = D.log_normal.LogNormal(loc_param, scale_param)
    lmm  = D.mixture_same_family.MixtureSameFamily(mix, comp)

from scipy import stats
class GammaMixture():

    def _ppf(self, q, *args, **kwds):
        print(q)
        mix = D.categorical.Categorical(torch.ones(2,))
        comp = D.gamma.Gamma(torch.tensor([2.0, 7.5]), torch.tensor([2.0, 1.0]))
        m = D.mixture_same_family.MixtureSameFamily(mix, comp)
        return m.icdf(torch.tensor(q))
