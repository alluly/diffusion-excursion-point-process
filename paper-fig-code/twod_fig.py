import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
from matplotlib.collections import LineCollection

import numpy as np
import torch

from sim import ito_diffusion_nd

plt.rcParams.update({'font.size': 40})

def mu(t,x):
    return torch.stack((-x[0] - 5 * x[1], -x[1] + 5 * x[0]))

def sigma(t, x):
    return 1

dt = 0.0001
N = 30000
xt, t, sn, sm = ito_diffusion_nd(mu, sigma, dt, N, torch.randn(2))
zeros = ( xt[1:,:] * xt[:-1,:] ) < 0
mult = ( xt[1:,:] * xt[:-1,:] ) 
zero_inds = np.logical_or(mult[:,0] < 0, mult[:,1] < 0)
xt_small = xt[:-1]
xt_nz = xt_small[zero_inds,:].nonzero()

xt_z = xt_small[zero_inds,:][xt_nz]

xt_z = xt_small[zero_inds,:]
points = xt.reshape(-1,1,2).numpy()
segs = np.concatenate([points[:-1],points[1:]],axis=1)

lc = LineCollection(segs, cmap=plt.get_cmap('inferno'), alpha =0.31)
lc.set_array(t)
plt.figure(figsize=(18,9))
plt.title(r'Point Process by $X_t$') 
plt.subplot(121)
plt.gca().add_collection(lc)
plt.xlim(xt[:,0].min(), xt[:,0].max())
plt.ylim(xt[:,1].min(), xt[:,1].max())
plt.scatter(np.zeros_like(xt[:-1,0][zeros[:,0]]), xt[:-1,1][zeros[:,0]], color='white',  alpha =0.8, label='Class A', marker='^')
plt.scatter(xt[:-1,0][zeros[:,1]], np.zeros_like(xt[:-1,1][zeros[:,1]]), color='black',   alpha =0.8, label='Class B', marker='>')
#plt.legend()

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.subplot(122)

plt.scatter(t[:-1][zeros[:,0]], ['Class A'] * len(zeros[:,0].nonzero()), color='white', marker='^', s=800*torch.ones_like(zeros[:,0].nonzero()), alpha=0.5)
plt.scatter(t[:-1][zeros[:,1]], ['Class B'] * len(zeros[:,1].nonzero()), color='black', marker='>', s=800*torch.ones_like(zeros[:,1].nonzero()), alpha=0.5)
plt.xlabel(r'$t$')
plt.tight_layout()
plt.savefig('example_2d.pdf')
