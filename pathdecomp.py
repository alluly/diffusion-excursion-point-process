import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
 
import numpy as np
import torch

from utils import excursion 

plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 20})

m = lambda t, x: -np.tanh(x)
 
dt = 0.01
N = 1000
T = dt * N
print(T)
 
ts = np.linspace(0, T, N)
bm = np.zeros_like(ts)

excolor = '#4a4e4d'
linecolor = '#00A19B'
markerline= '#3da4ab'
markercolor= '#FFEA19'
 
for idx, t in enumerate(ts):
    if idx == N - 1:
        break 
    bm[idx + 1] = bm[idx] + m(t, bm[idx]) * dt + np.sqrt(dt) * np.random.randn(1)
 

zero_inds = (bm[1:] * bm[:-1]) < 0
 
zi = np.arange(zero_inds.shape[0])[zero_inds[:]]
zm1 = 0  
for z in zi:
    neg = bm[zm1:z].mean() < 0
    if neg:
        plt.plot(ts[zm1:z], bm[zm1:z], color='tab:red', alpha=0.8)
    else:
        plt.plot(ts[zm1:z], bm[zm1:z], color='tab:blue', alpha=0.8)
    zm1 = z

plt.plot(ts[zm1:], bm[zm1:])
plt.plot(ts, bm, '--', alpha=0.1)
plt.xlabel(r'$t$')
plt.ylabel(r'$Z_t$')
plt.tight_layout()
plt.savefig('figure_path_decomp.pdf')  
plt.close('all')

zm1 = 0
for z in zi:
    length = torch.tensor(ts[zm1:z]).unsqueeze(0)
    #plt.plot(ts[zm1:z], bm[zm1:z])
    if length.shape[-1] > 1:
        neg = bm[zm1:z].mean() < 0
        e = excursion(length, N=100, neg=neg)[0]
        for idx in range(e.shape[0]):
            plt.plot(length[0], e[idx,0], alpha=0.2, color=excolor)
            #plt.plot(length[0], e[idx,0], alpha=0.1, color='#565F64')
    zm1 = z

length = torch.tensor(ts[zm1:]).unsqueeze(0)
if length.shape[-1] > 1:
    neg = bm[zm1:].mean() < 0
    e = excursion(length, N=100, neg=False)[0]
    if neg:
        e = e*-1
    for idx in range(e.shape[0]):
        plt.plot(length[0], e[idx,0], alpha=0.2, color=excolor)
        #plt.plot(length[0], e[idx,0], alpha=0.1, color='#565F64')
    plt.plot(length[0], e[idx,0], alpha=0.2, color=excolor, label='Excursion Paths')
plt.plot(ts, bm, '--',color=linecolor, alpha=1, label='True Path')
plt.scatter(ts[1:][zero_inds], np.zeros_like(ts[1:][zero_inds]), color=markercolor, label='Arrival Times', s=30, zorder=10000, edgecolors=markerline)
plt.xlabel(r'$t$')
plt.ylabel(r'$Z_t$')
plt.tight_layout()
plt.legend()
plt.savefig('figure_path_dist.pdf')  


