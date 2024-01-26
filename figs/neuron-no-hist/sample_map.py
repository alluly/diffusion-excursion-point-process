import pickle

import torch

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 25})

import numpy as np
start_idx = 1
with open('stim_and_z.p','rb') as f:
    data = pickle.load(f)

stim = data['stim']
t_stim = data['stim_time']
dt = t_stim[1] - t_stim[0]
path   = (( data['paths'].squeeze().abs()).log().mean(0))
#path   = (data['paths'].squeeze() + 1e-2).cumsum(1).mean(0)
dstim = ( stim[1:] - stim[:-1] ) / dt
stim = (stim * (np.concatenate((dstim, np.zeros(1))) > 0))
t_path = data['paths_time']


points = [1, 500, 1000, 1500, 2000]

for idx in range(1,len(points)):
    print(idx)
    start_idx = points[idx-1]
    end_idx   = points[idx]

    print(start_idx)
    print(end_idx)

    #path = np.cumsum(np.abs(path[1:] - path[:-1]))
    app = torch.ones_like(path)
    A = torch.stack((path, app), -1).numpy()[start_idx:end_idx]
    b = stim[start_idx:end_idx]

    #b = np.maximum(stim[start_idx:],0)
    #b = np.cumsum(np.abs(stim[1:] - stim[:-1]))

    x, r, _, _ = np.linalg.lstsq(A, b)
    plt.figure(figsize=(20,6))
    #plt.plot(t_stim[1:], stim[1:], alpha=0.5)
    plt.plot(t_path[start_idx:end_idx], A @ x, alpha=0.75, color='salmon', label='Mean Transformed Sample')
    plt.plot(t_stim[start_idx:end_idx], b, alpha=0.75, color='dodgerblue', label='True Stimulus')
    #plt.plot(t_path[1:], path[1:])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\mathcal{S}(t)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('learned_stim_{}_notform_.pdf'.format(idx))
    plt.close('all')

    print(x)

