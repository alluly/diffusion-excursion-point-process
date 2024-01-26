import scipy.io
import torch
import pickle
import numpy as np

dt = 0.001

spike_trainFile = 'MC.mat'
stim_file = 'stim.mat'

stim = scipy.io.loadmat(stim_file)
stim = np.array(stim['stim']).reshape(-1)
stim = np.insert(stim, 0, 0)

MC = scipy.io.loadmat(spike_trainFile)
MC = MC['MC'].reshape(-1)

neuron_index = 17
spikeTimes = MC[neuron_index][0].reshape(-1)
spikeIndices = MC[neuron_index][1].reshape(-1)
Trial_Index = range(max(spikeIndices))
print(f'Number of neurons {len(MC)}, each has {spikeIndices.max()} repeated trials.')

stim_len = len(stim)
stim_time = np.linspace(0, stim_len*dt, stim_len)

spk = spikeTimes[spikeIndices == Trial_Index[0]+1] * dt
spk_val = spikeTimes[spikeIndices == Trial_Index[1]+1] * dt
all_hspk = []
for ii in Trial_Index:
    t_spk = torch.tensor(spikeTimes[spikeIndices == Trial_Index[ii]+1] * dt)
    all_hspk.append(torch.histc(t_spk, bins=stim_len , min=0, max=stim_len*dt))
all_hspk = torch.stack(all_hspk)

d = {'stim' : stim, 
     'spk'  : spk,
     'Trial_Index' : Trial_Index,
     'spikeTimes' : spikeTimes,
     'spikeIndices' : spikeIndices,
     'stim_time' : stim_time,
     'all_hspk'  : all_hspk}

with open('neuron_data.p', 'wb') as f:
    pickle.dump(d, f)

