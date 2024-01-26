import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from nets import MLP
# import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import scipy
from scipy import stats

import utils
from data import setup_data, get_data_input_dependent

s = lambda t, x: 1

def kernel(delta, sigma=.05):
     return torch.exp(-delta / sigma)

def m_full(t, x, history):
    drift = -x
    for i in range(1, 1+1):
        drift += 1 * kernel(t-history[-i])
    return drift

def m(t, x):  # For state part, only used  for plotting later.
    drift = -x
    return drift

dt, N = 0.05, 500
stim = torch.rand(60) * dt * N 
stim, _ = torch.sort(stim)

data = get_data_input_dependent(m_full, s, stim, dt=dt, N=N, offset=0)

use_curve = False
t, t0, t0_orig, xt, pp, a, b = setup_data(data, use_curve=use_curve)

from train_fp import train_stimulus

use_curve = False
loss = 'mle'
mu = train_stimulus(1, t, t0, xt, pp, stim, kernel, m, s, loss=loss, use_curve=use_curve)
