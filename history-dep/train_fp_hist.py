from local_tpp import LocalTimePP
from data import setup_data, get_data
import utils
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from nets import MLP

# import statsmodels.api as sm

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
# plt.style.use('seaborn-darkgrid')

import numpy as np
from scipy import stats

from nets import SplineRegression, SplineRegressionHistory

import pickle

DEVICE = 'cpu'

width = 36
layers = 3

gamma = lambda x:  (x ** 4) * torch.exp(- x ) / 24
exponential = lambda x : torch.exp(-x)
weibull = lambda x : 1.5 * torch.sqrt(x) * torch.exp(-x**1.5)


def train_historical(
        d : int,
        t  : torch.tensor,
        t0 : torch.tensor,
        xt : torch.tensor = None,
        pp : torch.tensor = None,
        history_kernel = None,
        m  = None,
        s  : type(lambda t,x : x) = None,
        loss : str = 'mle',
        use_curve : bool = False,
        validation = None, 
        true_pdf : type(lambda t : t) = None,
        param_model : str = None,
        use_bridge : bool = False):
    """"""
    epochs = 200
    n_print = 5

    assert loss=='mle' or loss=='mse', 'Loss type not defined.'

    if t0 == None:
        assert true_pdf is not None and param_model is not None, 'Must supply pdf and model name'

    t.requires_grad = True
    use_time = False

    history = pp[:,0]
    # knots = [xt.min(), -0.2, 0, 0.2, xt.max()]
    # knots = 5
    knots = utils.allocate_knots(xt, 5, verbose=False)
    mu = SplineRegressionHistory(input_range=[xt.min(), xt.max()],
        history_kernel=history_kernel, knots=5, order=3, history_order=1)
    print('-'*120)
    lr_mu = 0.01
    l2_lambda = 0.002

    eps = nn.Parameter(torch.zeros(1))
    lr_eps = 0.0

    eps = nn.Parameter(1e-0*torch.ones(1))
    lr_eps = 0.005

    sigma = MLP(d, width, layers, d, bias=True).to(DEVICE)
    nn.init.zeros_(sigma.out.weight.data)
    nn.init.ones_(sigma.out.bias.data)
    lr_sigma = 0

    opt_params = [{'params': mu.parameters(), 'lr': lr_mu},
            {'params': [eps], 'lr': lr_eps}]

    if use_curve:
        pass
    else:
        x0 = torch.zeros_like(t)
    if xt is None:
        xt = torch.linspace(-1,1, 100)

    if t0 is not None:
        # if we're using MLE then we getobservations
        dt0_data = []

        # Prepare the validation data for plotting
        if type(t0) == list or type(t0) == tuple:
            for t0s in t0:
                dt0_data.append(t0s[1:] - t0s[:-1])
            dt0_data  = torch.cat(dt0_data)
        else:
            dt0_data  = (t0[1:] - t0[:-1])
        cdf_data = dt0_data.sort()[0]

    # Prepare the model
    tpp = LocalTimePP(mu, sigma, t, t0, x0, history=history, use_bridge=use_bridge,
        use_time=use_time, mu_type='state_history')

    opt = optim.AdamW(opt_params, weight_decay=0.0)
    best_loss = np.inf
    mse = np.inf
    loss_list = []

    #pbar = tqdm(range(epochs), ncols=100, file=sys.stdout)
    for epoch in range(epochs):
        opt.zero_grad()

        loss_e = -tpp.girsanov_history(eps=eps, neg=False, K=400)

        loss_e.backward()
        opt.step()
        #pbar.set_description(f"Epoch {epoch} | NLL {loss_e.item():.3f}  eps {eps.item():.3f} w {mu.history_linear.weight[0].item():.3f}")
        loss_list.append(loss_e.item())

        if (epoch + 1) % n_print == 0:

            with torch.no_grad():
                if loss_e < best_loss:
                    best_loss = loss_e.clone()
                    
    plt.figure()
    plt.plot(loss_list)
    plt.title('Loss')
    return mu


def train_stimulus(
        d : int,
        t  : torch.tensor,
        t0 : torch.tensor,
        xt : torch.tensor = None,
        pp : torch.tensor = None,
        stim : torch.tensor = None,
        stim_kernel = None,
        m  : type(lambda t,x : x) = None,
        s  : type(lambda t,x : x) = None,
        loss : str = 'mle',
        use_curve : bool = False,
        validation = None, 
        true_pdf : type(lambda t : t) = None,
        param_model : str = None,
        use_bridge : bool = False):
    """"""
    epochs = 200
    n_print = 5

    assert loss=='mle' or loss=='mse', 'Loss type not defined.'

    if t0 == None:
        assert true_pdf is not None and param_model is not None, 'Must supply pdf and model name'

    t.requires_grad = True
    use_time = False

    # knots = [xt.min(), -0.2, 0, 0.2, xt.max()]
    # knots = 5
    knots = utils.allocate_knots(xt, 5, verbose=False)
    mu = SplineRegressionHistory(input_range=[xt.min(), xt.max()],
        history_kernel=stim_kernel, knots=5, order=3, history_order=1)
    print('-'*120)
    lr_mu = 0.01
    l2_lambda = 0.002


    eps = nn.Parameter(torch.zeros(1))
    lr_eps = 0

    sigma = MLP(d, width, layers, d, bias=True).to(DEVICE)
    nn.init.zeros_(sigma.out.weight.data)
    nn.init.ones_(sigma.out.bias.data)
    lr_sigma = 0

    opt_params = [{'params': mu.parameters(), 'lr': lr_mu},
            {'params': [eps], 'lr': lr_eps}]

    if use_curve:
        pass
    else:
        x0 = torch.zeros_like(t)
    if xt is None:
        xt = torch.linspace(-1,1, 100)

    if t0 is not None:
        # if we're using MLE then we getobservations
        dt0_data = []

        # Prepare the validation data for plotting
        if type(t0) == list or type(t0) == tuple:
            for t0s in t0:
                dt0_data.append(t0s[1:] - t0s[:-1])
            dt0_data  = torch.cat(dt0_data)
        else:
            dt0_data  = (t0[1:] - t0[:-1])
        cdf_data = dt0_data.sort()[0]

    # Prepare the model
    tpp = LocalTimePP(mu, sigma, t, t0, x0, stim=stim, use_bridge=use_bridge,
        use_time=use_time, mu_type='state_stim')

    opt = optim.AdamW(opt_params, weight_decay=0.0)
    best_loss = np.inf
    mse = np.inf
    loss_list = []

    #pbar = tqdm(range(epochs), ncols=100, file=sys.stdout)
    for epoch in range(epochs):
        opt.zero_grad()

        loss_e = -tpp.girsanov_stim(eps=eps, neg=False, K=400)

        # L2 loss.
        for name, par in mu.named_parameters():
            if 'state' in name and 'weight' in name:
                loss_l2 = torch.square(par).sum()

        loss_e.backward()
        opt.step()
        #pbar.set_description(f"Epoch {epoch} | NLL {loss_e.item():.3f}  eps {eps.item():.3f} w {mu.history_linear.weight[0].item():.3f}")
        loss_list.append(loss_e.item())

        if (epoch + 1) % n_print == 0:

            with torch.no_grad():
                if loss_e < best_loss:
                    best_loss = loss_e.clone()

    return mu
    
def run_hist_dep_exp(m, s, weight=2, n_runs=5, use_bridge=False):
    
    from data import setup_data, get_data_history_dependent
    
    def kernel(delta, sigma=1):
        return torch.exp(-delta / sigma)
    
    def m_full(t, x, history):
        drift = m(t,x)
        for i in range(1, 1+1):
            drift -= weight * kernel(t-history[-i])
        return drift

    dt = 0.02
    N  = 1000
    
    use_curve = False
    all_mse = []
    
    loss = 'mle'

    for idx in range(n_runs):

        try:

            with open('hist_w={}_r={}.p'.format(weight, idx), 'rb') as f:
                data = pickle.load(f)

        except:

            data = get_data_history_dependent(m_full, s, dt=dt, N=N, offset=0)
            with open('hist_w={}_r={}.p'.format(weight, idx),'wb') as f:
                pickle.dump(data, f)

        t, t0, t0_orig, xt, pp, a, b = setup_data(data, use_curve=use_curve)
        plt.plot(t, xt)
        plt.scatter(t0, np.zeros_like(t0))
        plt.savefig('data_path_hist_{}.pdf'.format(idx))
        plt.close('all')
        mu = train_historical(1, t, t0_orig, xt, pp, kernel, m_full, s, loss=loss, use_curve=use_curve, use_bridge=use_bridge)
        mse = F.mse_loss(mu.history_linear.weight[0], torch.ones(1)*weight).item()
        all_mse.append(mse)
        
    all_mse = np.stack(all_mse)

    print(all_mse)
    
    return all_mse


def run_input_dep_exp(m, s, weight=2, n_runs=5, use_bridge=False):
    
    from data import setup_data, get_data_input_dependent
    
    def kernel(delta, sigma=1):
        return torch.exp(-delta / sigma)
    
    def m_full(t, x, history):
        drift = m(t,x)
        for i in range(1, 1+1):
            drift += weight * kernel(t-history[-i])
        return drift
    
    dt, N = 0.02, 1000
    all_mse = []
    for idx in range(n_runs):

        stim = 2*torch.rand(100) * dt * N 
        stim, _ = torch.sort(stim)

        try:

            with open('exo_w={}_r={}.p'.format(weight, idx), 'rb') as f:
                data = pickle.load(f)

        except:

            data = get_data_input_dependent(m_full, s, stim, dt=dt, N=N, offset=0)
            with open('exo_w={}_r={}.p'.format(weight, idx),'wb') as f:
                pickle.dump(data, f)

        use_curve = False
        t, t0, t0_orig, xt, pp, a, b = setup_data(data, use_curve=use_curve)
        plt.plot(t, xt)
        plt.scatter(t0, np.zeros_like(t0))
        plt.savefig('data_path_exo_{}.pdf'.format(idx))
        plt.close('all')
        loss = 'mle'
        mu = train_stimulus(1, t, t0_orig, xt, pp, stim, kernel, m, s, loss=loss, use_curve=use_curve, use_bridge=use_bridge)
        mse = F.mse_loss(mu.history_linear.weight[0], torch.ones(1)*weight).item()
        all_mse.append(mse)
        
    all_mse = np.stack(all_mse)
    print(all_mse)
    return all_mse

if __name__ == '__main__':
    m = lambda t, x: -x
    s = lambda t, x: 1

    print('Running: History Dependent')
    for weight in [0.5, 1, 2]:
        for b in [0, 1]:
            mse = run_hist_dep_exp(m, s, use_bridge=b, weight=weight, n_runs=10)
            print('+'*100)
            print('New History Dependent Result')
            print('Use Bridge? : {}'.format(bool(b)))
            print('Kernel weight: {}'.format(weight))
            print('MSE: {} \pm {}'.format(mse.mean(), mse.std()))

    print('Running: Input Dependent')
    for weight in [0.5, 1, 2]:
        for b in [0, 1]:
            mse = run_input_dep_exp(m, s, use_bridge=b, weight=weight, n_runs=10)
            print('+'*100)
            print('New Input Dependent Result')
            print('Use Bridge? : {}'.format(bool(b)))
            print('Kernel weight: {}'.format(weight))
            print('MSE: {} \pm {}'.format(mse.mean(), mse.std()))

