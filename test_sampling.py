import torch
from local_tpp import LocalTimePP

from data import setup_data, get_data

import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nets import MLP, SplineRegression

import statsmodels.api as sm
import pingouin as pg

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

import numpy as np
from scipy import stats


def sample(t0, t, param_model, true_pdf, validation = None):
    loss='sampling'
    path = 'figs/{}/tpp.p'.format(param_model)

    d = 1
    DEVICE='cpu'

    width =16
    layers=2
    use_time = True

    if use_time:
        mu = MLP(d+1, width, layers, d, bias=True).to(DEVICE)
    else:
        mu = SplineRegression(input_range=[-4, 4])
        #mu = MLP(d, width, layers, d, bias=True).to(DEVICE)
    nn.init.zeros_(mu.out.weight.data)
    nn.init.constant_(mu.out.bias.data, 1e-3)

    sigma = MLP(d, width, layers, d, bias=True).to(DEVICE)
    nn.init.zeros_(sigma.out.weight.data)
    nn.init.ones_(sigma.out.bias.data)

    lr_mu = 1e-3
    if isinstance(mu, SplineRegression):
        lr_mu = 1e-2
    lr_sigma = 0
    lr_eps   = 1e-3

    x0 = torch.zeros_like(t)
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

    eps = nn.Parameter(1e0*torch.ones(1))

    # Prepare the model
    tpp = LocalTimePP(mu, sigma, t, t0, eps, x0, use_bridge=False, use_time=use_time, prior=None)
    tpp.load_state_dict(torch.load(path))
    tpp.eval()

    print('Epsilon')
    print(tpp.eps)

    with torch.no_grad():

        t0, dt0, paths, t_p = tpp.sample(5000,20000,dt0_data.max(),1.5*(tpp.eps).abs().item())
        print('MLE est {:.4f}'.format((1/dt0.mean()).item()))
        print('KS Test')
        print(stats.ks_2samp(dt0, dt0_data))

        # plot p(t)
        logp_noise = torch.randn((100,1,500))
        plot_list = []
        for t_ in t:
            log_p_hat, t_new = tpp.logp(t_,  noise=logp_noise)
            p_hat = log_p_hat.exp()
            plot_list.append(p_hat.detach().item())
        plot_  = np.array(plot_list) 

        # plot histogram
        kde = stats.gaussian_kde(dt0.numpy())
        
        plt.hist(dt0.numpy(), bins=200, histtype='step', density=True)
        plt.plot(t.detach(), kde(t.detach()), label='KDE')
        plt.plot(t.detach(), plot_, label='Model')
        if true_pdf is not None:
            plt.plot(t.detach(), true_pdf(t).detach(), label='Truth')
        plt.legend()
        plt.tight_layout()
        plt.savefig('figs/{}/hist_samples_{}_ts.pdf'.format(param_model, loss))
        plt.close('all')


        # plot some paths
        for path_ind in range(5):
            plt.plot(t_p, paths[path_ind], alpha=0.4)
        plt.plot(t_p, eps.item()*torch.ones_like(t_p), label=r'$\epsilon$')
        plt.tight_layout()
        plt.savefig('figs/{}/paths_{}_ts.pdf'.format(param_model, loss))
        plt.close('all')

        # plot qq plot
        if param_model is not None:
            if param_model == 'exp':
                sm.qqplot(dt0.numpy(), stats.expon, line='45', fit=False)
                plt.tight_layout()
                plt.savefig('figs/{}/qq_{}_nc_ts.pdf'.format(param_model, loss))
                plt.close('all')
                pg.qqplot(dt0.numpy(), stats.expon)
            elif param_model == 'gamma':
                sm.qqplot(dt0.numpy(), stats.gamma, line='45', distargs=(9,), fit=False)
                plt.tight_layout()
                plt.savefig('figs/{}/qq_{}_nc_ts.pdf'.format(param_model, loss))
                plt.close('all')
                pg.qqplot(dt0.numpy(), stats.gamma, sparams=(9,))
            elif param_model == 'weibull':
                sm.qqplot(dt0.numpy(), stats.weibull_min, line='45', distargs=(1.5,), fit=False)
                plt.tight_layout()
                plt.savefig('figs/{}/qq_{}_nc_ts.pdf'.format(param_model, loss))
                plt.close('all')
                pg.qqplot(dt0.numpy(), stats.weibull_min, sparams=(1.5,))
            plt.tight_layout()
            plt.savefig('figs/{}/qq_{}_ts.pdf'.format(param_model, loss))
            plt.close('all')

            if validation is not None:
                all_log_p = []
                if type(validation) == list:
                    for t0_ in validation:
                        tdiff = t0_[1:] - t0_[:-1]
                        for t0 in tdiff:
                            if torch.isnan(tpp.logp(t0)[0]):
                                print('Error, t={} < min t'.format(t0.item))
                                print(t0)
                            all_log_p.append(tpp.logp(t0)[0])
                else:
                    tdiff = validation[1:] - validation[:-1]
                    for t0 in tdiff:
                        if torch.isnan(tpp.logp(t0, eps=eps)[0]):
                            print('Error, t={} < min t'.format(t0.item))
                            print(t0)
                        all_log_p.append(tpp.logp(t0)[0])
                print('{} Validation LL'.format(param_model))
                print(torch.cat(all_log_p).mean().item() )
                print((torch.cat(all_log_p) ).std().item())

param_model = 'exp'
def load_experiment(synth_type):
    if synth_type == 'exp':
        m = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
    elif synth_type == 'gamma':
        m = torch.distributions.gamma.Gamma(torch.tensor([9.0]), torch.tensor([1.0]))
    elif synth_type == 'log_normal':
        m = torch.distributions.log_normal.LogNormal(torch.tensor([-1.8]), torch.tensor([1.9]))
    elif synth_type == 'weibull':
        m = torch.distributions.weibull.Weibull(torch.tensor([1.0]), torch.tensor([1.5]))
    elif synth_type == 'mix-gamma':
        mix = torch.distributions.categorical.Categorical(torch.ones(2,))
        comp = torch.distributions.gamma.Gamma(torch.tensor([2.0, 7.5]), torch.tensor([2.0, 1.0]))
        m = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)

    t0_train_ = m.sample((20,10))
    t0_train = []
    for idx in range(t0_train_.shape[0]):
        cs = t0_train_[idx].cumsum(0).squeeze()
        a = torch.zeros(cs.shape[0]+1)
        a[1:] = cs
        t0_train.append(a)

    t0_val_ = m.sample((100,10))
    t0_val = []
    for idx in range(t0_val_.shape[0]):
        cs = t0_val_[idx].cumsum(0).squeeze()
        a = torch.zeros(cs.shape[0]+1)
        a[1:] = cs
        t0_val.append(a)

    print('Validation LL')
    print(m.log_prob(t0_val_).mean().item())
    t = torch.linspace(1e-6,t0_train_.max(),100)
    true_pdf = lambda t: m.log_prob(t).exp()
    sample(t0_train, t, synth_type, true_pdf, validation=t0_val)

load_experiment(param_model)

