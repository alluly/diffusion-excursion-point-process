from local_tpp import LocalTimePP

from data import setup_data, get_data

import utils

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nets import MLP, SplineRegression, HistoryMLP

import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 20})

import os

import numpy as np

from scipy import stats

torch.manual_seed(0)
np.random.seed(0)

DEVICE = 'cpu'

width =  16
layers = 4 

gamma = lambda x:  (x ** 4) * torch.exp(- x ) / 24
exponential = lambda x : torch.exp(-x)
weibull = lambda x : 1.5 * torch.sqrt(x) * torch.exp(-x**1.5)

def train(d : int, 
        t  : torch.tensor, 
        t0 : torch.tensor, 
        xt : torch.tensor = None, 
        m  : type(lambda t,x : x) = None, 
        s  : type(lambda t,x : x) = None, 
        loss : str = 'mle',
        use_curve : bool = False,
        validation = None, 
        true_pdf : type(lambda t : t) = None,
        param_model : str = None, 
        run_num : int = None,
        use_time : bool = True, 
        n_print : int = 1000, 
        warm_epochs : int = 2000, 
        epochs : int = 10001,
        prior_type : str = None,
        marks = None, 
        stim = None, 
        save_best_loss = True,
        val_callback = None,
        act = nn.Softplus(), 
        width = 16,
        layers = 4
        ):

    '''
    Training routine. Takes in time horizon, arrival times, and state space.

    d int -- integer describing dimensionality of the problem.
    t tensor  -- tensor describing time horizon
    t0 tensor or list of tensors -- tensor describing crossing times
                                 -- if t0 is a list, then each tensor in the list is considered a sample

    xt tensor -- tensor describing state space (optional)

    m function -- function for mu (optional)
    s function -- function for sigma (optional)

    loss string -- string determining which loss to use (optional)

    use_curve boolean -- flag to use a curve or not (optional)

    validation tensor or list of tensors -- data to validate experimentss (optional)

    true_pdf function -- function describing the true PDF for parametric cases

    param_model string -- underlying parametric model
    run_num int -- batch number

    stim tensor tuple -- stimulus parameter

    val_callback function -- function to run during validation
    '''

    assert loss == 'reg-mle' or loss == 'reg-elbo' or loss=='mle' or loss=='mse' or loss=='diff', 'Loss type not defined.'

    if t0 == None:
        assert true_pdf is not None and param_model is not None, 'Must supply pdf and model name'

    if use_time:
        mu = MLP(d+1, width, layers, d, bias=True, act=act).to(DEVICE)
    else:
        #mu = SplineRegression(input_range=[-4, 4])
        mu = MLP(d, width, layers, d, bias=True, act=act, use_pe=0).to(DEVICE)

    lr_sigma=0

    if stim is not None:

        warm_epochs = 1

        mu = HistoryMLP(stim[0], stim[1], d, width, layers, d, bias=True).to(DEVICE)
        nn.init.zeros_(mu.out.weight.data)
        nn.init.zeros_(mu.out.bias.data)

        sigma = HistoryMLP(stim[0], stim[1], d, width, layers, d, bias=True, in_x=False).to(DEVICE)
        nn.init.zeros_(sigma.out.weight.data)
        nn.init.ones_(sigma.out.bias.data)

        sigma_param = {'params': sigma.parameters(), 'lr': lr_sigma}

        if 'no-hist' in param_model:
            print('Warning: Using a stimulus dataset without the stimulus.')

            mu = HistoryMLP(stim[0], stim[1], d, width, layers, d, bias=True, use_hist=False, pe=10).to(DEVICE)
            nn.init.zeros_(mu.out.weight.data)
            nn.init.zeros_(mu.out.bias.data)

            sigma = HistoryMLP(stim[0], stim[1], d, width, layers, d, bias=True, in_x=False, use_hist=False, pe=10).to(DEVICE)
            nn.init.zeros_(sigma.out.weight.data)
            nn.init.ones_(sigma.out.bias.data)
            lr_sigma = 0

            sigma_param = {'params': sigma.parameters(), 'lr': lr_sigma}
    else:
        sigma = MLP(d, width, layers, d, bias=True).to(DEVICE)
        nn.init.zeros_(sigma.out.weight.data)
        nn.init.ones_(sigma.out.bias.data)
        sigma = nn.Parameter(torch.ones(1, requires_grad=True)).to(DEVICE)
        sigma_param = {'params': [sigma], 'lr': lr_sigma}


    pytorch_total_params = sum(p.numel() for p in mu.parameters())
    print('Mu params {}'.format(pytorch_total_params))

    dstats = {'epoch':[], 'ks' : [], 'p':[]}

    if 'eda' in param_model:
        lr_mu = 2e-3
    elif 'welltory' in param_model:
        lr_mu = 1e-2
        epochs = 3001
        warm_epochs = 2000
        n_print = 500
    else:
        lr_mu = 1e-3 
    if isinstance(mu, SplineRegression):
        lr_mu = 1e-2

    if stim is not None:
        lr_eps  = 0
        lr_sigma = 1e-4
    else:
        lr_eps  = 1e-2
        lr_sigma = 0

        sigma_param = {'params': [sigma], 'lr': lr_sigma}
    ks = None # save the ks

    print('Parameters:')
    print('lr_mu: {}'.format(lr_mu))
    print('epochs: {}'.format(epochs))
    print('warmup epochs: {}'.format(warm_epochs))
    print('print iterations: {}'.format(n_print))
    print('lr_eps: {}'.format(lr_eps))
    print('lr_sigma: {}'.format(lr_sigma))
    print('prior_type: {}'.format(prior_type))

    eps = nn.Parameter(1e-2*torch.ones(1))


    opt_params = [{'params': mu.parameters(), 'lr': lr_mu},
            {'params': [eps], 'lr': lr_eps}]
    opt_params.append(sigma_param)

    if use_curve:
        pass
    else:
        x0 = torch.zeros_like(t)
    if xt is None:
        xt = torch.linspace(-1,1,100)

    if t0 is not None:
        # if we're using MLE then we get observations
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
    tpp = LocalTimePP(mu, sigma, t, t0, eps, x0, use_bridge=False, use_time=use_time, prior=prior_type, param_model=param_model, marks=marks)

    opt = optim.AdamW(opt_params)

    best_loss = np.inf 
    mse = np.inf

    if prior_type is not 'diffusion':
        for warm_iter in range(warm_epochs):
            opt.zero_grad()
            loss_p = -tpp.init_prior()
            loss_p.backward()
            opt.step()

    print('Final Eps: {:.4f}'.format(eps.item()))

    if type(validation) == list:
        val_ten = []
        for t0_ in validation:
            tdiff = t0_[1:] - t0_[:-1]
            val_ten.append(tdiff)
        val_ten = torch.cat(val_ten)

    for epoch in range(epochs):
        opt.zero_grad()

        if loss == 'mle':
            loss_e = -tpp.girsanov(neg=False)
            
        elif loss == 'diff':
            loss_e = -tpp.girsanov(neg=True)

        elif loss == 'mse':
            loss_e = 0
            K = 1000
            noise = torch.randn((K, 1, 200)) # fix the noise
            for t_ in t: 
                log_p_hat, t_new = tpp.logp(t_,  noise=noise, prior_type=None)
                p_hat = log_p_hat.exp()

                loss_e += F.mse_loss(true_pdf(t_), p_hat.float().squeeze())
        elif loss == 'reg-elbo':
            loss_l = 0
            mass = 0 
            K = 1000
            noise = torch.randn((K, 1, 200)) # fix the noise
            loss_l = -tpp.girsanov(neg=False)

            for tidx,_ in enumerate(t): 
                log_p_hat_m, t_new = tpp.logp(t[tidx], prior_type=None, noise=noise)
                mass += log_p_hat_m.exp() * (t[1]-t[0])

            loss_e = loss_l + F.l1_loss(mass, torch.ones_like(mass))

        elif loss == 'reg-mle':
            loss_l = 0
            mass   = 0 
            K      = 200 # number of excursions
            noise = torch.randn((K, 1, 100)) # fix the noise
            for tidx, t_ in enumerate(dt0_data): 
                if tidx==0:
                    tlast = 1e-7
                log_p_hat, t_new = tpp.logp(t_, prior_type=None, neg=False, noise=noise, print_prior=False)
                tlast = t_
                loss_l -= log_p_hat
                if tidx < t.shape[0]:
                    # add the regularization factor to keep the PDF to integrate to 1.
                    log_p_hat_m, t_new = tpp.logp(1e-5+t[tidx], prior_type=None, noise=noise)
                    mass += log_p_hat_m.exp() * (t[1]-t[0])
                if torch.isnan(log_p_hat):
                    tpp.logp(t_, print_prior=True)
            loss_e = loss_l/dt0_data.shape[0] + F.l1_loss(mass, torch.ones_like(mass))

        loss_e.backward()
        opt.step()

        logp_noise = torch.randn((100,1,500))

        # Plotting / Validation code
        if (epoch ) % n_print == 0:
            print('Epoch number {}'.format(epoch))
            print('NLL {}'.format(loss_e.item()))
            print('Epsilon {}'.format(eps.item()))
            with torch.no_grad():
                if loss_e < best_loss or save_best_loss == False:

                    if save_best_loss == False:
                        print('NOT SAVING BEST LOSS')

                    best_loss = loss_e.clone()

                    # plot p(t)
                    plot_list = []
                    for t_ in t:
                        log_p_hat, t_new = tpp.logp(t_,  noise=logp_noise)
                        p_hat = log_p_hat.exp()
                        plot_list.append(p_hat.detach().item())

                    num_  = np.array(plot_list) 
                    norm_ = (num_.sum() * (t[1]-t[0])).clone()

                    plot_ = num_ 

                    plt.hist((dt0_data.numpy()).reshape(-1,1),bins=200,histtype='step',density=True,label='True Hist')
                    plt.plot(t.detach(), plot_, label='Estimated')
                    if true_pdf is not None:
                        plt.plot(t.detach(), true_pdf(t).detach(), label='Truth')

                    plt.xlabel(r'$t$')
                    plt.ylabel(r'$p(t)$')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig('figs/{}/p_t_{}.pdf'.format(param_model, loss))
                    plt.close('all')

                    xtp = torch.linspace(0, 4, 200).unsqueeze(1) # plot over the ordered space
                    if m is not None:
                        # Compute error with ground truth m
                        xtp = torch.linspace(xt.min(), xt.max(), 200).unsqueeze(1) # plot over the ordered space

                        muest  = mu(xt).detach()
                        mutrue = m(0,xt) * torch.ones_like(xt)
                        mse = F.mse_loss(muest, mutrue).item()

                        print('MSE {:4f}'.format(mse)) # compute MSE
                        print('Pct Err {:4f}'.format(100*((mutrue-muest)/mutrue.norm()).abs().mean().item())) # compute PctE

                        plt.plot(xtp, m(0,xtp) * torch.ones_like(xtp), label=r'$\mu$')

                    if use_time:
                        gx,gt = torch.meshgrid(xt, t, indexing='ij')
                        with torch.no_grad():
                            mu_grid = tpp.mu_t(gx.reshape(-1,1), gt.reshape(-1,1))
                            plt.grid(b=None)
                            plt.imshow(mu_grid.reshape(xt.shape[0],t.shape[0]),cmap='turbo', extent=(0, gt.max().item(), gx.min().item(), gx.max().item()), aspect='auto')
                            plt.colorbar()
                            plt.xlabel(r'$t$')
                            plt.ylabel(r'$X_t$')
                            plt.tight_layout()
                            plt.savefig('figs/{}/mut_grid.png'.format(param_model,loss))
                            plt.close('all')
                        plt.plot(xtp, tpp.mu_t(xtp, torch.zeros_like(xtp)).detach(), label=r'$\hat{\mu}$')
                        plt.plot(xtp, tpp.mu_t(xtp, torch.zeros_like(xtp)).detach() / norm_, label=r'$\hat{\mu} / Z$')
                    else:
                        plt.plot(xtp, mu(xtp).detach(), label=r'$\hat{\mu}$')
                        plt.ylabel(r'$\mu(x)$')

                    plt.legend()
                    plt.savefig('figs/{}/mu_est_{}.pdf'.format(param_model,loss))
                    plt.close('all')

                    if validation is not None and stim == None:
                        all_log_p = []
                        if type(validation) == list:
                            for t0_ in validation:
                                tdiff = t0_[1:] - t0_[:-1]
                                for t0__ in tdiff:
                                    if torch.isnan(tpp.logp(t0__)[0]):
                                        print('Error, t={} < min t'.format(t0__.item()))
                                        tpp.logp(t0__, print_prior=True)
                                    all_log_p.append(tpp.logp(t0__)[0])
                        else:
                            tdiff = validation #validation[1:] - validation[:-1]
                            for t0_ in tdiff:
                                if torch.isnan(tpp.logp(t0_)[0]):
                                    print('Error, t={} < min t'.format(t0_.item()))
                                    tpp.logp(t0_, print_prior=True)
                                all_log_p.append(tpp.logp(t0_)[0])
                        print('Norm')
                        print(norm_.item())
                        print('Validation LL')
                        alp = torch.cat(all_log_p)
                        print('{} \pm {}'.format(alp.mean().item(), alp.std().item()) )

                        if torch.isnan(alp.mean()):
                            return tpp, eps, ks

                    if m is None :

                        # sample and compute KS test
                        Tf = dt0_data.max()
                        use_first = True
                        if 'elbo' in loss:
                            use_first = False
                            if isinstance(t0, list):
                                Tf = t0[0].max()
                            else:
                                Tf = t0.max()
                            Tf = Tf / 2
                        Nsamp = 50000
                        dtsamp = Tf/Nsamp
                        
                        t0_, dt0, paths, t_p = tpp.sample(5000,100000,Tf,(eps).abs().item(), use_first=use_first)
                        print('MLE est {:.4f}'.format((1/dt0.mean()).item()))
                        print('KS Test')

                        if validation is not None:

                            if isinstance(validation, list):
                                ks = stats.ks_2samp(dt0.numpy()[:val_ten.shape[0]], val_ten.numpy())
                            else:
                                ks = stats.ks_2samp(dt0.numpy()[:validation.shape[0]], validation.numpy())
                            print(ks)

                            dstats['epoch'].append(epoch)
                            dstats['ks'].append(ks[0])
                            dstats['p'].append(ks[1])

                            ks_str = 'figs/{}/ks.txt'.format(param_model)

                            if run_num is not None:
                                ks_str = 'figs/{}/ks{}.txt'.format(param_model, run_num)

                            with open(ks_str, 'w') as convert_file:
                                convert_file.write(json.dumps(dstats))
                        else:
                            ks = 0

                        # plot histogram
                        kde = stats.gaussian_kde(dt0.numpy())
                        
                        plt.hist(dt0.numpy(), bins=200, histtype='step', density=True, label='Samples')
                        #plt.plot(t.detach(), kde(t.detach()), label='KDE')
                        plt.plot(t.detach(), plot_, label='Model')
                        if true_pdf is not None:
                            plt.plot(t.detach(), true_pdf(t).detach(), label='Truth')
                        plt.xlabel(r'$\tau$')
                        plt.ylabel(r'$p(\tau)$')
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig('figs/{}/hist_samples_{}.pdf'.format(param_model, loss))
                        plt.close('all')


                        # plot some paths
                        for path_ind in range(5):
                            plt.plot(t_p, paths[path_ind], alpha=0.4)
                        plt.plot(t_p, eps.item()*torch.ones_like(t_p), label=r'$\epsilon$')
                        plt.tight_layout()
                        plt.savefig('figs/{}/paths_{}.pdf'.format(param_model, loss))
                        plt.close('all')

                        # plot qq plot
                        if param_model is not None:
                            if 'exp' in param_model:
                                sm.qqplot(dt0.numpy(), stats.expon, line='45', fit=False)
                                plt.tight_layout()
                                plt.savefig('figs/{}/qq_{}_nc.pdf'.format(param_model, loss))
                                plt.close('all')
                            elif 'gamma' in param_model:
                                sm.qqplot(dt0.numpy(), stats.gamma, line='45', distargs=(9,), fit=False)
                                plt.tight_layout()
                                plt.savefig('figs/{}/qq_{}_nc.pdf'.format(param_model, loss))
                                plt.close('all')
                            elif 'weibull' in param_model:
                                sm.qqplot(dt0.numpy(), stats.weibull_min, line='45', distargs=(1.5,), fit=False)
                                plt.tight_layout()
                                plt.savefig('figs/{}/qq_{}_nc.pdf'.format(param_model, loss))
                                plt.close('all')
                            elif param_model == 'deep_gauss':
                                sm.qqplot(dt0.numpy(), stats.norm, line='45', loc=4, scale=1, fit=False)
                                plt.tight_layout()
                                plt.savefig('figs/{}/qq_{}_nc.pdf'.format(param_model, loss))
                                plt.close('all')
                            elif 'uniform' in param_model:
                                sm.qqplot(dt0.numpy(), stats.uniform, line='45', fit=False)
                                plt.tight_layout()
                                plt.savefig('figs/{}/qq_{}_nc.pdf'.format(param_model, loss))
                                plt.close('all')
                            elif 'log-normal' in param_model:
                                sm.qqplot(dt0.numpy(), stats.lognorm, line='45', fit=False, distargs=(1,))
                                plt.tight_layout()
                                plt.savefig('figs/{}/qq_{}_nc.pdf'.format(param_model, loss))
                                plt.close('all')

                            elif 'eda' in param_model or 'healthy-ecg' in param_model or 'btc' in param_model:
                                qqplot_2samples(dt0.numpy()[:validation[:250].shape[0]], validation[:250].numpy(), line='45')
                                plt.tight_layout()
                                plt.savefig('figs/{}/qq_{}.pdf'.format(param_model, loss))
                                plt.close('all')

                        # plot the CDFs
                        plt.plot(torch.sort(dt0)[0].detach(), np.linspace(0,1,dt0.shape[0], endpoint=False), label='Est')
                        plt.plot(cdf_data.detach(), np.linspace(0,1,dt0_data.shape[0], endpoint=False), label='True')
                        if validation is not None and not isinstance(validation, list):
                            plt.plot(torch.sort(validation)[0].detach(), np.linspace(0,1,validation.shape[0], endpoint=False), label='Val')
                        plt.legend()
                        plt.savefig('figs/{}/cdf_{}.pdf'.format(param_model, loss))
                        plt.close('all')

                    # run the validation callback
                    if val_callback is not None:
                        val_callback(tpp, eps, epoch)

                    torch.save(tpp.state_dict(), 'figs/{}/tpp.p'.format(param_model))

    return tpp, eps, ks, mu

def run_diffusion_experiment_2d(m, s, name=None):

    all_mse = []

    all_lambda = []

    all_mu = []

    for ind in range(5):

        print('Run number ' + str(ind))

        init = 0 if name=='gbm' else 0
        usemax = False if name=='gbm' else False

        data = get_data(m, s, dt=0.1, N=2000, offset=0, x_init=init, use_max=usemax)
        loss = 'diff'
        t, t0, xt, pp, a, b, marks = setup_data(data, use_curve=False, marks=True)
        mse, flow, lambda_, mu = train(1, t, t0, xt, m, s, loss=loss, use_curve=False, use_time=False, param_model=name, prior_type='diffusion', epochs=1001, n_print=250, marks=marks)
        all_mse.append(mse)
        all_lambda.append(lambda_)

        xt = torch.linspace(-2,2, 100).unsqueeze(-1)
        all_mu.append(mu(xt).detach())

        print('-------------')

    am = torch.stack(all_mu)
    amm = am.mean(0)
    ams = am.std(0)

    plt.plot(xt, amm, label=r'$\hat{\mu}$')
    plt.fill_between(xt.squeeze(-1), (amm - ams).squeeze(-1), (amm + ams).squeeze(-1), alpha=0.4)
    plt.plot(xt, m(0,xt), label=r'$\mu$')
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\mu(x)$')
    plt.tight_layout()
    plt.savefig('figs/{}/mean_mu.pdf'.format(name))

    return np.array(all_mse)

def run_diffusion_experiment(m, s, name=None):

    all_mse = []

    all_lambda = []

    all_mu = []

    for ind in range(5):

        print('Run number ' + str(ind))

        init = 0 if name=='gbm' else 0
        usemax = False if name=='gbm' else False

        data = get_data(m, s, dt=0.1, N=2000, offset=0, x_init=init, use_max=usemax)
        use_curve = False
        loss = 'diff'
        t, t0, xt, pp, a, b = setup_data(data, use_curve=use_curve)
        mse, flow, lambda_, mu = train(1, t, t0, xt, m, s, loss=loss, use_curve=False, use_time=False, param_model=name, prior_type='diffusion', epochs=1001, n_print=250)
        all_mse.append(mse)
        all_lambda.append(lambda_)

        xt = torch.linspace(-2,2, 100).unsqueeze(-1)
        all_mu.append(mu(xt).detach())

        print('-------------')

    am = torch.stack(all_mu)
    amm = am.mean(0)
    ams = am.std(0)

    plt.plot(xt, amm, label=r'$\hat{\mu}$')
    plt.fill_between(xt.squeeze(-1), (amm - ams).squeeze(-1), (amm + ams).squeeze(-1), alpha=0.4)
    plt.plot(xt, m(0,xt), label=r'$\mu$')
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\mu(x)$')
    plt.tight_layout()
    plt.savefig('figs/{}/mean_mu.pdf'.format(name))

    return np.array(all_mse)

def intensity_free_data(name : str = 'synth/stationary_renewal.pkl', seed : int = 82):
    '''
        Thank you to these researchers:
        https://github.com/shchur/ifl-tpp/
        who have very nice code for benchmarking. 
    '''

    import sys
    sys.path.append('../parison/ifl-tpp/code/')
    import dpp
    ds = dpp.data.load_dataset(name)
    print(ds)

    print('using dataset: {}'.format(name))

    d_train, d_val, d_test = ds.train_val_test_split(seed=seed)

    t0_train = [timest.inter_times.cumsum(0) for timest in d_train]
    t0_val   = [timesv.inter_times.cumsum(0) for timesv in d_val]
    t0_test  = [timestt.inter_times.cumsum(0) for timestt in d_test]

    dt0_train = [timest.inter_times for timest in d_train]
    dt0_val   = [timesv.inter_times for timesv in d_val]
    dt0_test  = [timestt.inter_times for timestt in d_test]
    t = torch.linspace(d_train[0].t_start+1e-5, d_train[0].t_end, 100)

    return t0_train, t0_val, t0_test, t, dt0_train, dt0_val, dt0_test

def run_if_benchmarks(synth_type : str):
    '''
    Run the benchmarks from the intensity free paper.

    synth_type : one of synth/stationary_poisson, nonstationary_poisson, 
                 stationary_renewal, nonstationary_renewal,
                 hawkes1, hawkes2, self_correcting, stack_overflow
    '''
    t0_train, t0_val, t0_test, t, dt0_train, dt0_val, dt0_test = intensity_free_data(synth_type)
    mse, flow, lambda_, tpp, eps = train(1, t, t0_train, validation=dt0_val, loss='reg-mle', param_model='synth')
    print('Validation loss')
    all_log_p=[]
    with torch.no_grad():
        for t0_ in t0_val:
            tdiff = t0_[1:] - t0_[:-1]
            for t0 in tdiff:
                if torch.isnan(tpp.logp(t0, eps=eps)[0]):
                    print(t0)
                all_log_p.append(tpp.logp(t0, eps=eps)[0])
        print(torch.cat(all_log_p).mean().item())

def run_benchmarks_com(synth_type : str, elbo=False, use_time=True):
    '''
    Run the benchmarks for where we are given samples and want to maximize the likelihood.
    '''
    width  = 16
    layers = 4 

    if synth_type == 'exp':
        m = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
    elif synth_type == 'gamma':
        m = torch.distributions.gamma.Gamma(torch.tensor([9.0]), torch.tensor([1.0]))
    elif synth_type == 'log-normal':
        m = torch.distributions.log_normal.LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
    elif synth_type == 'weibull':
        m = torch.distributions.weibull.Weibull(torch.tensor([1.0]), torch.tensor([1.5]))
    elif synth_type == 'uniform':
        m = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
    elif synth_type == 'mix-gamma':
        mix = torch.distributions.categorical.Categorical(torch.ones(2,))
        comp = torch.distributions.gamma.Gamma(torch.tensor([2.0, 7.5]), torch.tensor([2.0, 1.0]))
        m = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)

    if not use_time:
        synth_type = synth_type + '-nt'


    t0_train_ = m.sample((40,5))
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
    if elbo:
        train(1, t, t0_train, validation=t0_val, epochs=2001,n_print=250,warm_epochs=2000, true_pdf=true_pdf, loss='reg-elbo', param_model=synth_type+'-elbo')
    else:
        # 2001
        # 250
        train(1, t, t0_train, validation=t0_val, true_pdf=true_pdf, loss='reg-mle', param_model=synth_type, epochs=2001, n_print=250, act=nn.Softplus(), use_time=use_time, warm_epochs=2000)

def run_ground_truth(synth_type):
    use_time=False
    t = torch.linspace(0.5,5,100)
    if synth_type == 'exp':
        t = torch.linspace(0.5,5,100)
        m = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
    elif synth_type == 'gamma':
        t = torch.linspace(0.5,10,100)
        m = torch.distributions.gamma.Gamma(torch.tensor([5.0]), torch.tensor([1.0]))
    elif synth_type == 'log_normal':
        m = torch.distributions.log_normal.LogNormal(torch.tensor([-1.8]), torch.tensor([1.9]))
    elif synth_type == 'weibull':
        t = torch.linspace(0.5,5,100)
        m = torch.distributions.weibull.Weibull(torch.tensor([1.0]), torch.tensor([1.5]))
    elif synth_type == 'uniform':
        t = torch.linspace(0.05,1,100)
        m = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
    elif synth_type == 'deep_gauss':
        t = torch.linspace(0.05,10,100)
        m = torch.distributions.normal.Normal(torch.tensor([4.0]), torch.tensor([1.0]))
    elif synth_type == 'mix-gamma':
        mix = torch.distributions.categorical.Categorical(torch.ones(2,))
        comp = torch.distributions.gamma.Gamma(torch.tensor([2.0, 7.5]), torch.tensor([2.0, 1.0]))
        m = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)
    t0_train_ = m.sample((100,10))
    t0_train = []
    for idx in range(t0_train_.shape[0]):
        cs = t0_train_[idx].cumsum(0).squeeze()
        a = torch.zeros(cs.shape[0]+1)
        a[1:] = cs
        t0_train.append(a)

    true_pdf = lambda t: m.log_prob(t).exp().squeeze()
    train(1, t, t0=t0_train, true_pdf=true_pdf, param_model=synth_type, loss='mse', use_time=True, epochs=3001, n_print=250)

def run_healthy_ecg_experiment():
    bpath = 'data/physionet.org/files/rr-interval-healthy-subjects/1.0.0/'
    import pandas as pd
    import os 
    info = pd.read_csv(os.path.join(bpath, 'patient-info.csv'))
    idx = '003.txt'
    t0 = (np.loadtxt(os.path.join(bpath, idx)) / 5e3)
    print(t0)

    t0_train = torch.tensor(t0[:200].cumsum())
    idxv = '005.txt'
    t0v = (np.loadtxt(os.path.join(bpath, idxv)) / 5e3)
    t0_val= torch.tensor(t0v[:200])
    t = torch.linspace(1e-4, t0.max(), 50)
    train(1, t, t0=t0_train, param_model='healthy-ecg', loss='reg-mle', validation=t0_val)

def run_welltory_experiment():
    ks_all = []
    param_model = 'healthy-ecg-well-med-elbo'
    for i in range(1,6):
        bpath = 'data/welltory/0{}.txt'.format(i)
        t0_ = np.loadtxt(bpath) / 1e4
        t0_val = torch.tensor(t0_[t0_.shape[0] //2:])
        t0 = t0_[:t0_.shape[0] //2 ]
        print('Training size: {}'.format(t0.shape))
        print('Testing  size: {}'.format(t0_val.shape))
        t0_train = torch.tensor(t0.cumsum())
        t = torch.linspace(1e-5, t0.max(), 50)
        _,_, ks, _ = train(1, t, t0=t0_train, param_model=param_model, loss='reg-mle', validation=t0_val, run_num=i)
        ks_all.append(ks[0])
    print(ks_all)
    print(np.array(ks_all).mean())
    print(np.array(ks_all).std())

    np.savetxt('figs/{}/summary.txt'.format(param_model), np.array(ks_all))

def run_eda_experiment(propofol=False):
    width  = 4
    layers = 2
    ks_all = []
    for i in range(1,6):
        param_model='eda-reg-elbo'
        bpath = 'data/eda/pulse_times_0{}.csv'.format(i)
        use_time=True
        epochs = 2001
        n_print = 1000
        loss = 'reg-mle'
        if propofol:
            param_model='eda-p'
            bpath = 'data/eda/propofol/pulse_times_0{}.csv'.format(i)
            use_time = False
            epochs  = 5001
            n_print = 1000
            loss = 'reg-elbo'
        t0_ = np.loadtxt(bpath) / 1e2
        t0_val = torch.tensor(t0_[t0_.shape[0] //2:])
        t0 = t0_[:t0_.shape[0] //2 ]
        t0_val = t0_val[1:] - t0_val[:-1]
        t0_train = torch.tensor(t0)
        print('Training size: {}'.format(t0.shape))
        print('Testing  size: {}'.format(t0_val.shape))
        t = torch.linspace(1e-5, (t0[1:]-t0[:-1]).max(), 50)
        _,_, ks, _ = train(1, t, t0=t0_train, param_model=param_model, loss=loss, validation=t0_val, run_num=i, use_time=use_time, epochs=epochs, n_print=n_print, width=wdith, layers=layers)
        ks_all.append(ks[0])
    print(ks_all)
    print(np.array(ks_all).mean())
    print(np.array(ks_all).std())

    np.savetxt('figs/{}/summary.txt'.format(param_model), np.array(ks_all))

def run_btc_max():

    import pandas as pd

    use_time=False
    epochs=2000
    n_print = 250
    param_model='btc'

    for i in range(5):

        df = pd.read_csv('data/btcusd.csv')
        df.time = pd.to_datetime(df.time,unit='ms')
        time_diff_df = df.groupby(df.time.dt.day).cummax()[1:][df.groupby(df.time.dt.day).cummax().diff().high > 0]
        dt0 = time_diff_df.groupby(time_diff_df.time.dt.day).time.diff().dropna().values / np.timedelta64(1, 's') / 3600

        t0_ = dt0.cumsum(0)
        t0_val = torch.tensor(t0_[200*(i+1):200*(i+2)])
        t0 = t0_[200*i:200*(i+1)]
        t0_val = t0_val[1:] - t0_val[:-1]
        t0_train = torch.tensor(t0)
        print(t0_val)
        print('Training size: {}'.format(t0.shape))
        print('Testing  size: {}'.format(t0_val.shape))
        t = torch.linspace(1e-5, (t0[1:]-t0[:-1]).max(), 50)
        tpp,_, ks, _ = train(1, t, t0=t0_train, param_model=param_model, loss='reg-mle', validation=t0_val, run_num=i, use_time=use_time, epochs=epochs, n_print=n_print)

def run_neuron_experiment_no_hist_pickle():

    param_model = 'neuron-no-hist-pickle'
    if not os.path.isdir('figs/{}'.format(param_model)):
        os.mkdir('figs/{}'.format(param_model))

    width  = 64
    layers = 2

    with open('data/neuron_recordings/neuron_data.p', 'rb') as f:
        import pickle
        data = pickle.load(f)

    stim = data['stim']
    stim_len = len(stim)
    stim_time = data['stim_time']
    spikeTimes = data['spikeTimes']
    spikeIndices = data['spikeIndices']
    Trial_Index = data['Trial_Index']
    spk = data['spk']
    all_hspk = data['all_hspk']

    use_time = True
    use_stim = True 

    epochs = 3001
    n_print = 1000

    loss = 'diff'

    dt = 0.001 

    t0 = spk
    t0_train = torch.tensor(t0)
    t = torch.linspace(1e-5, (t0[1:]-t0[:-1]).max(), 50)

    if use_stim:
        in_stim = (stim_time, stim)
    else:
        in_stim = None

    def val_callback(tpp, eps, epoch):
        # plot the samples histogram
        plt.figure(figsize=(20,3))
        all_ht0 = []
        ap  = []
        for sim_n in Trial_Index:
            t0_, dt0, paths, t_p = tpp.sample(1,stim.shape[0],stim_time[-1],(eps).abs().item(), use_first=False, use_abs=loss == 'diff')
            if t0_ is not None:
                ht0 = torch.histc(t0_, bins=stim_len , min=0, max=stim_len*dt)
                all_ht0.append(ht0)
            if paths is not None:
                ap.append(paths)
        ht0   = torch.stack(all_ht0)
        paths = torch.stack(ap)
        plt.plot(stim_time, ht0.mean(0),      label='Model Samples', alpha=0.75, color='salmon')
        plt.plot(stim_time, all_hspk.mean(0), label='Real Samples', alpha=0.75, color='dodgerblue')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\mathbb{E}[N(t)]$')
        plt.legend()
        plt.tight_layout()
        plt.savefig('figs/neuron-no-hist-pickle/learned_spikes_hist_{}_{}.pdf'.format(loss, epoch))
        plt.close('all')

        with open('figs/neuron-no-hist-pickle/stim_and_z.p','wb') as f:
            data_dict = {'stim': stim, 'paths' : paths, 'stim_time' : stim_time, 'paths_time' : t_p}
            pickle.dump(data_dict, f)


    tpp, eps, ks, _ = train(1, t, t0=t0_train, 
            param_model=param_model, 
            loss=loss, 
            validation=t0_train, 
            use_time=use_time, 
            epochs=epochs, 
            n_print=n_print, 
            stim=in_stim, 
            val_callback=val_callback, 
            width=width, layers=layers)

    plt.figure(figsize=(20,3))
    all_ht0 = []
    for sim_n in Trial_Index:
        t0_, dt0, paths, t_p = tpp.sample(1,stim.shape[0],stim_time[-1],(eps).abs().item(), use_first=False, use_abs=loss == 'diff')
        if t0_ is not None:
            ht0 = torch.histc(t0_, bins=stim_len, min=0, max=stim_len*dt)
            all_ht0.append(ht0)
    ht0 = torch.stack(all_ht0)
    plt.plot(stim_time, ht0.mean(0),      label='Simulated')
    plt.plot(stim_time, all_hspk.mean(0), label='Real')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\mathbb{E}[N(t)]$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/neuron-no-hist-pickle/learned_spikes_hist_{}.pdf'.format(loss))
    plt.close('all')


def run_neuron_experiment():
    import scipy.io

    param_model = 'neuron'
    spike_trainFile = 'data/neuron_recordings/MC.mat'
    stim_file = 'data/neuron_recordings/stim.mat'

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

    use_time = True
    use_stim = True 
    epochs = 50001
    n_print = 1000
    loss = 'mle'

    dt = 0.001 #0.1
    stim_len = len(stim)
    stim_time = np.linspace(0, stim_len*dt, stim_len)

    spk = spikeTimes[spikeIndices == Trial_Index[0]+1] * dt
    spk_val = spikeTimes[spikeIndices == Trial_Index[1]+1] * dt
    all_hspk = []
    for ii in Trial_Index:
        t_spk = torch.tensor(spikeTimes[spikeIndices == Trial_Index[ii]+1] * dt)
        all_hspk.append(torch.histc(t_spk, bins=stim_len , min=0, max=stim_len*dt))
    all_hspk = torch.stack(all_hspk)
    plt.figure(figsize=(16,5))
    #plt.plot(stim_time, all_hspk.mean(0), linewidth=0.5)
    plt.plot(all_hspk.mean(0), linewidth=0.5)
    plt.savefig('figs/neuron/orig_spikes_hist.pdf')
    plt.close('all')

    t0 = spk
    t0_val = torch.tensor(spk_val[1:] - spk_val[:-1])
    t0_train = torch.tensor(t0)
    print('Training size: {}'.format(t0.shape))
    print('Testing  size: {}'.format(t0_val.shape))
    t = torch.linspace(1e-5, (t0[1:]-t0[:-1]).max(), 50)

    if use_stim:
        in_stim = (stim_time, stim)
    else:
        in_stim = None

    def val_callback(tpp, eps, epoch):
        # plot the samples histogram
        plt.figure(figsize=(20,3))
        all_ht0 = []
        for sim_n in Trial_Index:
            t0_, dt0, paths, t_p = tpp.sample(1,10000,stim_time[-1],(eps).abs().item(), use_first=False, use_abs=loss == 'diff')
            if t0_ is not None:
                ht0 = torch.histc(t0_, bins=stim_len , min=0, max=stim_len*dt)
                all_ht0.append(ht0)
        ht0 = torch.stack(all_ht0)
        diff = (F.mse_loss(ht0.mean(0), all_hspk.mean(0))).item()
        print(diff)
        plt.plot(t_p, ht0.mean(0),      label='Model')
        plt.plot(stim_time, all_hspk.mean(0), label='Real')
        plt.legend()
        plt.tight_layout()
        plt.savefig('figs/neuron/learned_spikes_hist_{}_{}_{:2f}.pdf'.format(loss, epoch, diff))
        plt.close('all')

    tpp, eps, ks, _ = train(1, t, t0=t0_train, param_model=param_model, loss=loss, validation=t0_val, use_time=use_time, epochs=epochs, n_print=n_print, stim=in_stim, val_callback=val_callback)

    plt.figure(figsize=(20,3))
    all_ht0 = []
    for sim_n in Trial_Index:
        print(sim_n)
        t0_, dt0, paths, t_p = tpp.sample(1,10000,stim_time[-1],(eps).abs().item(), use_first=False, use_abs=loss == 'diff')
        if t0_ is not None:
            ht0 = torch.histc(t0_, bins=stim_len, min=0, max=stim_len*dt)
            all_ht0.append(ht0)
    ht0 = torch.stack(all_ht0)
    plt.plot(stim_time, ht0.mean(0),      label='sim')
    plt.plot(stim_time, all_hspk.mean(0), label='real')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/neuron/learned_spikes_hist_{}.pdf'.format(loss))
    plt.close('all')


def run_neuron_no_hist_experiment():
    import scipy.io
    import pickle

    width  = 64
    layers = 2

    param_model = 'neuron-no-hist'
    spike_trainFile = 'data/neuron_recordings/MC.mat'
    stim_file = 'data/neuron_recordings/stim.mat'

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

    use_time = True
    use_stim = True 

    epochs = 3001
    n_print = 1000

    loss = 'diff'

    dt = 0.001 #0.1
    stim_len = len(stim)
    stim_time = np.linspace(0, stim_len*dt, stim_len)

    plt.figure(figsize=(16,5))
    plt.plot(stim_time[:-1], stim[1:] - stim[:-1], linewidth=0.5)
    plt.savefig('figs/neuron-no-hist/orig_stim.pdf')
    plt.close('all')

    spk = spikeTimes[spikeIndices == Trial_Index[0]+1] * dt
    spk_val = spikeTimes[spikeIndices == Trial_Index[1]+1] * dt
    all_hspk = []
    for ii in Trial_Index:
        t_spk = torch.tensor(spikeTimes[spikeIndices == Trial_Index[ii]+1] * dt)
        all_hspk.append(torch.histc(t_spk, bins=stim_len , min=0, max=stim_len*dt))
    all_hspk = torch.stack(all_hspk)
    plt.figure(figsize=(16,5))
    #plt.plot(stim_time, all_hspk.mean(0), linewidth=0.5)
    plt.plot(all_hspk.mean(0), linewidth=0.5)
    plt.savefig('figs/neuron-no-hist/orig_spikes_hist.pdf')
    plt.close('all')

    t0 = spk
    t0_val = torch.tensor(spk_val[1:] - spk_val[:-1])
    t0_train = torch.tensor(t0)
    print('Training size: {}'.format(t0.shape))
    print('Testing  size: {}'.format(t0_val.shape))
    t = torch.linspace(1e-5, (t0[1:]-t0[:-1]).max(), 50)

    if use_stim:
        in_stim = (stim_time, stim)
    else:
        in_stim = None


    def val_callback(tpp, eps, epoch):
        # plot the samples histogram
        plt.figure(figsize=(20,3))
        all_ht0 = []
        ap  = []
        for sim_n in Trial_Index:
            t0_, dt0, paths, t_p = tpp.sample(1,stim.shape[0],stim_time[-1],(eps).abs().item(), use_first=False, use_abs=loss == 'diff')
            if t0_ is not None:
                ht0 = torch.histc(t0_, bins=stim_len , min=0, max=stim_len*dt)
                all_ht0.append(ht0)
            if paths is not None:
                ap.append(paths)
        ht0   = torch.stack(all_ht0)
        paths = torch.stack(ap)
        diff =  (F.mse_loss(ht0.mean(0), all_hspk.mean(0))).item()
        print(diff)
        plt.plot(stim_time, ht0.mean(0),      label='Model Samples', alpha=0.75, color='salmon')
        plt.plot(stim_time, all_hspk.mean(0), label='Real Samples', alpha=0.75, color='dodgerblue')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\mathbb{E}[N(t)]$')
        plt.legend()
        plt.tight_layout()
        plt.savefig('figs/neuron-no-hist/learned_spikes_hist_{}_{}_{:2f}.pdf'.format(loss, epoch, diff))
        plt.close('all')

        with open('figs/neuron-no-hist/stim_and_z.p','wb') as f:
            data_dict = {'stim': stim, 'paths' : paths, 'stim_time' : stim_time, 'paths_time' : t_p}
            pickle.dump(data_dict, f)

        neg_log_sig = -(paths.mean(0).abs().squeeze(0).squeeze(-1)+1e-2).log()[1:]
        scale = (stim[1:] / neg_log_sig).mean()

        plt.figure(figsize=(16,5))

        plt.plot(t_p[1:], neg_log_sig, label='Mean Sample', color='salmon', alpha=0.75)
        plt.plot(stim_time[1:], stim[1:], label='True Stimulus', color='dodgerblue', alpha=0.75)
        plt.legend()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\mathcal{S}(t)$')
        plt.savefig('figs/neuron-no-hist/learned_stim_{}_{}.pdf'.format(loss, epoch))
        plt.close('all')

        app = torch.ones_like(neg_log_sig)
        A = torch.stack((neg_log_sig, app), -1).numpy()
        b = stim[1:]

        x, r, _, _ = np.linalg.lstsq(A, b)

        plt.figure(figsize=(16,5))
        plt.plot(t_p[1:], A @ x, label='Mean Transformed Sample', color='salmon', alpha=0.75)
        plt.plot(stim_time[1:], stim[1:], label='True Stimulus', color='dodgerblue', alpha=0.75)
        plt.legend()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\mathcal{S}(t)$')
        plt.savefig('figs/neuron-no-hist/scaled_learned_stim_{}_{}.pdf'.format(loss, epoch))
        plt.close('all')

        plt.figure(figsize=(16,5))
        plt.plot(t_p[1:], A @ x, label='Mean Transformed Sample', color='salmon', alpha=0.75)
        plt.plot(stim_time[1:], np.abs(stim[1:]), label='True Stimulus', color='dodgerblue', alpha=0.75)
        plt.legend()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\mathcal{S}(t)$')
        plt.savefig('figs/neuron-no-hist/scaled_learned_stim_abs_{}_{}.pdf'.format(loss, epoch))
        plt.close('all')

        mean_path = paths.mean(0).abs().squeeze(0).squeeze(-1)[1:]

        app = torch.ones_like(mean_path)
        A = torch.stack((mean_path, app), -1).numpy()
        b = np.abs(stim[1:] - stim[:-1])

        x, r, _, _ = np.linalg.lstsq(A, b)

        plt.figure(figsize=(16,5))
        plt.plot(t_p[1:], A @ x, label='Mean Transformed Sample', color='salmon', alpha=0.75)
        plt.plot(stim_time[1:], b, label='True Stimulus', color='dodgerblue', alpha=0.75)
        plt.legend()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\mathcal{S}(t)$')
        plt.savefig('figs/neuron-no-hist/scaled_learned_stim_dt_{}_{}.pdf'.format(loss, epoch))
        plt.close('all')

    tpp, eps, ks, _ = train(1, t, t0=t0_train, 
            param_model=param_model, 
            loss=loss, 
            validation=t0_val, 
            use_time=use_time, 
            epochs=epochs, 
            n_print=n_print, 
            stim=in_stim, 
            val_callback=val_callback, 
            width=width, layers=layers)

    plt.figure(figsize=(20,3))
    all_ht0 = []
    for sim_n in Trial_Index:
        print(sim_n)
        t0_, dt0, paths, t_p = tpp.sample(1,stim.shape[0],stim_time[-1],(eps).abs().item(), use_first=False, use_abs=loss == 'diff')
        if t0_ is not None:
            ht0 = torch.histc(t0_, bins=stim_len, min=0, max=stim_len*dt)
            all_ht0.append(ht0)
    ht0 = torch.stack(all_ht0)
    plt.plot(stim_time, ht0.mean(0),      label='Simulated')
    plt.plot(stim_time, all_hspk.mean(0), label='Real')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\mathbb{E}[N(t)]$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/neuron-no-hist/learned_spikes_hist_{}.pdf'.format(loss))
    plt.close('all')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('experiment')

args = parser.parse_args()

if __name__ == '__main__':

    if args.experiment  == 'neuron':
        run_neuron_experiment()

    elif args.experiment == 'neuron-nohist':
        run_neuron_no_hist_experiment()

    elif args.experiment == 'neuron-nohistp':
        run_neuron_experiment_no_hist_pickle()

    elif args.experiment == 'hrv':
        run_welltory_experiment()

    elif args.experiment == 'eda':
        run_eda_experiment()

    elif args.experiment == 'eda-p':
        run_eda_experiment(propofol=True)

    elif args.experiment == 'exp':
        run_benchmarks_com('exp')

    elif args.experiment == 'gamma':
        run_benchmarks_com('gamma')

    elif args.experiment == 'weibull':
        run_benchmarks_com('weibull')

    elif args.experiment == 'uniform':
        run_benchmarks_com('uniform')

    elif args.experiment == 'lognormal':
        run_benchmarks_com('log-normal')
    
    elif args.experiment == 'lognormal-nt':
        run_benchmarks_com('log-normal', use_time=False)



