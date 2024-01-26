from local_tpp import LocalTimePP

from data import setup_data, get_data_nd

import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nets import MLP

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

import numpy as np

DEVICE = 'cpu'

width = 64 
layers = 2

class Linear(nn.Module):
    def __init__(self, in_size, width, layers, out_size):
        super(Linear, self).__init__()

        self.net = nn.Linear(in_size, out_size)

    def forward(self, x):
        return self.net(x)

def train(d : int, 
        t  : torch.tensor, 
        t0 : torch.tensor, 
        xt : torch.tensor = None, 
        m  : type(lambda t,x : x) = None, 
        s  : type(lambda t,x : x) = None, 
        use_curve : bool = False,
        ex_name : str = 'test',
        use_bridge : bool = False):

    '''
    Training routine. Takes in time horizon, arrival times, and state space.

    d int -- integer describing dimensionality of the problem.
    t tensor  -- tensor describing time horizon
    t0 tensor -- tensor describing interarrival times

    xt tensor -- tensor describing state space (optional)

    m function -- function for mu (optional)
    s function -- function for sigma (optional)

    use_curve boolean -- flag to use a curve or not
    '''

    t.requires_grad = True

    mu = MLP(d, width, layers, d).to(DEVICE)

    sigma = MLP(d, width, layers, d, bias=True).to(DEVICE)
    nn.init.zeros_(sigma.out.weight.data)
    nn.init.ones_(sigma.out.bias.data)

    lr_mu = 5e-4
    lr_mu = 1e-3
    lr_sigma = 0

    opt_params = [{'params': mu.parameters(), 'lr': lr_mu},
            {'params': sigma.parameters(), 'lr': lr_sigma}]

    if use_curve:
        pass
    else:
        x0 = torch.zeros_like(t)
    if xt is None:
        xt = torch.linspace(-1,1)

    tpp = LocalTimePP(mu, sigma, t, t0, torch.zeros(1), x0, d=d, use_bridge=use_bridge, prior='diffusion')

    opt = optim.Adam(opt_params)

    epochs = 200
    
    best_loss = 1000

    for epoch in range(epochs):

        opt.zero_grad()

        loss = tpp.train_step()

        loss.backward()
        opt.step()

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                print(loss)
                if loss < best_loss:
                    best_loss = loss.clone()
                    if m is not None:
                        muest  = mu(xt).detach()
                        mutrue = m(0,xt) * torch.ones_like(xt)
                        mse = F.mse_loss(muest, mutrue).item()
                        nmse = (((mutrue-muest)**2).mean() / (mutrue **2).mean()).item()

                        print('MSE {:4f}'.format(mse)) # compute MSE
                        print('Normalized Err {:4f}'.format(nmse)) # compute PctE

                        xtp = torch.zeros(100, d)

                        for ind in range(d):
                            xtp[:,ind] = torch.linspace(xt[:,ind].min(), xt[:,ind].max(), 100)

                        #xtp = torch.linspace(xt.min(), xt.max()).unsqueeze(1) # plot over the ordered space
                        plt.plot(xtp, m(0,xtp) * torch.ones_like(xtp), color='tab:blue')#, label=r'$\mu$')
                        plt.plot(xtp, mu(xtp).detach(),'--', color='tab:red')#, label=r'$\hat{\mu}$')
                        #plt.legend()
                        plt.savefig('figs/multi_d/{}_mu_est_nd.pdf'.format(ex_name))
                        plt.close('all')
    return mse, nmse

def run_diffusion_experiment(m, s, d, ex_name, use_bridge):

    all_mse = []

    for ind in range(10):

        print('Run number ' + str(ind))
        if use_bridge:
            print('WARNING: MUST RUN EXCURSION EXPERIMENT FIRST (to generate the data), an error will occur otherwise.')
            with open('experiments/synthetic/{}{}{}_data-log.pkl'.format(ex_name,d,ind), 'rb') as f:
                import pickle
                data = pickle.load(f)
        else:
            data = get_data_nd(m, s, d=d, offset=0)
            with open('experiments/synthetic/{}{}{}_data-log.pkl'.format(ex_name,d,ind), 'wb') as f:
                import pickle
                pickle.dump(data, f)

        use_curve = False
        t, t0, xt, pp, a, b = setup_data(data, use_curve=use_curve, ex_name=ex_name)
        mse = train(d, t, t0, xt, m, s, False, ex_name=ex_name, use_bridge=use_bridge)
        all_mse.append(mse)

        print('-------------')

    return np.array(all_mse)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bridge', action='store_true')
parser.add_argument('--no-bridge', dest='bridge', action='store_false')
parser.set_defaults(bridge=False)

args = parser.parse_args()

if __name__ == '__main__':
    import pickle
    s = lambda t, x: 1*torch.ones_like(x)

    if args.bridge:
        use_bridge = True
    else:
        use_bridge = False

    if use_bridge:
        app = '-bb-log'
    else:
        app = '-ex-log'
    m = lambda t, x: -torch.tanh(x)
    all_mse = run_diffusion_experiment(m, s, d=10, ex_name='tanh', use_bridge=use_bridge)
    with open('experiments/synthetic/tanh10{}-10.pkl'.format(app), 'wb') as f:
        pickle.dump(all_mse, f)
    m = lambda t, x: -x**3
    all_mse = run_diffusion_experiment(m, s, d=10, ex_name='cubic', use_bridge=use_bridge)
    with open('experiments/synthetic/cubic10{}-10.pkl'.format(app), 'wb') as f:
        pickle.dump(all_mse, f)
    def m(t,x):
        if len(x.shape) >  1:
            return torch.stack((-x[:,0] - 1*x[:,1], -x[:,1] + 5*x[:,0]),1)
        else:
            return torch.stack((-x[0] - 1*x[1], -x[1] + 5*x[0]))
    all_mse = run_diffusion_experiment(m, s, d=2, ex_name='lv', use_bridge=use_bridge)
    with open('experiments/synthetic/lv2{}-10.pkl'.format(app), 'wb') as f:
        pickle.dump(all_mse, f)
    m = lambda t, x: -4*x
    all_mse = run_diffusion_experiment(m, s, d=5, ex_name='ou', use_bridge=use_bridge)
    with open('experiments/synthetic/ou5{}-10.pkl'.format(app), 'wb') as f:
        pickle.dump(all_mse, f)
