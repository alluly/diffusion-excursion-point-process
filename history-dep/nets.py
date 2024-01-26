import numpy as np
import scipy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


def init_weights(net, init_dict, gain=1, input_class=None):
    def init_func(m):
        if input_class is None or type(m) == input_class:
            for key, value in init_dict.items():
                param = getattr(m, key, None)
                if param is not None:
                    if value == 'normal':
                        nn.init.normal_(param.data, 0.0, gain)
                    elif value == 'xavier':
                        nn.init.xavier_normal_(param.data, gain=gain)
                    elif value == 'kaiming':
                        nn.init.kaiming_normal_(param.data, a=0, mode='fan_in')
                    elif value == 'orthogonal':
                        nn.init.orthogonal_(param.data, gain=gain)
                    elif value == 'uniform':
                        nn.init.uniform_(param.data)
                    elif value == 'zeros':
                        nn.init.zeros_(param.data)
                    elif value == 'very_small':
                        nn.init.constant_(param.data, 1e-3*gain)
                    elif value == 'xavier1D':
                        nn.init.normal_(param.data, 0.0, gain/param.numel().sqrt())
                    elif value == 'identity':
                        nn.init.eye_(param.data)
                    else:
                        raise NotImplementedError('initialization method [%s] is not implemented' % value)
#activation functions
class quadratic(nn.Module):
    def __init__(self):
        super(quadratic,self).__init__()

    def forward(self,x):
        return x**2

class quadratic(nn.Module):
    def __init__(self):
        super(quadratic,self).__init__()

    def forward(self,x):
        return x*F.relu(x)

class cos(nn.Module):
    def __init__(self):
        super(cos,self).__init__()

    def forward(self,x):
        return torch.cos(x)

class sin(nn.Module):
    def __init__(self):
        super(sin,self).__init__()

    def forward(self,x):
        return torch.sin(x)

class swish(nn.Module):
    def __init__(self):
        super(swish,self).__init__()

    def forward(self,x):
        return torch.sigmoid(x)*x

class relu2(nn.Module):
    def __init__(self,order=2):
        super(relu2,self).__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.order = order

    def forward(self,x):
        #return F.relu(self.a.to(x.device)*x)**(self.order)
        return F.relu(x)**(self.order)

class leakyrelu2(nn.Module):
    def __init__(self,order=2):
        super(leakyrelu2,self).__init__()
        self.a = nn.Parameter(torch.ones(1))
        #self.a = torch.ones(1)
        self.order = order

    def forward(self,x):
        return F.leaky_relu(self.a.to(x.device)*x)**self.order

class mod_softplus(nn.Module):
    def __init__(self):
        super(mod_softplus,self).__init__()

    def forward(self,x):
        return F.softplus(x) + x/2 - torch.log(torch.ones(1)*2).to(device=x.device)

class mod_softplus2(nn.Module):
    def __init__(self):
        super(mod_softplus2,self).__init__()

    def forward(self,x,d):
        return d*(1+d)*(2*F.softplus(x) - x  - 2*torch.log(torch.ones(1)*2).to(device=x.device))

class mod_softplus3(nn.Module):
    def __init__(self):
        super(mod_softplus3,self).__init__()

    def forward(self,x):
        return F.relu(x) + F.softplus(-torch.abs(x)) 

class swish(nn.Module):
    def __init__(self):
        super(swish,self).__init__()

    def forward(self,x):
        return x*torch.sigmoid(x) 

class soft2(nn.Module):
    def __init__(self):
        super(soft2,self).__init__()

    def forward(self,x):
        return torch.sqrt(x**2 + 1) / 2 + x / 2

class soft3(nn.Module):
    def __init__(self):
        super(soft3,self).__init__()

    def forward(self,x):
        return torch.logsigmoid(-x) 
class Shallow(nn.Module):
    def __init__(self,input_size,out_size):
        super(Shallow, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size,input_size),quadratic(),nn.Linear(input_size,out_size))

    def forward(self,x):
        return self.net(x)

class PositiveLinear(nn.Linear):
    def __init__(self, **args):
        super(PositiveLinear, self).__init__()


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, layers, out_size, act=nn.Softplus(), bn=False, bias=False):
        super(MLP, self).__init__()

        self.act = act

        self.fc1 = nn.Linear(input_size,hidden_size)
        if bn:
            self.bn = nn.BatchNorm1d(hidden_size)
        else:
            self.bn = None
        mid_list = []
        for i in range(layers):
            if bn:
                mid_list += [nn.Linear(hidden_size,hidden_size), nn.BatchNorm1d(hidden_size), act]
            else:
                mid_list += [nn.Linear(hidden_size,hidden_size), act]
        self.mid = nn.Sequential(*mid_list)
        self.out = nn.Linear(hidden_size, out_size, bias=bias)
        #init_weights(self, {'weights':'xavier', 'bias':'zeros'})

    def forward(self, x):
        out = self.fc1(x)
        if self.bn:
            out = self.bn(out)
        out = self.act(out)
        out = self.mid(out)
        out = self.out(out)#.clamp(max=0.1)
        #out = 0.5*torch.tanh(out)
        #out = -x ** 2 + 5
        #out = -10*x  + 5
        #out = 10*torch.exp(-x) -8*x
        return out


class SplineRegression(torch.nn.Module):
    def __init__(
            self,
            input_range,
            order=3,
            knots=10):
        super(SplineRegression, self).__init__()
        if isinstance(knots, int):
            knots = np.linspace(input_range[0], input_range[1], knots)
        num_knots = len(knots)
        knots = np.hstack([knots[0]*np.ones(order),
                           knots,
                           knots[-1]*np.ones(order)])
        self.basis_funcs = scipy.interpolate.BSpline(
            knots, np.eye(num_knots+order-1), k=order)
        self.linear = torch.nn.Linear(num_knots+order-1, 1)

        x = np.linspace(input_range[0], input_range[1], 100)
        y = self.basis_funcs(x)
        plt.plot(x, y)
        plt.show()

    def forward(self, x):
        x_shape = x.shape
        x_basis = self.basis_funcs(x.reshape(-1))
        x_basis = torch.from_numpy(x_basis).float()
        out = self.linear(x_basis)
        return out.reshape(x_shape)


class SplineRegressionHistory(torch.nn.Module):
    def __init__(
            self,
            input_range,
            history_kernel,
            order=3,
            knots=10,
            history_order=2):
        """
        Args:
            history_kernel: history kernel basis.
        """
        super(SplineRegressionHistory, self).__init__()
        if isinstance(knots, int):
            knots = np.linspace(input_range[0], input_range[1], knots)
        num_knots = len(knots)
        knots = np.hstack([knots[0]*np.ones(order),
                           knots,
                           knots[-1]*np.ones(order)])
        self.basis_funcs = scipy.interpolate.BSpline(
            knots, np.eye(num_knots+order-1), k=order)
        self.state_linear = torch.nn.Linear(num_knots+order-1, 1)
        self.history_linear = torch.nn.Linear(history_order, 1, bias=False)

        self.state_linear.weight.data.fill_(0.0)
        self.history_linear.weight.data.fill_(torch.rand(1).item())
        # for param in self.history_linear.parameters():
        #     param.requires_grad = False

        self.history_order = history_order
        self.history_kernel = history_kernel
        print('self.history_order', self.history_order)

        x = np.linspace(input_range[0], input_range[1], 100)
        y = self.basis_funcs(x)
        tau = torch.linspace(0, 4, 100)
        y_kernel = self.history_kernel(tau)

        # fig, axes = plt.subplots(1, 2, figsize=[10, 3])
        fig, axes = plt.subplots(1, 1, figsize=[4, 2])
        axes.plot(x, y)

        fig, axes = plt.subplots(1, 1, figsize=[4, 2])
        axes.plot(tau, y_kernel)
        plt.ylim(0)
        plt.savefig('tau.pdf')
        plt.close('all')


    def stack_taus(self, t, history, batch_repeat=True):
        """Stack past k events.

        Debug code for top k.
        x = 10 - torch.arange(1., 20.)
        x[x < 0] = float('inf')
        print(x)
        x,_ = torch.topk(-x, 3)
        print(-x)
        """
        num_batch = t.shape[0]

        if batch_repeat:
            t_ = t[0].float()
            tau = t_ - history.float()
            tau[tau < 0] = float('inf')
            tau, _ = torch.topk(tau, k=self.history_order, dim=-1, largest=False)
            taus = tau.unsqueeze(0).repeat(num_batch,1,1,1)
            # print('tau')
            # print(t.shape)
            # print(tau.shape)
            # plt.figure(figsize=[22, 3])
            # plt.plot(t[0,33].reshape(-1), tau[33].reshape(-1))
            # plt.plot(t[0,33].reshape(-1), self.history_kernel(tau[33].reshape(-1)))
            # plt.show()
            # raise ValueError

        else:  # TODO
            history = torch.cat([torch.zeros(20)-99, history]).float()
            t = t.float().reshape(num_batch, -1, 1)
            taus = torch.zeros_like(t).repeat(1,1,2)
            t_ = t[:,:,0]

            #import seaborn
            #plt.figure()
            #seaborn.distplot(t, bins=101)
            #plt.show()

            for b in range(num_batch):
                tau = t[b] - history
                # Find the most recent past histories.
                tau[tau < 0] = float('inf')
                taus[b],_ = torch.topk(tau, k=self.history_order, dim=1, largest=False)

                #plt.figure()
                #plt.plot(t_[b], taus[b][:,0].reshape(-1))
                # plt.plot(tau_)
                # seaborn.distplot(tau_)
                #plt.show()

        assert (taus >= 0).any()
        return taus.reshape(-1, self.history_order)

    def state_forward(self, x, return_input_shape=True):
        x_shape = x.shape
        '''

        x_basis = self.basis_funcs(x.reshape(-1))
        x_basis = torch.from_numpy(x_basis).float()
        out = self.state_linear(x_basis)
        if return_input_shape:
            return out.reshape(x_shape)
        else:
            return out
        #'''

        # Debugging as fixed.
        if return_input_shape:
            return -x
        else:
            return -x.reshape(-1,1)

    def history_forward(self, t, history, batch_repeat=True):
        taus = self.stack_taus(t, history, batch_repeat=batch_repeat)
        history_basis = self.history_kernel(taus)
        out = self.history_linear(history_basis)
        return out

    def forward(self, x, t, history, batch_repeat=True):
        x_shape = x.shape
        out = (self.state_forward(x, return_input_shape=False) +
               self.history_forward(t, history, batch_repeat=batch_repeat))

        # out = self.state_forward(x, return_input_shape=False)
        # print('.......', x_basis.shape, history_basis.shape, x_basis_cat.shape)
        # print(x.shape, t.shape, history.shape, type(history))
        return out.reshape(x_shape)







