import torch
import numpy as np

def ito_diffusion(mu, sigma, dt, N, x_init, tau=0):
    assert tau == 0, 'tau ~= 0 not yet implemented' 

    dt = torch.tensor(dt)
    tN = dt * (N + tau)

    t = torch.linspace(0, tN, N+tau)
    x = torch.zeros(N+tau)
    snell = torch.zeros_like(x)
    smell = torch.zeros_like(x)

    x[0] = x_init
    snell[0] = x_init
    smell[0] = x_init


    b0 = 0

    for i in range(N-1):
        x[i+1] = x[i] + mu(t[i], x[i]) * dt + sigma(t[i], x[i]) * dt.sqrt() * torch.randn(1)
        #snell[i] = torch.max(torch.cat((x[i+1], x[i] + mu(t[i], x[i]) * dt),0))
        #snell[i] = x[i] + mu(t[i], x[i]) * dt
        snell[i+1] = torch.tensor(np.max((x[i+1], snell[i] + mu(t[i], snell[i]) * dt)))
        smell[i+1] = torch.tensor(np.min((x[i+1], smell[i] + mu(t[i], smell[i]) * dt)))

    return x, t, snell, smell



def ito_diffusion_nd(mu, sigma, dt, N, x_init, tau=0):
    assert tau == 0, 'tau ~= 0 not yet implemented' 
    assert len(x_init.shape) == 1

    dt = torch.tensor(dt)
    tN = dt * N

    d = x_init.shape[0]

    t = torch.linspace(0, tN, N)
    x = torch.zeros(N, d)
    snell = torch.zeros_like(x)
    smell = torch.zeros_like(x)

    x[0] = x_init
    snell[0] = x_init
    smell[0] = x_init

    b0 = 0

    for i in range(N-1):
        x[i+1] = x[i] + mu(t[i], x[i]) * dt + sigma(t[i], x[i]) * dt.sqrt() * torch.randn(d)
        #snell[i+1] = torch.tensor(np.max((x[i+1], snell[i] + mu(t[i], snell[i]) * dt)))
        #smell[i+1] = torch.tensor(np.min((x[i+1], smell[i] + mu(t[i], smell[i]) * dt)))

    return x, t, snell, smell
