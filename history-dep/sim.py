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


def ito_diffusion_history_running_max_min(mu, sigma, dt, N, x_init, tau=0):
    """History of running max or min."""

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
    # history_min = []
    # history_max = []
    history = [-999] * 100  # Enough long history.

    for i in range(N-1):
        x[i+1] = (x[i] + mu(t[i], x[i], history) * dt +
                  sigma(t[i], x[i]) * dt.sqrt() * torch.randn(1))
        #snell[i] = torch.max(torch.cat((x[i+1], x[i] + mu(t[i], x[i]) * dt),0))
        #snell[i] = x[i] + mu(t[i], x[i]) * dt
        snell[i+1] = torch.tensor(np.max((x[i+1], snell[i] + mu(t[i], snell[i], history) * dt)))
        smell[i+1] = torch.tensor(np.min((x[i+1], smell[i] + mu(t[i], smell[i], history) * dt)))

        if snell[i+1] == x[i+1] or smell[i+1] == x[i+1]:
            history.append(t[i+1])
    return x, t, snell, smell


def ito_diffusion_history_crossings(mu, sigma, dt, N, x_init, tau=0):
    """History of running max or min."""

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
    # history_min = []
    # history_max = []
    history = [-999] * 100  # Enough long history.
    zero_cnt = 0

    for i in range(N-1):
        x[i+1] = (x[i] + mu(t[i], x[i], history) * dt +
                  sigma(t[i], x[i]) * dt.sqrt() * torch.randn(1))
        snell[i+1] = torch.tensor(np.max((x[i+1], snell[i] + mu(t[i], snell[i], history) * dt)))
        smell[i+1] = torch.tensor(np.min((x[i+1], smell[i] + mu(t[i], smell[i], history) * dt)))

        if x[i+1] * x[i] < 0:  # zero crossings.
            history.append(t[i+1])
            zero_cnt += 1

    print('zero_cnt', zero_cnt)
    return x, t, snell, smell


def ito_diffusion_external_stimulus(mu, sigma, stim, dt, N, x_init, tau=0):
    """History of running max or min."""

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
    stim = torch.cat((-999*torch.ones(20), stim), 0)

    b0 = 0

    for i in range(N-1):
        stim_current = stim[stim <= t[i]]
        x[i+1] = (x[i] + mu(t[i], x[i], stim_current) * dt +
                  sigma(t[i], x[i]) * dt.sqrt() * torch.randn(1))
        snell[i+1] = torch.tensor(np.max((x[i+1], snell[i] + mu(t[i], snell[i], stim_current) * dt)))
        smell[i+1] = torch.tensor(np.min((x[i+1], smell[i] + mu(t[i], smell[i], stim_current) * dt)))

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
