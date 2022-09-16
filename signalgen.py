import numpy as np

def signal1d_derivatives(signal, max_order, antiderivative=True, noise=False, random_state=None):
    x = np.array(signal)
    y = np.zeros((max_order+1, *x.shape))
    if noise:
        xe = random_state.randn(x.size) * 0.01
    else:
        xe = 0
    y[0] = x + xe

    for order in np.arange(1, max_order+1):
        y[order, :-order] = y[order-1, order:] - y[order-1, :-order]

    if antiderivative:
        z = np.zeros_like(x)
        for i in range(x.size):
            z[i] = z[i - 1] + x[i]
        y = np.vstack((y, z))

    return y.T

def updown_generator(N, n_changepoints, random_state):
    changepoints = np.insert(np.sort(random_state.randint(0, N, n_changepoints)), [0, n_changepoints], [0, N])
    const_intervals = list(zip(changepoints, np.roll(changepoints, -1)))[:-1]
    updown_control = np.ones((N, 2))
    up = True
    for (t0, t1) in const_intervals:
        if t0 != t1:
            if up:
                updown_control[t0:t1, 0] = np.linspace(0, 1, t1-t0)
            else:
                updown_control[t0:t1, 0] = np.linspace(1, 0, t1-t0)
            up = not up
    return updown_control