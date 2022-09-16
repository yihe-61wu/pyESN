import numpy as np

def signal1d_derivatives(signal, max_order):
    X = np.array(signal)
    Y = np.zeros((max_order+1, *X.shape))
    Y[0] = X
    for order in np.arange(1, max_order+1):
        Y[order, :-order] = Y[order-1, order:] - Y[order-1, :-order]
    return Y.T