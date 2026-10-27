import numpy as np

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    
    x_copy = x.copy()

    x_W1 = x_copy @ W1.T

    relu_x_W1 = np.maximum(0, x_W1)

    res_part = relu_x_W1 @ W2.T + x_copy

    output = np.maximum(0, res_part)

    return output