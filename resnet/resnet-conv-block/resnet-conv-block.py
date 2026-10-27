import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns the projection residual-block output as a nested list.
    """
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    Ws = np.array(Ws,dtype=np.float64)
    
    x_copy = x.copy()
    
    x_W1 = x_copy @ W1
    relu_x_W1 = np.maximum(0, x_W1)

    skip_branch = x_copy @ Ws
    
    res_part = relu_x_W1 @ W2 + skip_branch

    y = np.maximum(0, res_part)

    return y