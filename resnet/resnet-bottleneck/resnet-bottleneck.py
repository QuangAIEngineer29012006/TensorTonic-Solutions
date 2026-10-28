import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    W3 = np.array(W3, dtype=np.float64)

    if Ws is None or isinstance(Ws, (list, np.array)) and len(Ws) == 0:
        x_Ws = x
    else:
        Ws = np.array(Ws, dtype=np.float64)
        x_Ws = x @ Ws

    compress = np.maximum(0, x @ W1)

    operate = np.maximum(0, compress @ W2)

    output = np.maximum(0, operate @ W3 + x_Ws)

    return output
    