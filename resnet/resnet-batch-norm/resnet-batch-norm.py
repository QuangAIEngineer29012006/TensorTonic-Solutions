import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode in a dictionary.
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    gamma1 = np.asarray(gamma1, dtype=np.float64)
    beta1 = np.asarray(beta1, dtype=np.float64)
    gamma2 = np.asarray(gamma2, dtype=np.float64)
    beta2 = np.asarray(beta2, dtype=np.float64)

    eps = 1e-5
    residual = x.copy()
    
    def batch_norm(tensor, gamma, beta):
        mean = tensor.mean(axis=0)
        var = tensor.var(axis=0)
        x_hat = (tensor - mean) / np.sqrt(var + eps)
        return gamma * x_hat + beta

    if mode == 'post':
        out = x @ W1
        out = batch_norm(out, gamma1, beta1)
        out = np.maximum(0, out)

        out = out @ W2
        out = batch_norm(out, gamma2, beta2)
        out = out + residual
        out = np.maximum(0, out)

    if mode == 'pre':
        out = batch_norm(x, gamma1, beta1)
        out = np.maximum(0, out)
        out = out @ W1

        out = batch_norm(out, gamma2, beta2)
        out = np.maximum(0, out)
        out = out @ W2
        out = out + residual

    return {
        "output": out,
        "mode": mode
    }
        