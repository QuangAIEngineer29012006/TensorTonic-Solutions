import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns the network logits as a nested list.
    """
    x = np.asarray(x, dtype=np.float64)
    conv1 = np.asarray(conv1, dtype=np.float64)
    W1_b1 = np.asarray(W1_b1, dtype=np.float64)
    W2_b1 = np.asarray(W2_b1, dtype=np.float64)
    W1_b2 = np.asarray(W1_b2, dtype=np.float64)
    W2_b2 = np.asarray(W2_b2, dtype=np.float64)
    Ws_b2 = np.asarray(Ws_b2, dtype=np.float64)
    fc = np.asarray(fc, dtype=np.float64)

    out = np.maximum(0, x @ conv1)

    shortcut_b1 = out.copy()
    main_b1 = np.maximum(0, out @ W1_b1)
    main_b1 = main_b1 @ W2_b1
    out = np.maximum(0, main_b1 + shortcut_b1)

    shortcut_b2 = out @ Ws_b2
    main_b2 = np.maximum(0, out @ W1_b2)
    main_b2 = main_b2 @ W2_b2
    out = np.maximum(0, main_b2 + shortcut_b2)

    logits = out @ fc
    return logits