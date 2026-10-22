import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    position = np.arange(seq_length, dtype=np.float64)[:, np.newaxis]

    div_part = np.exp(np.arange(0, d_model, 2, dtype=np.float64) * -(np.log(10000) / d_model))

    pe = np.zeros((seq_length, d_model), dtype=np.float64)

    pe[:, 0::2] = np.sin(position * div_part)
    pe[:, 1::2] = np.cos(position * div_part)

    return pe
    