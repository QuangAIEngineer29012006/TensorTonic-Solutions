import numpy as np
import math
import torch.nn.functional as F

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch_size, seq_length_q, d_model = Q.shape
    _, seq_length_k, _ = K.shape
    d_k = d_model // num_heads

    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)

    Q_head = Q_proj.reshape(batch_size, seq_length_q, num_heads, d_k).swapaxes(1, 2)
    K_head = K_proj.reshape(batch_size, seq_length_k, num_heads, d_k).swapaxes(1, 2)
    V_head = V_proj.reshape(batch_size, seq_length_k, num_heads, d_k).swapaxes(1, 2)

    score = np.matmul(Q_head, K_head.swapaxes(-2, -1)) / math.sqrt(d_k)

    exp_score = np.exp(score - np.max(score, axis=-1, keepdims=True))
    attn_score = exp_score / np.sum(exp_score, axis=-1, keepdims=True)

    head_output = np.matmul(attn_score, V_head)

    concat = head_output.swapaxes(1, 2).reshape(batch_size, seq_length_q, d_model)

    output = np.matmul(concat, W_o)

    return output
    
    

    