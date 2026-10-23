import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(variance + eps) + beta


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                        W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                        W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch_size, seq_len_q, d_model = Q.shape
    _, seq_len_k, _ = K.shape
    d_k = d_model // num_heads

    # 1. Chiếu tuyến tính (Linear Projections) Q, K, V
    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)

    # 2. Tách thành nhiều heads và đổi trục để chạy song song
    Q_heads = Q_proj.reshape(batch_size, seq_len_q, num_heads, d_k).swapaxes(1, 2)
    K_heads = K_proj.reshape(batch_size, seq_len_k, num_heads, d_k).swapaxes(1, 2)
    V_heads = V_proj.reshape(batch_size, seq_len_k, num_heads, d_k).swapaxes(1, 2)

    # 3. Tính Scaled Dot-Product Attention cho từng head
    scores = np.matmul(Q_heads, K_heads.swapaxes(-2, -1)) / np.sqrt(d_k)
    
    # Softmax ổn định số học dọc theo trục key-position
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 4. Nhân trọng số attention với V_heads
    head_outputs = np.matmul(attn_weights, V_heads)

    # 5. Ghép nối (Concatenate) các heads lại với nhau
    concat = head_outputs.swapaxes(1, 2).reshape(batch_size, seq_len_q, d_model)

    # 6. Chiếu đầu ra cuối cùng bằng ma trận W_o
    output = np.matmul(concat, W_o)

    return output

def feed_forward(z: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray):
    q = np.matmul(z, W1) + b1
    relu_q = np.maximum(0, q)
    ffn = np.matmul(relu_q, W2) + b2
    return ffn

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    attn_output = multi_head_attention(
        Q=x,
        K=x,
        V=x,
        W_q=W_q,
        W_k=W_k,
        W_v=W_v,
        W_o=W_o,
        num_heads=num_heads
    )

    res1 = x + attn_output
    norm1 = layer_norm(res1, gamma1, beta1)

    ffn = feed_forward(norm1, W1=W1, b1=b1, W2=W2, b2=b2)

    res2 = norm1 + ffn
    output = layer_norm(res2, gamma2, beta2)

    return output