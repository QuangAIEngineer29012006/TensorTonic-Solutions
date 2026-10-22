import torch
import math
import torch.nn.functional as F

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))

    softmax_score = F.softmax(score, dim=-1)

    scaled_dot = torch.matmul(softmax_score, V)

    return scaled_dot