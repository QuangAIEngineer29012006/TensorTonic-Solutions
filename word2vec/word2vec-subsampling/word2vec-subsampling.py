import torch

def subsample_keep_probs(counts: torch.Tensor,
                         t: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 keep probability for every vocabulary word.
    """
    counts = counts.to(torch.float64)

    f = counts / counts.sum()

    keep_probs = torch.sqrt(t / f)
    
    keep_probs = torch.clamp(keep_probs, max=1.0)
    
    return keep_probs