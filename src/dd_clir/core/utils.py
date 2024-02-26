from typing import Literal

import torch

def pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor, method: Literal["avg", "cls"] = "avg"):
    """Pooling function for a batch of hidden states"""
    if method == "avg":
        pooled = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
    elif method == "cls":
        pooled = hidden_states[:, 0]
    else:
        raise ValueError(f"Pooling method {method} not supported")
    return pooled