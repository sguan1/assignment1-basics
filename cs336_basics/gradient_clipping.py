import math
from typing import Iterable

import torch

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    
    l2_norm = torch.sqrt(sum(g.detach().pow(2).sum() for g in grads))
    if l2_norm >= max_l2_norm:
        factor = max_l2_norm / (l2_norm + 1e-6)
        for g in grads:
            g.detach().mul_(factor)
