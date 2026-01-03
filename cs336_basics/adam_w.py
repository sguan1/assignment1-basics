import math
from typing import Callable, Optional
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay: float = 0.1, betas:  tuple[float, float] = (0.9,0.999), eps: float = 1e-8):
        defaults = {"lr":lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self,  closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            b1,b2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad # Get the gradient of loss with respect to p.
                state["m"] = b1 * m + (1 - b1) * grad
                state["v"] = b2 * v + (1 - b2) * grad.pow(2)
                lr_t = lr * math.sqrt(1 - b2**t) / (1 - b1**t)
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1 # Increment iteration number.

        return loss

