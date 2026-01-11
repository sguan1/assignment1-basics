import torch

def cross_entropy_loss(logits: torch.Tensor , targets: torch.Tensor) -> torch.Tensor:
    logits_shifted = logits - logits.max(dim=-1, keepdim=True).values
    logits_exp = torch.exp(logits_shifted)
    negative_log_probability =  torch.log(logits_exp.sum(dim=-1, keepdim=True)) -logits_shifted
    target_probability = negative_log_probability.gather(dim = -1, index =targets.unsqueeze(-1).long())
    return torch.mean(target_probability)