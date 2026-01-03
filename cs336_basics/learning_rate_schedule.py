import math

def learning_rate_schedule(t: int, lr_max: float, lr_min: float, num_warm_up: int, num_cosine_annealing: int) -> float:
    if t < num_warm_up:
        return t / num_warm_up * lr_max
    elif t < num_cosine_annealing:
        return lr_min + 1 / 2 * (1 + math.cos((t - num_warm_up) / (num_cosine_annealing - num_warm_up) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min

