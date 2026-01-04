import numpy.typing as npt
import numpy as np
import torch


def data_loader(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    max_index = len(dataset) - context_length - 1

    start_indexes = np.random.randint(0, max_index + 1, batch_size)

    input_sequence = np.zeros((batch_size, context_length), dtype=np.int32)
    next_token_sequence = np.zeros((batch_size, context_length), dtype=np.int32)

    for i, start_index in enumerate(start_indexes):
        input_sequence[i] = dataset[start_index: start_index + context_length]
        next_token_sequence[i] = dataset[start_index + 1: start_index + 1 + context_length]

    input = torch.from_numpy(input_sequence).to(device)
    next_token = torch.from_numpy(next_token_sequence).to(device)

    return input, next_token



