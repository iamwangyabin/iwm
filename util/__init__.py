
import torch
from .lr_decay import add_weight_decay
from .lr_sched import cosine_scheduler



def nested_to_gpu(nested_list, device):
    if isinstance(nested_list, torch.Tensor):
        return nested_list.to(device, non_blocking=True)  # Move tensor to GPU
    else:
        return [nested_to_gpu(item, device) for item in nested_list]  # Recursively process list
