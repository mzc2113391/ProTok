import math
import torch
from torch.optim.lr_scheduler import LambdaLR



def get_cosine_scheduler_with_warmup(optimizer, warmup_steps, total_steps, max_lr, min_lr,init_lr = None):

    if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError("Require total_steps > 0 and 0 <= warmup_steps < total_steps.")
    if max_lr <= 0 or not 0 <= min_lr <= max_lr:
        raise ValueError("Require max_lr > 0 and 0 <= min_lr <= max_lr.")
    if init_lr is None:
        init_lr = 0.0
    if not 0 <= init_lr <= max_lr:
        raise ValueError("Require 0 <= init_lr <= max_lr.")
    init_lr_factor = init_lr / max_lr 
    min_lr_factor = min_lr / max_lr    

    def lr_lambda(current_step: int):
        
        if current_step < warmup_steps:
            progress = float(current_step) / float(max(1, warmup_steps))

            return init_lr_factor + progress * (1.0 - init_lr_factor)
        else:
            progress = min(1.0, float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cosine_decay * (1 - min_lr_factor) + min_lr_factor

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
