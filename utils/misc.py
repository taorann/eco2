# utils/misc.py
# ---------------------------------------------------------
# Small utilities: seeding, device, grad clipping, param count
# ---------------------------------------------------------
from __future__ import annotations
import os, random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device(device_str: str = "cuda"):
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clip_grad_norm_(params, max_norm: float):
    torch.nn.utils.clip_grad_norm_(params, max_norm)
