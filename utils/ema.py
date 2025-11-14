# utils/ema.py
# ---------------------------------------------------------
# Exponential Moving Average for model parameters
# ---------------------------------------------------------
from __future__ import annotations
import copy
import torch

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).cpu()
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self._names = [n for n, _ in model.named_parameters()]

    @torch.no_grad()
    def update(self):
        src = self._get_src_model()
        for (n, p_src), (_, p_tgt) in zip(src.named_parameters(), self.shadow.named_parameters()):
            if p_tgt.dtype != p_src.dtype:
                p_tgt.data = p_src.detach().to(p_tgt.dtype).cpu()
            else:
                p_tgt.data.mul_(self.decay).add_(p_src.detach().cpu(), alpha=1.0 - self.decay)

    def _get_src_model(self):
        # The caller should ensure training model is on right device
        # We locate it from the shadow's structure only to keep names aligned.
        # In this simple implementation we assume single model reference.
        # For more robust use, pass the live model on update().
        return self._live_model  # set by attach()

    def attach(self, model: torch.nn.Module):
        self._live_model = model
