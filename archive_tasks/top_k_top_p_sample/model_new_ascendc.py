import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _top_k_top_p_sample_ext as _ext  # noqa: E402


def run_top_k_top_p_sample(logits, top_ks, top_ps, q, is_need_logits=False, top_k_guess=32, eps=1e-8):
    return _ext.run_top_k_top_p_sample(logits, top_ks, top_ps, q, is_need_logits, top_k_guess, eps)


class ModelNew(nn.Module):
    def __init__(self, eps: float = 1e-8, top_k_guess: int = 32) -> None:
        super().__init__()
        self.eps = float(eps)
        self.top_k_guess = int(top_k_guess)
        self._pending_outputs = []

    def forward(self, logits, top_ks, top_ps, q, is_need_logits=False):
        if logits.device.type == "npu" and self._pending_outputs:
            torch.npu.synchronize(logits.device)
            self._pending_outputs.clear()
        if logits.device.type == "npu":
            torch.npu.synchronize(logits.device)
        logits_2d = logits.contiguous().reshape(-1, logits.shape[-1])
        top_ks_1d = top_ks.contiguous().reshape(-1).to(torch.int32)
        top_ps_1d = top_ps.contiguous().reshape(-1).to(logits.dtype)
        q_2d = q.contiguous().reshape(-1, logits.shape[-1]).to(torch.float32)
        selected_idx, selected_logits = run_top_k_top_p_sample(
            logits_2d,
            top_ks_1d,
            top_ps_1d,
            q_2d,
            is_need_logits,
            self.top_k_guess,
            self.eps,
        )
        selected_idx = selected_idx.reshape(top_ks.shape)
        selected_logits = selected_logits.reshape(logits.shape)
        if logits.device.type == "npu":
            torch.npu.synchronize(logits.device)
            self._pending_outputs.append((selected_idx, selected_logits))
        return selected_idx, selected_logits


def get_init_inputs():
    return []
