import sys
from pathlib import Path

import torch
import torch.nn as nn


def _is_running_tilelang_verification() -> bool:
    return any("verification_tilelang.py" in arg for arg in sys.argv)


if _is_running_tilelang_verification() and hasattr(torch, "npu"):
    torch.npu.is_available = lambda: False

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from design.tile_level.top_k_top_p_sample import top_k_top_p_sample as tl_top_k_top_p_sample


class ModelNew(nn.Module):
    def __init__(self, eps: float = 1e-8, top_k_guess: int = 32) -> None:
        super().__init__()
        self.eps = float(eps)
        self.top_k_guess = int(top_k_guess)

    def _build_kernel(self, logits_2d: torch.Tensor, is_need_logits: bool):
        row_num, row_len = logits_2d.shape
        return tl_top_k_top_p_sample(
            row_num,
            row_len,
            is_need_logits=bool(is_need_logits),
            top_k_guess=self.top_k_guess,
            eps=self.eps,
            dtype=str(logits_2d.dtype).split(".")[-1],
        )

    def forward(self, logits, top_ks, top_ps, q, is_need_logits=False):
        assert logits.ndim >= 2
        assert top_ks.shape == logits.shape[:-1]
        assert top_ps.shape == logits.shape[:-1]
        assert q.shape == logits.shape
        assert logits.dtype in (torch.float16, torch.float32, torch.bfloat16)
        assert top_ps.dtype == logits.dtype
        assert q.dtype == torch.float32

        original_logits_shape = logits.shape
        original_index_shape = top_ks.shape

        logits_2d = logits.reshape(-1, logits.shape[-1]).contiguous()
        top_ks_1d = top_ks.reshape(-1).to(torch.int32).contiguous()
        top_ps_1d = top_ps.reshape(-1).contiguous()
        q_2d = q.reshape(-1, q.shape[-1]).contiguous()

        kernel = self._build_kernel(logits_2d, bool(is_need_logits))
        selected_idx, selected_logits = kernel(
            logits_2d,
            top_ks_1d,
            top_ps_1d,
            q_2d,
        )
        return selected_idx.reshape(original_index_shape), selected_logits.reshape(original_logits_shape)


def get_init_inputs():
    return []
