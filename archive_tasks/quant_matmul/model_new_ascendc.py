import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"

try:
    import quant_matmul_ext
except ImportError:
    import glob as _glob
    _libs = _glob.glob(str(_KERNEL_BUILD / "quant_matmul_ext*.so"))
    if _libs:
        torch.ops.load_library(_libs[0])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, scale):
        return torch.ops.npu.quant_matmul(a, b, scale)
