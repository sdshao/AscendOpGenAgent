import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"

try:
    import reshape_matmul_quant_ext
except ImportError:
    import glob as _glob
    _libs = _glob.glob(str(_KERNEL_BUILD / "reshape_matmul_quant_ext*.so"))
    if _libs:
        torch.ops.load_library(_libs[0])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, h):
        return torch.ops.npu.reshape_matmul_quant(x, h)
