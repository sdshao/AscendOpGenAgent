import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"

try:
    import kv_sort_ext
except ImportError:
    import glob as _glob
    _libs = _glob.glob(str(_KERNEL_BUILD / "kv_sort_ext*.so"))
    if _libs:
        torch.ops.load_library(_libs[0])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keys, values):
        return torch.ops.npu.kv_sort(keys, values)
