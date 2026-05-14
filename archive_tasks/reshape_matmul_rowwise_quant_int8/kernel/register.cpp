#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("reshape_matmul_quant(Tensor x, Tensor h) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("reshape_matmul_quant", TORCH_FN(ascend_kernel::reshape_matmul_quant));
}

}  // namespace
