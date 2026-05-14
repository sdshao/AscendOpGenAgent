#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("quant_matmul(Tensor a, Tensor b, Tensor scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("quant_matmul", TORCH_FN(ascend_kernel::quant_matmul));
}

}  // namespace
