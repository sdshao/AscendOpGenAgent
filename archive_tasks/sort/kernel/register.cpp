#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("kv_sort(Tensor keys, Tensor values) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("kv_sort", TORCH_FN(ascend_kernel::kv_sort));
}

}  // namespace
