#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor reshape_matmul_quant(const at::Tensor &x, const at::Tensor &h);

} // namespace ascend_kernel

#endif // OPS_H
