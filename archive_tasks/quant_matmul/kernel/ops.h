#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor quant_matmul(const at::Tensor &a, const at::Tensor &b, const at::Tensor &scale);

} // namespace ascend_kernel

#endif // OPS_H
