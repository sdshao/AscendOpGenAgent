#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor> kv_sort(const at::Tensor &keys, const at::Tensor &values);

} // namespace ascend_kernel

#endif // OPS_H
