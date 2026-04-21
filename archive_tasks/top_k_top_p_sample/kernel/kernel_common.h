#ifndef TOP_K_TOP_P_SAMPLE_KERNEL_COMMON_H
#define TOP_K_TOP_P_SAMPLE_KERNEL_COMMON_H

#include <cstddef>
#include <cstdint>

#include "kernel_operator.h"

template <typename T>
__aicore__ inline void CopyTiling(T *tiling, GM_ADDR tilingGM)
{
    int32_t *dst = reinterpret_cast<int32_t *>(tiling);
    auto *src = reinterpret_cast<__gm__ int32_t *>(tilingGM);
    for (size_t index = 0; index < sizeof(T) / sizeof(int32_t); ++index) {
        dst[index] = src[index];
    }
}

#endif
