#include "int8_matmul_scale.h"

extern "C" __global__ __aicore__ void quant_matmul(GM_ADDR a, GM_ADDR b, GM_ADDR scale,
                                                     GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    Int8MatmulScaleKernel kernel;
    kernel.Init(a, b, scale, c, workspace, tiling, &pipe);
    kernel.Process();
}
