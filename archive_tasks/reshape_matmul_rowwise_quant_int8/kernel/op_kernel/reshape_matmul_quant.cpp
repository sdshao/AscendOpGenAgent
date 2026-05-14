#include "reshape_matmul_quant.h"

extern "C" __global__ __aicore__ void reshape_matmul_quant(
    GM_ADDR x, GM_ADDR h, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    ReshapeMatmulQuantKernel<bfloat16_t, float, int8_t> kernel;
    kernel.Init(x, h, y, workspace, tiling, &pipe);
    kernel.Process();
}
