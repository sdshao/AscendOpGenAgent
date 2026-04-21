#include "kernel_operator.h"

#include "kernel_common.h"
#include "top_k_top_p_sample_tiling.h"
#include "top_k_top_p_sample.h"

using namespace AscendC;
using namespace TopKTopPSample;

extern "C" __global__ __aicore__ void top_k_top_p_sample_fp16_custom(
    GM_ADDR logits,
    GM_ADDR topKs,
    GM_ADDR topPs,
    GM_ADDR q,
    GM_ADDR logitsSelectIdx,
    GM_ADDR logitsTopKpSelect,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    TopKTopPSampleTilingData tilingData;
    CopyTiling(&tilingData, tiling);
    TopKTopPSampleKernel<half> kernel;
    kernel.Init(logits, topKs, topPs, q, logitsSelectIdx, logitsTopKpSelect, workspace, tilingData, &pipe);
    kernel.Process();
    pipe.Destroy();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void top_k_top_p_sample_do_fp16(
    uint32_t blockDim,
    void *stream,
    uint8_t *logits,
    uint8_t *topKs,
    uint8_t *topPs,
    uint8_t *q,
    uint8_t *logitsSelectIdx,
    uint8_t *logitsTopKpSelect,
    uint8_t *workspace,
    uint8_t *tiling)
{
    top_k_top_p_sample_fp16_custom<<<blockDim, nullptr, stream>>>(
        logits,
        topKs,
        topPs,
        q,
        logitsSelectIdx,
        logitsTopKpSelect,
        workspace,
        tiling);
}
#endif
