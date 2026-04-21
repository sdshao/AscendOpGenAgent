#include <algorithm>
#include <cstdint>
#include <limits>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "top_k_top_p_sample_tiling.h"

extern "C" uint32_t aclrtlaunch_top_k_top_p_sample_fp16_custom(
    uint32_t blockDim,
    void *stream,
    void *logits,
    void *topKs,
    void *topPs,
    void *q,
    void *logitsSelectIdx,
    void *logitsTopKpSelect,
    void *workspace,
    void *tiling);

extern "C" uint32_t aclrtlaunch_top_k_top_p_sample_bf16_custom(
    uint32_t blockDim,
    void *stream,
    void *logits,
    void *topKs,
    void *topPs,
    void *q,
    void *logitsSelectIdx,
    void *logitsTopKpSelect,
    void *workspace,
    void *tiling);

namespace top_k_top_p_sample_ext {

constexpr int64_t TOP_K_TOP_P_SAMPLE_SYSTEM_WORKSPACE_BYTES = 40LL * 1024LL * 1024LL;

using LaunchFn = uint32_t (*)(
    uint32_t,
    void *,
    void *,
    void *,
    void *,
    void *,
    void *,
    void *,
    void *,
    void *);

struct AclrtBuffer {
    void *ptr = nullptr;

    ~AclrtBuffer()
    {
        if (ptr != nullptr) {
            (void)aclrtFree(ptr);
        }
    }
};

uint8_t *DataPtr(const at::Tensor &tensor)
{
    return static_cast<uint8_t *>(tensor.data_ptr());
}

void CheckInputs(
    const at::Tensor &logits,
    const at::Tensor &topKs,
    const at::Tensor &topPs,
    const at::Tensor &q)
{
    TORCH_CHECK(logits.device().type() == at::kPrivateUse1, "logits must be on NPU/PrivateUse1 device");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
    TORCH_CHECK(logits.scalar_type() == at::kHalf || logits.scalar_type() == at::kBFloat16,
        "logits must be float16 or bfloat16");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");

    TORCH_CHECK(topKs.device() == logits.device(), "topKs must be on the same device as logits");
    TORCH_CHECK(topKs.dim() == 1, "topKs must be 1D");
    TORCH_CHECK(topKs.scalar_type() == at::kInt, "topKs must be int32");
    TORCH_CHECK(topKs.is_contiguous(), "topKs must be contiguous");

    TORCH_CHECK(topPs.device() == logits.device(), "topPs must be on the same device as logits");
    TORCH_CHECK(topPs.dim() == 1, "topPs must be 1D");
    TORCH_CHECK(topPs.scalar_type() == logits.scalar_type(), "topPs must have the same dtype as logits");
    TORCH_CHECK(topPs.is_contiguous(), "topPs must be contiguous");

    TORCH_CHECK(q.device() == logits.device(), "q must be on the same device as logits");
    TORCH_CHECK(q.dim() == 2, "q must be 2D");
    TORCH_CHECK(q.scalar_type() == at::kFloat, "q must be float32");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");

    TORCH_CHECK(topKs.sizes()[0] == logits.sizes()[0], "topKs batch size mismatch");
    TORCH_CHECK(topPs.sizes()[0] == logits.sizes()[0], "topPs batch size mismatch");
    TORCH_CHECK(q.sizes()[0] == logits.sizes()[0], "q batch size mismatch");
    TORCH_CHECK(q.sizes()[1] == logits.sizes()[1], "q vocab size mismatch");
}

TopKTopPSampleTilingData MakeTiling(
    uint32_t rowNum,
    uint32_t rowLen,
    bool isNeedLogits,
    float eps,
    uint32_t topKGuess)
{
    TopKTopPSampleTilingData tiling{};
    const uint32_t numCore = TOP_K_TOP_P_SAMPLE_DEFAULT_VECTOR_CORES;
    tiling.numCore = numCore;
    tiling.rowNum = rowNum;
    tiling.rowLen = rowLen;
    tiling.headCoreNum = rowNum % numCore;
    tiling.perHeadCoreRowNum = CeilDivU32(rowNum, numCore);
    tiling.tailCoreRowNum = rowNum / numCore;
    tiling.perHeadCorePartNum = 0;
    tiling.tailCorePartNum = 0;
    tiling.innerLoopEle = TOP_K_TOP_P_SAMPLE_INNER_LOOP_ELE;
    tiling.innerLoopTime = CeilDivU32(rowLen, tiling.innerLoopEle);
    tiling.innerLoopEleTail = rowLen % tiling.innerLoopEle;
    tiling.innerLoopEleTailPad = AlignUpU32(tiling.innerLoopEleTail, 32U);
    tiling.softmaxLoopTime = CeilDivU32(rowLen, TOP_K_TOP_P_SAMPLE_SOFTMAX_INNER_LOOP_ELE);
    tiling.softmaxLoopEleTail = rowLen % TOP_K_TOP_P_SAMPLE_SOFTMAX_INNER_LOOP_ELE;
    tiling.softmaxLoopEleTailPad = AlignUpU32(tiling.softmaxLoopEleTail, 32U);
    tiling.eightKPartNum = CeilDivU32(rowLen, TOP_K_TOP_P_SAMPLE_SORT_PER_MAX);
    tiling.eightKPartTail = rowLen % TOP_K_TOP_P_SAMPLE_SORT_PER_MAX;
    tiling.eightKPartTailPad = AlignUpU32(tiling.eightKPartTail, 32U);
    tiling.mrgMode = 1U;
    tiling.headOffset = tiling.headCoreNum * tiling.perHeadCoreRowNum * rowLen;
    tiling.isNeedLogits = isNeedLogits ? 1U : 0U;
    tiling.eps = eps;
    tiling.topKGuess = topKGuess;
    return tiling;
}

LaunchFn GetLaunchFn(const at::Tensor &logits)
{
    if (logits.scalar_type() == at::kHalf) {
        return aclrtlaunch_top_k_top_p_sample_fp16_custom;
    }
    if (logits.scalar_type() == at::kBFloat16) {
        return aclrtlaunch_top_k_top_p_sample_bf16_custom;
    }
    TORCH_CHECK(false, "unsupported logits dtype");
}

void RunSingleChunk(
    const at::Tensor &logits,
    const at::Tensor &topKs,
    const at::Tensor &topPs,
    const at::Tensor &q,
    at::Tensor &logitsSelectIdx,
    at::Tensor &logitsTopKpSelect,
    bool isNeedLogits,
    uint32_t topKGuess,
    float eps,
    LaunchFn launch,
    void *aclStream)
{
    const auto rowNum = static_cast<uint32_t>(logits.sizes()[0]);
    const auto rowLen = static_cast<uint32_t>(logits.sizes()[1]);
    const auto logitsSelectIdxBytes = static_cast<int64_t>(rowNum) * static_cast<int64_t>(sizeof(int64_t));
    const auto logitsTopKpSelectBytes =
        static_cast<int64_t>(rowNum) * static_cast<int64_t>(rowLen) * static_cast<int64_t>(sizeof(float));

    AclrtBuffer logitsSelectIdxRaw;
    const auto logitsSelectIdxAllocRet =
        aclrtMalloc(&logitsSelectIdxRaw.ptr, logitsSelectIdxBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    TORCH_CHECK(
        logitsSelectIdxAllocRet == ACL_ERROR_NONE,
        "aclrtMalloc for logitsSelectIdx failed with error code ",
        logitsSelectIdxAllocRet);

    AclrtBuffer logitsTopKpSelectRaw;
    const auto logitsTopKpSelectAllocRet =
        aclrtMalloc(&logitsTopKpSelectRaw.ptr, logitsTopKpSelectBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    TORCH_CHECK(
        logitsTopKpSelectAllocRet == ACL_ERROR_NONE,
        "aclrtMalloc for logitsTopKpSelect failed with error code ",
        logitsTopKpSelectAllocRet);

    auto logitsTopKpSelectInitCpu = at::full(
        {static_cast<int64_t>(rowNum), static_cast<int64_t>(rowLen)},
        -std::numeric_limits<float>::infinity(),
        at::device(at::kCPU).dtype(at::kFloat));
    const auto logitsTopKpSelectInitRet = aclrtMemcpy(
        logitsTopKpSelectRaw.ptr,
        logitsTopKpSelectBytes,
        logitsTopKpSelectInitCpu.data_ptr(),
        logitsTopKpSelectBytes,
        ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(
        logitsTopKpSelectInitRet == ACL_ERROR_NONE,
        "aclrtMemcpy for logitsTopKpSelect init failed with error code ",
        logitsTopKpSelectInitRet);

    const auto tiling = MakeTiling(rowNum, rowLen, isNeedLogits, eps, topKGuess);

    constexpr int64_t kTilingBufferBytes = 256;
    auto tilingCpu = at::zeros({kTilingBufferBytes}, at::device(at::kCPU).dtype(at::kByte));
    *reinterpret_cast<TopKTopPSampleTilingData *>(tilingCpu.data_ptr()) = tiling;

    AclrtBuffer tilingNpu;
    const auto tilingAllocRet = aclrtMalloc(&tilingNpu.ptr, kTilingBufferBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    TORCH_CHECK(tilingAllocRet == ACL_ERROR_NONE, "aclrtMalloc for tiling failed with error code ", tilingAllocRet);
    const auto tilingCopyRet = aclrtMemcpy(
        tilingNpu.ptr,
        kTilingBufferBytes,
        tilingCpu.data_ptr(),
        kTilingBufferBytes,
        ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(tilingCopyRet == ACL_ERROR_NONE, "aclrtMemcpy for tiling failed with error code ", tilingCopyRet);

    const auto userWorkspaceBytes = static_cast<int64_t>(rowNum) * static_cast<int64_t>(rowLen) *
        static_cast<int64_t>(sizeof(float)) * static_cast<int64_t>(TOP_K_TOP_P_SAMPLE_WORKSPACE_FACTOR);
    const auto workspaceBytes = TOP_K_TOP_P_SAMPLE_SYSTEM_WORKSPACE_BYTES + userWorkspaceBytes;
    AclrtBuffer workspace;
    const auto workspaceAllocRet = aclrtMalloc(&workspace.ptr, workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    TORCH_CHECK(workspaceAllocRet == ACL_ERROR_NONE, "aclrtMalloc for workspace failed with error code ", workspaceAllocRet);

    const auto launchRet = launch(
        tiling.numCore,
        aclStream,
        DataPtr(logits),
        DataPtr(topKs),
        DataPtr(topPs),
        DataPtr(q),
        logitsSelectIdxRaw.ptr,
        logitsTopKpSelectRaw.ptr,
        workspace.ptr,
        tilingNpu.ptr);
    TORCH_CHECK(launchRet == 0, "aclrtlaunch_top_k_top_p_sample failed with error code ", launchRet);

    const auto syncRet = aclrtSynchronizeStream(aclStream);
    TORCH_CHECK(syncRet == ACL_ERROR_NONE, "aclrtSynchronizeStream failed with error code ", syncRet);

    const auto logitsSelectIdxCopyRet = aclrtMemcpy(
        logitsSelectIdx.data_ptr(),
        logitsSelectIdxBytes,
        logitsSelectIdxRaw.ptr,
        logitsSelectIdxBytes,
        ACL_MEMCPY_DEVICE_TO_DEVICE);
    TORCH_CHECK(
        logitsSelectIdxCopyRet == ACL_ERROR_NONE,
        "aclrtMemcpy for logitsSelectIdx failed with error code ",
        logitsSelectIdxCopyRet);

    const auto logitsTopKpSelectCopyRet = aclrtMemcpy(
        logitsTopKpSelect.data_ptr(),
        logitsTopKpSelectBytes,
        logitsTopKpSelectRaw.ptr,
        logitsTopKpSelectBytes,
        ACL_MEMCPY_DEVICE_TO_DEVICE);
    TORCH_CHECK(
        logitsTopKpSelectCopyRet == ACL_ERROR_NONE,
        "aclrtMemcpy for logitsTopKpSelect failed with error code ",
        logitsTopKpSelectCopyRet);
}

pybind11::tuple run_top_k_top_p_sample(
    const at::Tensor &logits,
    const at::Tensor &topKs,
    const at::Tensor &topPs,
    const at::Tensor &q,
    bool isNeedLogits = false,
    int64_t topKGuess = 32,
    double eps = 1e-8)
{
    CheckInputs(logits, topKs, topPs, q);
    TORCH_CHECK(topKGuess >= 0, "topKGuess must be non-negative");
    TORCH_CHECK(topKGuess <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()), "topKGuess is too large");
    TORCH_CHECK(logits.sizes()[0] <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()), "rowNum is too large");
    TORCH_CHECK(logits.sizes()[1] <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()), "rowLen is too large");

    const auto rowNum = logits.sizes()[0];
    const auto rowLen = logits.sizes()[1];

    auto logitsSelectIdx = at::empty({rowNum}, logits.options().dtype(at::kLong));
    auto logitsTopKpSelect = at::full(
        {rowNum, rowLen},
        -std::numeric_limits<float>::infinity(),
        logits.options().dtype(at::kFloat));

    if (rowNum == 0 || rowLen == 0) {
        return pybind11::make_tuple(logitsSelectIdx, logitsTopKpSelect);
    }

    const auto launch = GetLaunchFn(logits);
    const auto currentStream = c10_npu::getCurrentNPUStream(static_cast<c10::DeviceIndex>(logits.get_device()));
    auto aclStream = currentStream.stream(false);

    constexpr int64_t kMaxRowsPerLaunch = TOP_K_TOP_P_SAMPLE_DEFAULT_VECTOR_CORES - 1;
    for (int64_t rowStart = 0; rowStart < rowNum; rowStart += kMaxRowsPerLaunch) {
        const auto chunkRows = std::min(kMaxRowsPerLaunch, rowNum - rowStart);
        auto logitsChunk = logits.narrow(0, rowStart, chunkRows);
        auto topKsChunk = topKs.narrow(0, rowStart, chunkRows);
        auto topPsChunk = topPs.narrow(0, rowStart, chunkRows);
        auto qChunk = q.narrow(0, rowStart, chunkRows);
        auto logitsSelectIdxChunk = logitsSelectIdx.narrow(0, rowStart, chunkRows);
        auto logitsTopKpSelectChunk = logitsTopKpSelect.narrow(0, rowStart, chunkRows);
        RunSingleChunk(
            logitsChunk,
            topKsChunk,
            topPsChunk,
            qChunk,
            logitsSelectIdxChunk,
            logitsTopKpSelectChunk,
            isNeedLogits,
            static_cast<uint32_t>(topKGuess),
            static_cast<float>(eps),
            launch,
            aclStream);
    }

    return pybind11::make_tuple(logitsSelectIdx, logitsTopKpSelect);
}

} // namespace top_k_top_p_sample_ext

PYBIND11_MODULE(_top_k_top_p_sample_ext, m)
{
    m.doc() = "top_k_top_p_sample AscendC extension";
    m.def("run_top_k_top_p_sample", &top_k_top_p_sample_ext::run_top_k_top_p_sample, "");
}
