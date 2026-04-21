#pragma once

#include <cstdint>

struct TopKTopPSampleTilingData {
    uint32_t numCore;
    uint32_t rowNum;
    uint32_t rowLen;
    uint32_t headCoreNum;
    uint32_t perHeadCoreRowNum;
    uint32_t tailCoreRowNum;
    uint32_t perHeadCorePartNum;
    uint32_t tailCorePartNum;
    uint32_t innerLoopEle;
    uint32_t innerLoopTime;
    uint32_t innerLoopEleTail;
    uint32_t innerLoopEleTailPad;
    uint32_t softmaxLoopTime;
    uint32_t softmaxLoopEleTail;
    uint32_t softmaxLoopEleTailPad;
    uint32_t eightKPartNum;
    uint32_t eightKPartTail;
    uint32_t eightKPartTailPad;
    uint32_t mrgMode;
    uint32_t headOffset;
    uint32_t isNeedLogits;
    float eps;
    uint32_t topKGuess;
};

constexpr uint32_t TOP_K_TOP_P_SAMPLE_DEFAULT_VECTOR_CORES = 40;
constexpr uint32_t TOP_K_TOP_P_SAMPLE_INNER_LOOP_ELE = 4096U * 2U;
constexpr uint32_t TOP_K_TOP_P_SAMPLE_SOFTMAX_INNER_LOOP_ELE = 32768U / sizeof(float);
constexpr uint32_t TOP_K_TOP_P_SAMPLE_SORT_PER_MAX = 1024U;
constexpr uint32_t TOP_K_TOP_P_SAMPLE_WORKSPACE_FACTOR = 6U;

inline uint32_t CeilDivU32(uint32_t lhs, uint32_t rhs)
{
    return rhs == 0U ? 0U : (lhs + rhs - 1U) / rhs;
}

inline uint32_t AlignUpU32(uint32_t value, uint32_t align)
{
    return align == 0U ? 0U : CeilDivU32(value, align) * align;
}
