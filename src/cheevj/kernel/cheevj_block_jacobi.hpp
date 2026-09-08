/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cheevj_block_jacobi.hpp
 * \brief  block-Jacobi helper for large dense complex Hermitian Cheevj.
 *
 * This helper implements the large-n block path around the planar workspace:
 *   1. round-robin scheduling of 32-column block pairs,
 *   2. extraction of a 64x64 principal block into UB,
 *   3. local two-sided Jacobi on that block while accumulating a 64x64 unitary,
 *   4. out-of-place right multiplication of global planar columns through
 *      scratch planes, followed by Hermitian row repair from scratch/local tile data,
 *   5. optional accumulation of V with the same local unitary.
 */

#ifndef CHEEVJ_C64_BLOCK_JACOBI_HPP
#define CHEEVJ_C64_BLOCK_JACOBI_HPP

#include <cstdint>

#include "cheevj_workspace.hpp"
#include "cheevj_pair_schedule.hpp"
#include "cheevj_ascendc_symbols.hpp"
#include "complex_vec.hpp"
#include "kernel_operator.h"

namespace Cheevj
{

constexpr int CHEEVJ_BLOCK_JACOBI_BLOCK_SIZE = 32;
constexpr int CHEEVJ_BLOCK_JACOBI_LOCAL_DIM = 64;
constexpr int CHEEVJ_BLOCK_JACOBI_LOCAL_ELEMS = CHEEVJ_BLOCK_JACOBI_LOCAL_DIM * CHEEVJ_BLOCK_JACOBI_LOCAL_DIM;
constexpr int CHEEVJ_BLOCK_JACOBI_ROW_TILE = 256;
constexpr int CHEEVJ_BLOCK_JACOBI_LOCAL_SWEEPS = 32;
constexpr int CHEEVJ_BLOCK_JACOBI_SCRATCH_PLANES = 2;
constexpr float CHEEVJ_BLOCK_JACOBI_REL_TOL = 1.0e-6f;
constexpr float CHEEVJ_BLOCK_JACOBI_MIN_PIVOT = 1.0e-20f;

struct CheevjComplex32
{
    float real;
    float imag;
};

using CheevjBlockRange = CheevjRoundRobinRange;
using CheevjBlockPair = CheevjRoundRobinPair;

__aicore__ inline CheevjComplex32 BlockMakeComplex(float real, float imag)
{
    CheevjComplex32 value{real, imag};
    return value;
}

__aicore__ inline float BlockAbsFloat(float value) { return value >= 0.0f ? value : -value; }

__aicore__ inline float BlockMaxFloat(float lhs, float rhs) { return lhs > rhs ? lhs : rhs; }

__aicore__ inline float BlockSqrtApprox(float value)
{
    if (value <= 0.0f)
    {
        return 0.0f;
    }
    float root = value > 1.0f ? value : 1.0f;
    for (int iter = 0; iter < 12; ++iter)
    {
        root = 0.5f * (root + value / root);
    }
    return root;
}

__aicore__ inline float ComplexAbs(CheevjComplex32 value)
{
    return BlockSqrtApprox(value.real * value.real + value.imag * value.imag);
}

__aicore__ inline int Mod(int value, int divisor)
{
    const int safeDivisor = divisor != 0 ? divisor : 1;
    const int result = value % safeDivisor;
    return result >= 0 ? result : result + divisor;
}

class CheevjBlockPairSchedule : public CheevjRoundRobinPairSchedule
{
   public:
    __aicore__ inline void Reset(int matrixN)
    {
        CheevjRoundRobinPairSchedule::Reset(matrixN, CHEEVJ_BLOCK_JACOBI_BLOCK_SIZE);
    }
};

class CheevjBlockJacobi
{
   public:
    __aicore__ inline CheevjBlockJacobi() : computeVectors(false), maxLocalSweeps(CHEEVJ_BLOCK_JACOBI_LOCAL_SWEEPS) {}

    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe)
    {
        pipe->InitBuffer(blockRealBuf, CHEEVJ_BLOCK_JACOBI_LOCAL_ELEMS * sizeof(float));
        pipe->InitBuffer(blockImagBuf, CHEEVJ_BLOCK_JACOBI_LOCAL_ELEMS * sizeof(float));
        pipe->InitBuffer(unitaryRealBuf, CHEEVJ_BLOCK_JACOBI_LOCAL_ELEMS * sizeof(float));
        pipe->InitBuffer(unitaryImagBuf, CHEEVJ_BLOCK_JACOBI_LOCAL_ELEMS * sizeof(float));
        pipe->InitBuffer(srcRealBuf, CHEEVJ_BLOCK_JACOBI_ROW_TILE * sizeof(float));
        pipe->InitBuffer(srcImagBuf, CHEEVJ_BLOCK_JACOBI_ROW_TILE * sizeof(float));
        pipe->InitBuffer(accRealBuf, CHEEVJ_BLOCK_JACOBI_ROW_TILE * sizeof(float));
        pipe->InitBuffer(accImagBuf, CHEEVJ_BLOCK_JACOBI_ROW_TILE * sizeof(float));
        pipe->InitBuffer(outRealBuf, CHEEVJ_BLOCK_JACOBI_ROW_TILE * sizeof(float));
        pipe->InitBuffer(outImagBuf, CHEEVJ_BLOCK_JACOBI_ROW_TILE * sizeof(float));
        pipe->InitBuffer(tmp0Buf, CHEEVJ_BLOCK_JACOBI_ROW_TILE * sizeof(float));
        pipe->InitBuffer(tmp1Buf, CHEEVJ_BLOCK_JACOBI_ROW_TILE * sizeof(float));
        pipe->InitBuffer(rowStageRealBuf, CHEEVJ_BLOCK_JACOBI_ROW_TILE * 8 * sizeof(float));
        pipe->InitBuffer(rowStageImagBuf, CHEEVJ_BLOCK_JACOBI_ROW_TILE * 8 * sizeof(float));
        pipe->InitBuffer(rowStageImagNegBuf, CHEEVJ_BLOCK_JACOBI_ROW_TILE * 8 * sizeof(float));
    }

    __aicore__ inline void SetWorkspace(const CheevjPlanarWorkspace& planarWorkspace, bool needVectors)
    {
        planar = planarWorkspace;
        computeVectors = needVectors;
        schedule.Reset(planar.layout.n);
    }

    __aicore__ inline void SetMaxLocalSweeps(int sweeps)
    {
        maxLocalSweeps = sweeps > 0 ? sweeps : CHEEVJ_BLOCK_JACOBI_LOCAL_SWEEPS;
    }

    __aicore__ inline bool RequiresScratchPlanes() const
    {
        return planar.layout.scratchPlanes >= CHEEVJ_BLOCK_JACOBI_SCRATCH_PLANES;
    }

    __aicore__ inline bool ProcessOuterSweeps(int outerSweeps)
    {
        return CheevjProcessOuterSweeps(*this, outerSweeps, RequiresScratchPlanes());
    }

    __aicore__ inline bool ProcessOneSweep()
    {
        bool allLocalConverged = true;
        const int stages = schedule.StageCount();
        for (int stage = 0; stage < stages; ++stage)
        {
            const int pairCount = schedule.PairCountPerStage();
            for (int pairIndex = 0; pairIndex < pairCount; ++pairIndex)
            {
                const CheevjBlockPair pair = schedule.PairForStage(stage, pairIndex);
                if (!pair.valid)
                {
                    continue;
                }
                allLocalConverged = ProcessPair(pair) && allLocalConverged;
                CheevjWorkspaceSync();
            }
        }
        return allLocalConverged;
    }

   private:
#include "cheevj_block_jacobi_buffers.inc"
    TBuf<TPosition::VECCALC> outRealBuf;
    TBuf<TPosition::VECCALC> outImagBuf;
    TBuf<TPosition::VECCALC> tmp0Buf;
    TBuf<TPosition::VECCALC> tmp1Buf;
    TBuf<TPosition::VECCALC> rowStageRealBuf;
    TBuf<TPosition::VECCALC> rowStageImagBuf;
    TBuf<TPosition::VECCALC> rowStageImagNegBuf;
    CheevjPlanarWorkspace planar;
    CheevjBlockPairSchedule schedule;
    bool computeVectors;
    int maxLocalSweeps;

#include "cheevj_block_jacobi_impl.inc"
};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_BLOCK_JACOBI_HPP
