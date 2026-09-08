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
 * \file cheevj_diagonal.hpp
 * \brief Diagonal-matrix fast path for Cheevj large-size device execution.
 */

#ifndef CHEEVJ_C64_DIAGONAL_HPP
#define CHEEVJ_C64_DIAGONAL_HPP

#include <cstdint>

#include "cheevj_workspace.hpp"
#include "kernel_operator.h"

using namespace AscendC;

namespace Cheevj
{

constexpr int CHEEVJ_DIAGONAL_TILE_LENGTH = 8192;
constexpr float CHEEVJ_DIAGONAL_OFFDIAG_TOL = 1.0e-12f;

class CheevjDiagonalFastPath
{
   public:
    __aicore__ inline CheevjDiagonalFastPath() {}

    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe)
    {
        pipe->InitBuffer(inRealQueue, 1, CHEEVJ_DIAGONAL_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(inImagQueue, 1, CHEEVJ_DIAGONAL_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(outQueue, 1, CHEEVJ_DIAGONAL_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(workBuf, CHEEVJ_DIAGONAL_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(scalarBuf, 8 * sizeof(float));
    }

    __aicore__ inline void SetMatrix(GM_ADDR compactInterleaved, GM_ADDR w,
                                     const CheevjPlanarWorkspace &planarWorkspace, bool computeVectors)
    {
        compactGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(compactInterleaved));
        wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(w));
        planar = planarWorkspace;
        needVectors = computeVectors;
    }

    __aicore__ inline bool Process()
    {
        CopyDiagonalToW();
        ZeroWorkspaceDiagonal();
        const float offdiagMax = ComputeOffdiagAbsMax();
        if (offdiagMax > CHEEVJ_DIAGONAL_OFFDIAG_TOL)
        {
            RestoreWorkspaceDiagonal();
            return false;
        }
        RestoreWorkspaceDiagonal();
        return true;
    }

   private:
    TQue<QuePosition::VECIN, 1> inRealQueue;
    TQue<QuePosition::VECIN, 1> inImagQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECCALC> scalarBuf;
    GlobalTensor<float> compactGlobal;
    GlobalTensor<float> wGlobal;
    CheevjPlanarWorkspace planar;
    bool needVectors;

    __aicore__ inline int PlanarDiagonalOffset(int idx) const { return idx * planar.layout.strideN + idx; }

    __aicore__ inline int CompactComplexOffset(int row, int col) const { return (row + col * planar.layout.n) * 2; }

    __aicore__ inline void CopyDiagonalToW()
    {
        LocalTensor<float> oneLocal = scalarBuf.Get<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        for (int idx = 0; idx < planar.layout.n; ++idx)
        {
            DataCopyPad(oneLocal, planar.aReal[PlanarDiagonalOffset(idx)], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(wGlobal[idx], oneLocal, copyParams);
        }
    }

    __aicore__ inline void ZeroWorkspaceDiagonal()
    {
        LocalTensor<float> zeroLocal = scalarBuf.Get<float>();
        Duplicate(zeroLocal, 0.0f, 1);
        PipeBarrier<PIPE_V>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        for (int idx = 0; idx < planar.layout.n; ++idx)
        {
            const int offset = PlanarDiagonalOffset(idx);
            DataCopyPad(planar.aReal[offset], zeroLocal, copyParams);
            DataCopyPad(planar.aImag[offset], zeroLocal, copyParams);
        }
    }

    __aicore__ inline void RestoreWorkspaceDiagonal()
    {
        LocalTensor<float> oneLocal = scalarBuf.Get<float>();
        LocalTensor<float> zeroLocal = workBuf.Get<float>();
        Duplicate(zeroLocal, 0.0f, 1);
        PipeBarrier<PIPE_V>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        for (int idx = 0; idx < planar.layout.n; ++idx)
        {
            const int offset = PlanarDiagonalOffset(idx);
            DataCopyPad(oneLocal, wGlobal[idx], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(planar.aReal[offset], oneLocal, copyParams);
            DataCopyPad(planar.aImag[offset], zeroLocal, copyParams);
            DataCopyPad(planar.aImagNeg[offset], zeroLocal, copyParams);
        }
    }

    __aicore__ inline float ComputeOffdiagAbsMax()
    {
        float offdiagMax = 0.0f;
        const int totalElems = planar.layout.planeElems;
        const int tileCount = AlignUp(totalElems, CHEEVJ_DIAGONAL_TILE_LENGTH) / CHEEVJ_DIAGONAL_TILE_LENGTH;
        for (int tileIdx = 0; tileIdx < tileCount; ++tileIdx)
        {
            const int offset = tileIdx * CHEEVJ_DIAGONAL_TILE_LENGTH;
            const int elems = MinInt(CHEEVJ_DIAGONAL_TILE_LENGTH, totalElems - offset);
            offdiagMax = MaxScalar(offdiagMax, ComputeTileAbsMax(offset, elems));
        }
        return offdiagMax;
    }

    __aicore__ inline float ComputeTileAbsMax(int offset, int elems)
    {
        LocalTensor<float> realLocal = inRealQueue.AllocTensor<float>();
        LocalTensor<float> imagLocal = inImagQueue.AllocTensor<float>();
        const uint32_t bytes = static_cast<uint32_t>(elems * sizeof(float));
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(realLocal, planar.aReal[offset], copyParams, padParams);
        DataCopyPad(imagLocal, planar.aImag[offset], copyParams, padParams);
        inRealQueue.EnQue(realLocal);
        inImagQueue.EnQue(imagLocal);

        realLocal = inRealQueue.DeQue<float>();
        imagLocal = inImagQueue.DeQue<float>();
        LocalTensor<float> workLocal = workBuf.Get<float>();

        Mul(realLocal, realLocal, realLocal, elems);
        Mul(imagLocal, imagLocal, imagLocal, elems);
        PipeBarrier<PIPE_V>();
        Add(workLocal, realLocal, imagLocal, elems);
        PipeBarrier<PIPE_V>();
        ReduceMax(workLocal, workLocal, workLocal, elems, true);
        PipeBarrier<PIPE_ALL>();
        const float tileMax = workLocal.GetValue(0);

        inRealQueue.FreeTensor(realLocal);
        inImagQueue.FreeTensor(imagLocal);
        return tileMax;
    }

    __aicore__ inline void WriteCompactIdentity()
    {
        ZeroCompactOutput();
        LocalTensor<float> oneLocal = scalarBuf.Get<float>();
        Duplicate(oneLocal, 1.0f, 1);
        PipeBarrier<PIPE_V>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        for (int idx = 0; idx < planar.layout.n; ++idx)
        {
            DataCopyPad(compactGlobal[CompactComplexOffset(idx, idx)], oneLocal, copyParams);
        }
    }

    __aicore__ inline void ZeroCompactOutput()
    {
        const int totalFloats = planar.layout.n * planar.layout.n * 2;
        const int tileCount = AlignUp(totalFloats, CHEEVJ_DIAGONAL_TILE_LENGTH) / CHEEVJ_DIAGONAL_TILE_LENGTH;
        for (int tileIdx = 0; tileIdx < tileCount; ++tileIdx)
        {
            const int offset = tileIdx * CHEEVJ_DIAGONAL_TILE_LENGTH;
            const int elems = MinInt(CHEEVJ_DIAGONAL_TILE_LENGTH, totalFloats - offset);
            LocalTensor<float> zeroLocal = outQueue.AllocTensor<float>();
            Duplicate(zeroLocal, 0.0f, elems);
            PipeBarrier<PIPE_V>();
            outQueue.EnQue(zeroLocal);

            zeroLocal = outQueue.DeQue<float>();
            const uint32_t bytes = static_cast<uint32_t>(elems * sizeof(float));
            DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
            DataCopyPad(compactGlobal[offset], zeroLocal, copyParams);
            outQueue.FreeTensor(zeroLocal);
        }
    }

    __aicore__ inline float MaxScalar(float lhs, float rhs) const { return lhs > rhs ? lhs : rhs; }
};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_DIAGONAL_HPP
