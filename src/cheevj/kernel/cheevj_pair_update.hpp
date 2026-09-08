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
 * \file cheevj_pair_update.hpp
 * \brief SIMD pair-column update primitives for the Cheevj planar workspace.
 */

#ifndef CHEEVJ_C64_PAIR_UPDATE_HPP
#define CHEEVJ_C64_PAIR_UPDATE_HPP

#include <cstdint>

#include "cheevj_ascendc_symbols.hpp"
#include "cheevj_workspace.hpp"
#include "complex_vec.hpp"
#include "kernel_operator.h"

namespace Cheevj
{

constexpr int CHEEVJ_PAIR_TILE_LENGTH = 256;

class CheevjPairColumnUpdater
{
   public:
    __aicore__ inline CheevjPairColumnUpdater() {}

    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe)
    {
        pipe->InitBuffer(pRealBuf, CHEEVJ_PAIR_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(pImagBuf, CHEEVJ_PAIR_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(qRealBuf, CHEEVJ_PAIR_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(qImagBuf, CHEEVJ_PAIR_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(outPRealBuf, CHEEVJ_PAIR_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(outPImagBuf, CHEEVJ_PAIR_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(outQRealBuf, CHEEVJ_PAIR_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(outQImagBuf, CHEEVJ_PAIR_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(scratchBuf, CHEEVJ_PAIR_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(rowStageRealBuf, CHEEVJ_PAIR_TILE_LENGTH * 8 * sizeof(float));
        pipe->InitBuffer(rowStageImagBuf, CHEEVJ_PAIR_TILE_LENGTH * 8 * sizeof(float));
        pipe->InitBuffer(rowStageImagNegBuf, CHEEVJ_PAIR_TILE_LENGTH * 8 * sizeof(float));
    }

    __aicore__ inline void SetWorkspace(const CheevjPlanarWorkspace &planarWorkspace) { planar = planarWorkspace; }

    __aicore__ inline void RotateMatrixColumns(int p, int q, float c, float sReal, float sImag)
    {
        RotateColumnPair(planar.aReal, planar.aImag, planar.aImagNeg, p, q, c, sReal, sImag, true);
    }

    __aicore__ inline void RotateMatrixColumnsSingleCore(int p, int q, float c, float sReal, float sImag)
    {
        RotateColumnPairSingleCore(planar.aReal, planar.aImag, planar.aImagNeg, p, q, c, sReal, sImag, true);
    }

    __aicore__ inline void RotateMatrixTwoSidedSingleCore(int p, int q, float c, float sReal, float sImag)
    {
        for (int rowOffset = 0; rowOffset < planar.layout.n; rowOffset += CHEEVJ_PAIR_TILE_LENGTH)
        {
            const int elems = MinInt(CHEEVJ_PAIR_TILE_LENGTH, planar.layout.n - rowOffset);
            RotateMatrixTwoSidedTile(p, q, rowOffset, elems, c, sReal, sImag);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void RotateVectorColumns(int p, int q, float c, float sReal, float sImag)
    {
        RotateColumnPair(planar.vReal, planar.vImag, planar.vImag, p, q, c, sReal, sImag, false);
    }

    __aicore__ inline void RotateVectorColumnsSingleCore(int p, int q, float c, float sReal, float sImag)
    {
        RotateColumnPairSingleCore(planar.vReal, planar.vImag, planar.vImag, p, q, c, sReal, sImag, false);
    }

    __aicore__ inline void MirrorMatrixColumnsToRows(int p, int q)
    {
        MirrorColumnToRow(planar.aReal, planar.aImag, planar.aImagNeg, p);
        MirrorColumnToRow(planar.aReal, planar.aImag, planar.aImagNeg, q);
        CheevjWorkspaceSync();
    }

    __aicore__ inline void MirrorMatrixColumnsToRowsSingleCore(int p, int q)
    {
        MirrorColumnToRowSingleCore(planar.aReal, planar.aImag, planar.aImagNeg, p);
        MirrorColumnToRowSingleCore(planar.aReal, planar.aImag, planar.aImagNeg, q);
    }

   private:
    TBuf<TPosition::VECCALC> pRealBuf;
    TBuf<TPosition::VECCALC> pImagBuf;
    TBuf<TPosition::VECCALC> qRealBuf;
    TBuf<TPosition::VECCALC> qImagBuf;
    TBuf<TPosition::VECCALC> outPRealBuf;
    TBuf<TPosition::VECCALC> outPImagBuf;
    TBuf<TPosition::VECCALC> outQRealBuf;
    TBuf<TPosition::VECCALC> outQImagBuf;
    TBuf<TPosition::VECCALC> scratchBuf;
    TBuf<TPosition::VECCALC> rowStageRealBuf;
    TBuf<TPosition::VECCALC> rowStageImagBuf;
    TBuf<TPosition::VECCALC> rowStageImagNegBuf;
    CheevjPlanarWorkspace planar;

    __aicore__ inline int ColumnOffset(int col, int rowOffset) const { return col * planar.layout.strideN + rowOffset; }

    __aicore__ inline void RotateColumnPair(GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane,
                                            GlobalTensor<float> imagNegPlane, int p, int q, float c, float sReal,
                                            float sImag, bool updateImagNeg)
    {
        const int workerCount = CheevjWorkspaceWorkerCount();
        const int rowsPerWorker = PairRowsPerWorker(workerCount);
        for (int rowOffset = GetBlockIdx() * rowsPerWorker; rowOffset < planar.layout.n;
             rowOffset += workerCount * rowsPerWorker)
        {
            const int elems = MinInt(rowsPerWorker, planar.layout.n - rowOffset);
            RotateColumnTile(realPlane, imagPlane, imagNegPlane, p, q, rowOffset, elems, c, sReal, sImag,
                             updateImagNeg);
        }
        CheevjWorkspaceSync();
    }

    __aicore__ inline void RotateColumnPairSingleCore(GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane,
                                                      GlobalTensor<float> imagNegPlane, int p, int q, float c,
                                                      float sReal, float sImag, bool updateImagNeg)
    {
        for (int rowOffset = 0; rowOffset < planar.layout.n; rowOffset += CHEEVJ_PAIR_TILE_LENGTH)
        {
            const int elems = MinInt(CHEEVJ_PAIR_TILE_LENGTH, planar.layout.n - rowOffset);
            RotateColumnTile(realPlane, imagPlane, imagNegPlane, p, q, rowOffset, elems, c, sReal, sImag,
                             updateImagNeg);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void RotateColumnTile(GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane,
                                            GlobalTensor<float> imagNegPlane, int p, int q, int rowOffset, int elems,
                                            float c, float sReal, float sImag, bool updateImagNeg)
    {
        LoadColumnPair(realPlane, imagPlane, p, q, rowOffset, elems);
        RotateLoadedPair(c, sReal, sImag, elems);
        StoreColumnPair(realPlane, imagPlane, p, q, rowOffset, elems);

        LocalTensor<float> outPImag = outPImagBuf.Get<float>();
        LocalTensor<float> outQImag = outQImagBuf.Get<float>();
        LocalTensor<float> scratch = scratchBuf.Get<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0};

        if (updateImagNeg)
        {
            Muls(scratch, outPImag, -1.0f, elems);
            PipeBarrier<PIPE_V>();
            DataCopyPad(imagNegPlane[ColumnOffset(p, rowOffset)], scratch, copyParams);
            Muls(scratch, outQImag, -1.0f, elems);
            PipeBarrier<PIPE_V>();
            DataCopyPad(imagNegPlane[ColumnOffset(q, rowOffset)], scratch, copyParams);
        }
    }

    __aicore__ inline void RotateMatrixTwoSidedTile(int p, int q, int rowOffset, int elems, float c, float sReal,
                                                    float sImag)
    {
        LoadColumnPair(planar.aReal, planar.aImag, p, q, rowOffset, elems);
        RotateLoadedPair(c, sReal, sImag, elems);
        StoreColumnPair(planar.aReal, planar.aImag, p, q, rowOffset, elems);

        LocalTensor<float> outPReal = outPRealBuf.Get<float>();
        LocalTensor<float> outPImag = outPImagBuf.Get<float>();
        LocalTensor<float> outQReal = outQRealBuf.Get<float>();
        LocalTensor<float> outQImag = outQImagBuf.Get<float>();
        LocalTensor<float> scratch = scratchBuf.Get<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0};

        Muls(scratch, outPImag, -1.0f, elems);
        PipeBarrier<PIPE_V>();
        DataCopyPad(planar.aImagNeg[ColumnOffset(p, rowOffset)], scratch, copyParams);
        Muls(scratch, outQImag, -1.0f, elems);
        PipeBarrier<PIPE_V>();
        DataCopyPad(planar.aImagNeg[ColumnOffset(q, rowOffset)], scratch, copyParams);

        for (int idx = 0; idx < elems; ++idx)
        {
            const int row = rowOffset + idx;
            const int rowP = row * planar.layout.strideN + p;
            planar.aReal.SetValue(rowP, outPReal.GetValue(idx));
            planar.aImag.SetValue(rowP, -outPImag.GetValue(idx));
            planar.aImagNeg.SetValue(rowP, outPImag.GetValue(idx));

            const int rowQ = row * planar.layout.strideN + q;
            planar.aReal.SetValue(rowQ, outQReal.GetValue(idx));
            planar.aImag.SetValue(rowQ, -outQImag.GetValue(idx));
            planar.aImagNeg.SetValue(rowQ, outQImag.GetValue(idx));
        }
    }

    __aicore__ inline void LoadColumnPair(GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane, int p, int q,
                                          int rowOffset, int elems)
    {
        LocalTensor<float> pReal = pRealBuf.Get<float>();
        LocalTensor<float> pImag = pImagBuf.Get<float>();
        LocalTensor<float> qReal = qRealBuf.Get<float>();
        LocalTensor<float> qImag = qImagBuf.Get<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(pReal, realPlane[ColumnOffset(p, rowOffset)], copyParams, padParams);
        DataCopyPad(pImag, imagPlane[ColumnOffset(p, rowOffset)], copyParams, padParams);
        DataCopyPad(qReal, realPlane[ColumnOffset(q, rowOffset)], copyParams, padParams);
        DataCopyPad(qImag, imagPlane[ColumnOffset(q, rowOffset)], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void RotateLoadedPair(float c, float sReal, float sImag, int elems)
    {
        LocalTensor<float> scratch = scratchBuf.Get<float>();
        ComplexRot2(outPRealBuf.Get<float>(), outPImagBuf.Get<float>(), outQRealBuf.Get<float>(),
                    outQImagBuf.Get<float>(), pRealBuf.Get<float>(), pImagBuf.Get<float>(), qRealBuf.Get<float>(),
                    qImagBuf.Get<float>(), c, sReal, sImag, scratch, scratch, scratch, scratch, elems);
    }

    __aicore__ inline void StoreColumnPair(GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane, int p, int q,
                                           int rowOffset, int elems)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0};
        DataCopyPad(realPlane[ColumnOffset(p, rowOffset)], outPRealBuf.Get<float>(), copyParams);
        DataCopyPad(imagPlane[ColumnOffset(p, rowOffset)], outPImagBuf.Get<float>(), copyParams);
        DataCopyPad(realPlane[ColumnOffset(q, rowOffset)], outQRealBuf.Get<float>(), copyParams);
        DataCopyPad(imagPlane[ColumnOffset(q, rowOffset)], outQImagBuf.Get<float>(), copyParams);
    }

    __aicore__ inline void MirrorColumnToRow(GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane,
                                             GlobalTensor<float> imagNegPlane, int col)
    {
        const int workerCount = CheevjWorkspaceWorkerCount();
        const int rowsPerWorker = PairRowsPerWorker(workerCount);
        for (int offset = GetBlockIdx() * rowsPerWorker; offset < planar.layout.n;
             offset += workerCount * rowsPerWorker)
        {
            const int elems = MinInt(rowsPerWorker, planar.layout.n - offset);
            MirrorColumnTile(realPlane, imagPlane, imagNegPlane, col, offset, elems);
        }
    }

    __aicore__ inline void MirrorColumnToRowSingleCore(GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane,
                                                       GlobalTensor<float> imagNegPlane, int col)
    {
        for (int offset = 0; offset < planar.layout.n; offset += CHEEVJ_PAIR_TILE_LENGTH)
        {
            const int elems = MinInt(CHEEVJ_PAIR_TILE_LENGTH, planar.layout.n - offset);
            MirrorColumnTile(realPlane, imagPlane, imagNegPlane, col, offset, elems);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline int PairRowsPerWorker() const { return PairRowsPerWorker(CheevjWorkspaceWorkerCount()); }

    __aicore__ inline int PairRowsPerWorker(int workerCount) const
    {
        const int activeWorkerCount = workerCount > 0 ? workerCount : 1;
        const int rows = AlignUp((planar.layout.n + activeWorkerCount - 1) / activeWorkerCount, 16);
        return MinInt(CHEEVJ_PAIR_TILE_LENGTH, rows > 0 ? rows : 16);
    }

    __aicore__ inline void MirrorColumnTile(GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane,
                                            GlobalTensor<float> imagNegPlane, int col, int offset, int elems)
    {
        LocalTensor<float> realLocal = pRealBuf.Get<float>();
        LocalTensor<float> imagLocal = pImagBuf.Get<float>();
        LocalTensor<float> imagNegLocal = scratchBuf.Get<float>();

        const uint32_t bytes = static_cast<uint32_t>(elems * sizeof(float));
        DataCopyExtParams colCopyParams{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(realLocal, realPlane[ColumnOffset(col, offset)], colCopyParams, padParams);
        DataCopyPad(imagLocal, imagPlane[ColumnOffset(col, offset)], colCopyParams, padParams);
        PipeBarrier<PIPE_ALL>();

#define CHEEVJ_JACOBI_COMMON_SECTION 6
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION
        const int rowOffset = offset * planar.layout.strideN + col;
        DataCopyPad(realPlane[rowOffset], rowStageReal, rowCopyParams);
        DataCopyPad(imagPlane[rowOffset], rowStageImag, rowCopyParams);
        DataCopyPad(imagNegPlane[rowOffset], rowStageImagNeg, rowCopyParams);
    }

};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_PAIR_UPDATE_HPP
