/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define CHEEVJ_TRIDIAG_COMMON_SECTION 1
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
    int rowWave = 0;

#define CHEEVJ_TRIDIAG_COMMON_SECTION 5
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
        AscendC::SyncAll<true>(barrierGlobal, barrierLocal, CHEEVJ_TRIDIAG2048_PARTICIPANTS);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadTridiagonal()
    {
        auto diagonal = diagonalBuf.Get<float>();
        auto offDiagonal = offDiagonalBuf.Get<float>();
        AscendC::DataCopyExtParams diagonalCopy{1, CHEEVJ_TRIDIAG2048_N * sizeof(float), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> diagonalPad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(diagonal, diagonalGlobal, diagonalCopy, diagonalPad);
        AscendC::DataCopyExtParams offDiagonalCopy{1, (CHEEVJ_TRIDIAG2048_N - 1) * sizeof(float), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> offDiagonalPad{true, 0, 1, 0.0f};
        AscendC::DataCopyPad(offDiagonal, offDiagonalGlobal, offDiagonalCopy, offDiagonalPad);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline float ScaleTridiagonal()
    {
        auto diagonal = diagonalBuf.Get<float>();
        auto offDiagonal = offDiagonalBuf.Get<float>();
        float scale = 0.0f;
        // These are recurrence controls, not Q matrix traffic.
        for (int index = 0; index < CHEEVJ_TRIDIAG2048_N; ++index)
        {
            scale = Max(scale, Abs(diagonal.GetValue(index)));
        }
        for (int index = 0; index + 1 < CHEEVJ_TRIDIAG2048_N; ++index)
        {
            scale = Max(scale, Abs(offDiagonal.GetValue(index)));
        }
        if (scale == 0.0f)
        {
            return 0.0f;
        }
        const float inverseScale = 1.0f / scale;
        AscendC::Muls(diagonal, diagonal, inverseScale, CHEEVJ_TRIDIAG2048_N);
        AscendC::Muls(offDiagonal, offDiagonal, inverseScale, CHEEVJ_TRIDIAG2048_N);
        AscendC::PipeBarrier<PIPE_V>();
        return scale;
    }

    __aicore__ inline void InitializeWorkerIdentity(int rowBegin)
    {
        auto localQ = localQBuf.Get<float>();
        auto rowIndex = scratch0Buf.Get<int32_t>();
        auto columnValue = scratch1Buf.Get<int32_t>();

        AscendC::CreateVecIndex(rowIndex, static_cast<int32_t>(rowBegin),
                                static_cast<uint32_t>(CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER));
        AscendC::PipeBarrier<PIPE_V>();
        for (int column = 0; column < CHEEVJ_TRIDIAG2048_N; ++column)
        {
            AscendC::Adds(columnValue, rowIndex, static_cast<int32_t>(-column), CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(columnValue, columnValue, columnValue, CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(columnValue, columnValue, static_cast<int32_t>(1), CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(columnValue, columnValue, static_cast<int32_t>(-1), CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(columnValue, columnValue, static_cast<int32_t>(1), CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(localQ[column * CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER], columnValue,
                          AscendC::RoundMode::CAST_NONE, CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline int FindSplit(int first)
    {
        auto diagonal = diagonalBuf.Get<float>();
        auto offDiagonal = offDiagonalBuf.Get<float>();
        for (int split = first; split + 1 < CHEEVJ_TRIDIAG2048_N; ++split)
        {
            const float edge = Abs(offDiagonal.GetValue(split));
            const float adjacent = Abs(diagonal.GetValue(split)) + Abs(diagonal.GetValue(split + 1));
            const float tolerance = Max(4.0f * CHEEVJ_TRIDIAG2048_EPSILON * adjacent, CHEEVJ_TRIDIAG2048_SAFE_MIN);
            if (edge <= tolerance)
            {
                offDiagonal.SetValue(split, 0.0f);
                return split;
            }
        }
        return CHEEVJ_TRIDIAG2048_N - 1;
    }
#define CHEEVJ_TRIDIAG_FIND_SPLIT_CLOSED
#define CHEEVJ_TRIDIAG_COMMON_SECTION 4
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
#undef CHEEVJ_TRIDIAG_FIND_SPLIT_CLOSED
        AscendC::DataCopyExtParams copy{1, CHEEVJ_TRIDIAG2048_COMMAND_INTS * sizeof(int32_t), 0, 0, 0};
#define CHEEVJ_TRIDIAG_COMMON_SECTION 2
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
        for (int first = 0; first < CHEEVJ_TRIDIAG2048_N; ++first)
        {
            for (int iteration = 0; iteration <= CHEEVJ_TRIDIAG2048_MAX_ITERATIONS; ++iteration)
            {
                const int split = FindSplit(first);
                if (split == first)
                {
                    break;
                }
                if (iteration == CHEEVJ_TRIDIAG2048_MAX_ITERATIONS)
#define CHEEVJ_TRIDIAG_COMMON_SECTION 7
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
        if (Abs(denominator) < CHEEVJ_TRIDIAG2048_SAFE_MIN)
        {
            denominator = CopySign(CHEEVJ_TRIDIAG2048_SAFE_MIN, denominator);
#define CHEEVJ_TRIDIAG_COMMON_SECTION 3
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
            if (radius <= CHEEVJ_TRIDIAG2048_SAFE_MIN)
#define CHEEVJ_TRIDIAG_COMMON_SECTION 6
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
            PublishCommand(CHEEVJ_TRIDIAG2048_ROTATE, index, cosine, sine, barrierLocal);
        }
        diagonal.SetValue(first, diagonal.GetValue(first) - correction);
        offDiagonal.SetValue(first, shift);
        offDiagonal.SetValue(split, 0.0f);
    }

    __aicore__ inline void SortEigenpairs(const AscendC::LocalTensor<int32_t> &barrierLocal)
    {
        auto diagonal = diagonalBuf.Get<float>();
        for (int output = 0; output + 1 < CHEEVJ_TRIDIAG2048_N; ++output)
        {
            int minimumIndex = output;
            float minimum = diagonal.GetValue(output);
            for (int candidate = output + 1; candidate < CHEEVJ_TRIDIAG2048_N; ++candidate)
#define CHEEVJ_TRIDIAG_COMMON_SECTION 10
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
                PublishCommand(CHEEVJ_TRIDIAG2048_SWAP, packed, 0.0f, 0.0f, barrierLocal);
            }
        }
    }

    __aicore__ inline bool StoreOrValidateEigenvalues(const AscendC::LocalTensor<float> &diagonal,
                                                      const AscendC::DataCopyExtParams &eigenvalueCopy)
    {
        if (rowWave == 0)
        {
            AscendC::DataCopyPad(eigenvaluesGlobal, diagonal, eigenvalueCopy);
            return false;
        }

        auto reference = scratch0Buf.Get<float>();
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(reference, eigenvaluesGlobal, eigenvalueCopy, pad);
        AscendC::PipeBarrier<PIPE_ALL>();
        auto diagonalBits = diagonal.template ReinterpretCast<uint32_t>();
        auto referenceBits = reference.template ReinterpretCast<uint32_t>();
        for (int index = 0; index < CHEEVJ_TRIDIAG2048_N; ++index)
        {
            if (diagonalBits.GetValue(index) != referenceBits.GetValue(index))
            {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline int MergeReplayInfo(const AscendC::LocalTensor<int32_t> &metadata, int convergenceInfo,
                                          bool replayMismatch)
    {
        if (rowWave == 0)
        {
            return convergenceInfo;
        }
        AscendC::DataCopyExtParams priorInfoCopy{1, sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> priorInfoPad{false, 0, 0, 0};
        AscendC::DataCopyPad(metadata, infoGlobal, priorInfoCopy, priorInfoPad);
        AscendC::PipeBarrier<PIPE_ALL>();
        const int priorInfo = metadata.GetValue(0);
        if (priorInfo != 0)
        {
            return priorInfo;
        }
        return replayMismatch ? CHEEVJ_TRIDIAG2048_N + rowWave : convergenceInfo;
    }

    __aicore__ inline void StoreControllerResults(float scale, int convergenceInfo)
    {
        auto diagonal = diagonalBuf.Get<float>();
        if (scale != 0.0f)
        {
            AscendC::Muls(diagonal, diagonal, scale, CHEEVJ_TRIDIAG2048_N);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::DataCopyExtParams eigenvalueCopy{1, CHEEVJ_TRIDIAG2048_N * sizeof(float), 0, 0, 0};

        const bool replayMismatch = StoreOrValidateEigenvalues(diagonal, eigenvalueCopy);
        auto metadata = commandBuf.Get<int32_t>();
        convergenceInfo = MergeReplayInfo(metadata, convergenceInfo, replayMismatch);
#define CHEEVJ_TRIDIAG_COMMON_SECTION 8
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
        PublishCommand(CHEEVJ_TRIDIAG2048_STOP, 0, 0.0f, 0.0f, barrierLocal);
    }

    __aicore__ inline void LoadCommand()
    {
        auto command = commandBuf.Get<int32_t>();
        AscendC::DataCopyExtParams copy{1, CHEEVJ_TRIDIAG2048_COMMAND_INTS * sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(command, commandGlobal, copy, pad);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void RotateWorkerColumns(int column, float cosine, float sine)
    {
        auto localQ = localQBuf.Get<float>();
        auto scratch0 = scratch0Buf.Get<float>();
        auto scratch1 = scratch1Buf.Get<float>();
        auto current = localQ[column * CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER];
        auto next = localQ[(column + 1) * CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER];

        AscendC::Muls(scratch0, next, cosine, CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
        AscendC::Muls(scratch1, next, -sine, CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(next, current, sine, CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
        AscendC::Muls(current, current, cosine, CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(next, next, scratch0, CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
        AscendC::Add(current, current, scratch1, CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SwapWorkerColumns(int lhs, int rhs)
    {
        auto localQ = localQBuf.Get<float>();
        auto scratch = scratch0Buf.Get<float>();
        auto lhsColumn = localQ[lhs * CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER];
        auto rhsColumn = localQ[rhs * CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER];
        AscendC::Adds(scratch, lhsColumn, 0.0f, CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(lhsColumn, rhsColumn, 0.0f, CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(rhsColumn, scratch, 0.0f, CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrepareGatherIndex()
    {
        auto signedIndex = gatherIndexBuf.Get<int32_t>();
        auto index = signedIndex.template ReinterpretCast<uint32_t>();
        AscendC::CreateVecIndex(signedIndex, static_cast<int32_t>(0),
                                static_cast<uint32_t>(CHEEVJ_TRIDIAG2048_GATHER_TILE));
        AscendC::PipeBarrier<PIPE_V>();
        // One column is 16 floats = 64 bytes in the worker-local layout.
        AscendC::ShiftLeft(index, index, static_cast<uint32_t>(6), CHEEVJ_TRIDIAG2048_GATHER_TILE);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreWorkerRows(int rowBegin)
    {
        auto localQ = localQBuf.Get<float>();
        auto row = scratch0Buf.Get<float>();
        auto index = gatherIndexBuf.Get<uint32_t>();
        PrepareGatherIndex();
        for (int localRow = 0; localRow < CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER; ++localRow)
        {
            // The 64-lane tiled form is deliberate: long-count Gather was
            // unreliable with a whole-leaf transpose at this tile width.
            for (int column = 0; column < CHEEVJ_TRIDIAG2048_N; column += CHEEVJ_TRIDIAG2048_GATHER_TILE)
            {
                AscendC::Gather(row[column], localQ[column * CHEEVJ_TRIDIAG2048_ROWS_PER_WORKER + localRow], index, 0,
                                CHEEVJ_TRIDIAG2048_GATHER_TILE);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyExtParams copy{1, CHEEVJ_TRIDIAG2048_N * sizeof(float), 0, 0, 0};
            AscendC::DataCopyPad(eigenvectorsGlobal[(rowBegin + localRow) * CHEEVJ_TRIDIAG2048_N], row, copy);
#define CHEEVJ_TRIDIAG_COMMON_SECTION 9
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
            if (operation == CHEEVJ_TRIDIAG2048_ROTATE)
            {
                RotateWorkerColumns(index, commandFloat.GetValue(16), commandFloat.GetValue(24));
            }
            else if (operation == CHEEVJ_TRIDIAG2048_SWAP)
            {
                const int lhs = index & 0xffff;
                const int rhs = (index >> 16) & 0xffff;
                SwapWorkerColumns(lhs, rhs);
            }
            else if (operation == CHEEVJ_TRIDIAG2048_STOP)
            {
                StoreWorkerRows(rowBegin);
            }
            CommandBarrier(barrierLocal);
            running = operation != CHEEVJ_TRIDIAG2048_STOP;
        }
    }
