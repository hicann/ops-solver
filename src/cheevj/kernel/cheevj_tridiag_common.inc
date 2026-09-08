/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if defined(CHEEVJ_TRIDIAG_COMMON_SECTION) && CHEEVJ_TRIDIAG_COMMON_SECTION == 1
    AscendC::GlobalTensor<float> diagonalGlobal;
    AscendC::GlobalTensor<float> offDiagonalGlobal;
    AscendC::GlobalTensor<float> eigenvaluesGlobal;
    AscendC::GlobalTensor<float> eigenvectorsGlobal;
    AscendC::GlobalTensor<int32_t> infoGlobal;
    AscendC::GlobalTensor<int32_t> commandGlobal;
    AscendC::GlobalTensor<int32_t> barrierGlobal;

    AscendC::TBuf<AscendC::TPosition::VECCALC> diagonalBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offDiagonalBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> localQBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratch0Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratch1Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherIndexBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> commandBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> barrierBuf;
#elif defined(CHEEVJ_TRIDIAG_COMMON_SECTION) && CHEEVJ_TRIDIAG_COMMON_SECTION == 2
        AscendC::DataCopyPad(commandGlobal, commandInt, copy);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PublishCommand(int32_t operation, int32_t index, float cosine, float sine,
                                          const AscendC::LocalTensor<int32_t> &barrierLocal)
    {
        StoreCommand(operation, index, cosine, sine);
        // First barrier publishes the command, second acknowledges completion
        // of the matching vector update on all 32 row slabs.
        CommandBarrier(barrierLocal);
        CommandBarrier(barrierLocal);
    }

    __aicore__ inline int ImplicitQl(const AscendC::LocalTensor<int32_t> &barrierLocal)
    {
#elif defined(CHEEVJ_TRIDIAG_COMMON_SECTION) && CHEEVJ_TRIDIAG_COMMON_SECTION == 3
        }
        const float shiftCorrection = denominator != 0.0f ? eFirst / denominator : 0.0f;
        shift = diagonal.GetValue(split) - dFirst + shiftCorrection;

        float sine = 1.0f;
        float cosine = 1.0f;
        float correction = 0.0f;
        for (int index = split - 1; index >= first; --index)
        {
            const float edgeTimesSine = sine * offDiagonal.GetValue(index);
            const float edgeTimesCosine = cosine * offDiagonal.GetValue(index);
            radius = SafeHypot(edgeTimesSine, shift);
            offDiagonal.SetValue(index + 1, radius);
#elif defined(CHEEVJ_TRIDIAG_COMMON_SECTION) && CHEEVJ_TRIDIAG_COMMON_SECTION == 4
#ifndef CHEEVJ_TRIDIAG_FIND_SPLIT_CLOSED
    }
#endif

    __aicore__ inline void StoreCommand(int32_t operation, int32_t index, float cosine, float sine)
    {
        auto commandInt = commandBuf.Get<int32_t>();
        auto commandFloat = commandBuf.Get<float>();
        // Each scalar is repeated over one aligned 32-byte block.  This
        // avoids scalar stores while keeping the consumer loads coherent.
        AscendC::Duplicate(commandInt, operation, 8);
        AscendC::Duplicate(commandInt[8], index, 8);
        AscendC::Duplicate(commandFloat[16], cosine, 8);
        AscendC::Duplicate(commandFloat[24], sine, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
#elif defined(CHEEVJ_TRIDIAG_COMMON_SECTION) && CHEEVJ_TRIDIAG_COMMON_SECTION == 5
    __aicore__ inline float Abs(float value) const { return value < 0.0f ? -value : value; }

    __aicore__ inline float Max(float lhs, float rhs) const { return lhs > rhs ? lhs : rhs; }

    __aicore__ inline float CopySign(float magnitude, float sign) const
    {
        return sign < 0.0f ? -Abs(magnitude) : Abs(magnitude);
    }

    __aicore__ inline float SqrtOneToTwo(float value) const
    {
        float root = 1.0f;
        for (int iteration = 0; iteration < 6; ++iteration)
        {
            root = 0.5f * (root + value / root);
        }
        return root;
    }

    __aicore__ inline float SafeHypot(float lhs, float rhs) const
    {
        float high = Abs(lhs);
        float low = Abs(rhs);
        if (low > high)
        {
            const float swap = high;
            high = low;
            low = swap;
        }
        if (high == 0.0f)
        {
            return 0.0f;
        }
        const float ratio = low / high;
        return high * SqrtOneToTwo(1.0f + ratio * ratio);
    }

    __aicore__ inline void CommandBarrier(const AscendC::LocalTensor<int32_t> &barrierLocal)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
#elif defined(CHEEVJ_TRIDIAG_COMMON_SECTION) && CHEEVJ_TRIDIAG_COMMON_SECTION == 6
            {
                diagonal.SetValue(index + 1, diagonal.GetValue(index + 1) - correction);
                offDiagonal.SetValue(split, 0.0f);
                return;
            }
            sine = radius > 0.0f ? edgeTimesSine / radius : 0.0f;
            cosine = radius > 0.0f ? shift / radius : 0.0f;
            shift = diagonal.GetValue(index + 1) - correction;
            radius = (diagonal.GetValue(index) - shift) * sine + 2.0f * cosine * edgeTimesCosine;
            correction = sine * radius;
            diagonal.SetValue(index + 1, shift + correction);
            shift = cosine * radius - edgeTimesCosine;
#elif defined(CHEEVJ_TRIDIAG_COMMON_SECTION) && CHEEVJ_TRIDIAG_COMMON_SECTION == 7
                {
                    return first + 1;
                }
                RunQlIteration(first, split, barrierLocal);
            }
        }
        return 0;
    }

    __aicore__ inline void RunQlIteration(int first, int split,
                                          const AscendC::LocalTensor<int32_t> &barrierLocal)
    {
        auto diagonal = diagonalBuf.Get<float>();
        auto offDiagonal = offDiagonalBuf.Get<float>();
        const float dFirst = diagonal.GetValue(first);
        const float eFirst = offDiagonal.GetValue(first);
        float shift = (diagonal.GetValue(first + 1) - dFirst) / (2.0f * eFirst);
        float radius = SafeHypot(shift, 1.0f);
        float denominator = shift + CopySign(radius, shift);
#elif defined(CHEEVJ_TRIDIAG_COMMON_SECTION) && CHEEVJ_TRIDIAG_COMMON_SECTION == 8
        AscendC::Duplicate(metadata, static_cast<int32_t>(convergenceInfo), 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams infoCopy{1, sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(infoGlobal, metadata, infoCopy);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ControllerProcess(float scale, const AscendC::LocalTensor<int32_t> &barrierLocal)
    {
        int convergenceInfo = 0;
        if (scale != 0.0f)
        {
            convergenceInfo = ImplicitQl(barrierLocal);
            SortEigenpairs(barrierLocal);
        }
        StoreControllerResults(scale, convergenceInfo);
#elif defined(CHEEVJ_TRIDIAG_COMMON_SECTION) && CHEEVJ_TRIDIAG_COMMON_SECTION == 9
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void WorkerProcess(int rowBegin, const AscendC::LocalTensor<int32_t> &barrierLocal)
    {
        // The controller terminates every worker by publishing CHEEVJ_TRIDIAG2048_STOP.
        bool running = true;
        while (running)
        {
            CommandBarrier(barrierLocal);
            LoadCommand();
            auto commandInt = commandBuf.Get<int32_t>();
            auto commandFloat = commandBuf.Get<float>();
            const int operation = commandInt.GetValue(0);
            const int index = commandInt.GetValue(8);
#elif defined(CHEEVJ_TRIDIAG_COMMON_SECTION) && CHEEVJ_TRIDIAG_COMMON_SECTION == 10
            {
                const float value = diagonal.GetValue(candidate);
                if (value < minimum)
                {
                    minimum = value;
                    minimumIndex = candidate;
                }
            }
            if (minimumIndex != output)
            {
                const float previous = diagonal.GetValue(output);
                diagonal.SetValue(output, minimum);
                diagonal.SetValue(minimumIndex, previous);
                // For a swap command, index packs lhs in the low 16 bits and
                // rhs in the high 16 bits.  Both are in [0,2047].
                const int packed = output | (minimumIndex << 16);
#endif
