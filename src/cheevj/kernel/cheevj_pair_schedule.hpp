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
 * \file cheevj_pair_schedule.hpp
 * \brief Shared round-robin block-pair schedule.
 */

#ifndef CHEEVJ_C64_PAIR_SCHEDULE_HPP
#define CHEEVJ_C64_PAIR_SCHEDULE_HPP

#include "cheevj_workspace.hpp"

namespace Cheevj
{

struct CheevjRoundRobinRange
{
    int blockIndex;
    int begin;
    int size;
};

struct CheevjRoundRobinPair
{
    CheevjRoundRobinRange left;
    CheevjRoundRobinRange right;
    bool valid;
};

class CheevjRoundRobinPairSchedule
{
   public:
    __aicore__ inline CheevjRoundRobinPairSchedule() : n(0), blockSize(1), blockCount(0), ringBlockCount(0) {}

    __aicore__ inline void Reset(int matrixN, int matrixBlockSize)
    {
        n = matrixN;
        blockSize = matrixBlockSize > 0 ? matrixBlockSize : 1;
        blockCount = AlignUp(n, blockSize) / blockSize;
        ringBlockCount = (blockCount % 2 == 0) ? blockCount : blockCount + 1;
    }

    __aicore__ inline int BlockSize() const { return blockSize; }
    __aicore__ inline int BlockCount() const { return blockCount; }
    __aicore__ inline int StageCount() const { return ringBlockCount > 1 ? ringBlockCount - 1 : 0; }
    __aicore__ inline int PairCountPerStage() const { return ringBlockCount / 2; }

    __aicore__ inline CheevjRoundRobinPair PairForStage(int stage, int pairIndex) const
    {
        CheevjRoundRobinPair pair;
        pair.valid = false;
        pair.left = EmptyRange();
        pair.right = EmptyRange();
        if (pairIndex < 0 || pairIndex >= PairCountPerStage())
        {
            return pair;
        }
        int leftSlot = pairIndex;
        int rightSlot = ringBlockCount - 1 - pairIndex;
        leftSlot = RotateSlot(leftSlot, stage);
        rightSlot = RotateSlot(rightSlot, stage);
        if (leftSlot >= blockCount || rightSlot >= blockCount || leftSlot == rightSlot)
        {
            return pair;
        }
        if (leftSlot > rightSlot)
        {
            const int temporary = leftSlot;
            leftSlot = rightSlot;
            rightSlot = temporary;
        }
        pair.left = MakeRange(leftSlot);
        pair.right = MakeRange(rightSlot);
        pair.valid = pair.left.size > 0 && pair.right.size > 0;
        return pair;
    }

   private:
    int n;
    int blockSize;
    int blockCount;
    int ringBlockCount;

    __aicore__ inline int RotateSlot(int slot, int stage) const
    {
        if (slot == 0 || ringBlockCount <= 1)
        {
            return 0;
        }
        const int divisor = ringBlockCount - 1;
        const int rotated = (slot - 1 + stage) % divisor;
        return 1 + (rotated >= 0 ? rotated : rotated + divisor);
    }

    __aicore__ inline CheevjRoundRobinRange EmptyRange() const
    {
        return {-1, 0, 0};
    }

    __aicore__ inline CheevjRoundRobinRange MakeRange(int blockIndex) const
    {
        CheevjRoundRobinRange range;
        range.blockIndex = blockIndex;
        range.begin = blockIndex * blockSize;
        range.size = MinInt(blockSize, n - range.begin);
        return range;
    }
};

template <typename Solver>
__aicore__ inline bool CheevjProcessOuterSweeps(Solver &solver, int outerSweeps, bool supported)
{
    if (!supported)
    {
        return false;
    }
    bool lastSweepLocalConverged = false;
    for (int sweep = 0; sweep < outerSweeps; ++sweep)
    {
        lastSweepLocalConverged = solver.ProcessOneSweep();
    }
    return lastSweepLocalConverged;
}

}  // namespace Cheevj

#endif  // CHEEVJ_C64_PAIR_SCHEDULE_HPP
