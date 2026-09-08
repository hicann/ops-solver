/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    __aicore__ inline int LocalOffset(int row, int col) const
    {
        return row + col * CHEEVJ_STAGE_BLOCK_JACOBI16_LOCAL_DIM;
    }

    __aicore__ inline int LeftRowsOffset(int localRow, int tileCol) const
    {
        return localRow * CHEEVJ_STAGE_BLOCK_JACOBI16_TILE + tileCol;
    }

    __aicore__ inline int ColumnOffset(int col, int rowOffset) const { return col * planar.layout.strideN + rowOffset; }

    __aicore__ inline int RowOffset(int row, int colOffset) const { return colOffset * planar.layout.strideN + row; }

    __aicore__ inline bool HasVectorPlanes() const { return planar.layout.basePlanes >= CHEEVJ_WS_PLANES_VECTOR; }

    __aicore__ inline int GlobalIndexForLocal(const CheevjStageBlockPair16 &pair, int localIndex) const
    {
        if (localIndex < pair.left.size)
        {
            return pair.left.begin + localIndex;
        }
        return pair.right.begin + (localIndex - pair.left.size);
    }

    __aicore__ inline int SplitUnitaryStorageOffset(int pairIndex) const { return pairIndex * UnitaryStorageElems(); }

    __aicore__ inline int UnitaryStorageDim() const { return 2 * schedule.BlockSize(); }

    __aicore__ inline int UnitaryStorageElems() const
    {
        const int storageDim = UnitaryStorageDim();
        return storageDim * storageDim;
    }

    __aicore__ inline GlobalTensor<float> StoredUnitaryReal(int pairIndex) const
    {
        return planar.scratch0[SplitUnitaryStorageOffset(pairIndex)];
    }

    __aicore__ inline GlobalTensor<float> StoredUnitaryImag(int pairIndex) const
    {
        return planar.scratch1[SplitUnitaryStorageOffset(pairIndex)];
    }

    __aicore__ inline GlobalTensor<float> StoredUnitaryImagNeg(int pairIndex) const
    {
        return planar.scratch2[PackedUnitaryStorageOffset(pairIndex, 2)];
    }

    __aicore__ inline bool HasUnitaryStorageCapacity() const
    {
        const int pairCount = schedule.PairCountPerStage();
        const int requiredElems = pairCount * UnitaryStorageElems();
        return requiredElems <= planar.layout.planeElems;
    }

    __aicore__ inline int PackedUnitaryStorageOffset(int pairIndex, int part) const
    {
        return (pairIndex * 3 + part) * UnitaryStorageElems();
    }

    __aicore__ inline int PanelStorageBegin() const { return schedule.PairCountPerStage() * UnitaryStorageElems(); }

    __aicore__ inline int PrimaryPanelSlotCapacity(int localDim) const
    {
        const int panelElems = PanelBufferElems(localDim);
        const int available = planar.layout.planeElems - PanelStorageBegin();
        return available > 0 ? available / (2 * panelElems) : 0;
    }

    __aicore__ inline int ExtraPanelSlotCapacity(int localDim) const
    {
        if (planar.layout.scratchPlanes < 5)
        {
            return 0;
        }
        const int panelElems = PanelBufferElems(localDim);
        return planar.layout.planeElems / (2 * panelElems);
    }

    __aicore__ inline int PanelSlotStorageOffset(int slotIndex, int localDim) const
    {
        const int primarySlots = PrimaryPanelSlotCapacity(localDim);
        const int localSlot = slotIndex < primarySlots ? slotIndex : slotIndex - primarySlots;
        const int base = slotIndex < primarySlots ? PanelStorageBegin() : 0;
        return base + localSlot * 2 * PanelBufferElems(localDim);
    }

    __aicore__ inline GlobalTensor<float> PanelSlotReal(int slotIndex, int localDim) const
    {
        const int offset = PanelSlotStorageOffset(slotIndex, localDim);
        if (slotIndex < PrimaryPanelSlotCapacity(localDim))
        {
            return planar.scratch0[offset];
        }
        return planar.scratch3[offset];
    }

    __aicore__ inline GlobalTensor<float> PanelSlotImag(int slotIndex, int localDim) const
    {
        const int offset = PanelSlotStorageOffset(slotIndex, localDim);
        if (slotIndex < PrimaryPanelSlotCapacity(localDim))
        {
            return planar.scratch1[offset];
        }
        return planar.scratch4[offset];
    }

    __aicore__ inline int RegularPanelServiceSlotCount(int localDim) const
    {
        int slotCount = PrimaryPanelSlotCapacity(localDim) + ExtraPanelSlotCapacity(localDim);
        const int imagNegSlots = Scratch2ImagNegSlotCapacity(localDim) + ExtraImagNegSlotCapacity(localDim);
        slotCount = MinInt(slotCount, imagNegSlots);
        slotCount = MinInt(slotCount, schedule.PairCountPerStage());
        const int aicCount = GetBlockNum() > 0 ? GetBlockNum() : 1;
        return MinInt(slotCount, aicCount);
    }

    __aicore__ inline bool HasSplitSpillPanelSlot(int localDim) const
    {
        // N mode has three scratch planes. After the regular packed panels,
        // scratch0/scratch1 each retain one panel-sized tail. Pair those tails
        // as the spill input and place its two outputs plus left-imagNeg in the
        // unused scratch2 tail. This balances 16 CUBE groups without allocating
        // extra workspace or introducing scalar copies.
        if (planar.layout.scratchPlanes != CHEEVJ_STAGE_BLOCK_JACOBI16_SCRATCH_PLANES)
        {
            return false;
        }
        const int regularSlots = RegularPanelServiceSlotCount(localDim);
        const int aicCount = GetBlockNum() > 0 ? GetBlockNum() : 1;
        if (regularSlots >= schedule.PairCountPerStage() || regularSlots >= aicCount)
        {
            return false;
        }
        const int panelElems = PanelBufferElems(localDim);
        const int primaryTailElems = planar.layout.planeElems - PrimaryPanelTailBegin(localDim);
        const int scratch2TailElems = planar.layout.planeElems - PackedUnitaryStorageEnd() - regularSlots * panelElems;
        return primaryTailElems >= panelElems && scratch2TailElems >= 3 * panelElems;
    }

    __aicore__ inline bool IsSplitSpillPanelSlot(int slotIndex, int localDim) const
    {
        return HasSplitSpillPanelSlot(localDim) && slotIndex == RegularPanelServiceSlotCount(localDim);
    }

    __aicore__ inline GlobalTensor<float> PanelSlotInputReal(int slotIndex, int localDim) const
    {
        if (IsSplitSpillPanelSlot(slotIndex, localDim))
        {
            return planar.scratch0[PrimaryPanelTailBegin(localDim)];
        }
        return PanelSlotReal(slotIndex, localDim);
    }

    __aicore__ inline GlobalTensor<float> PanelSlotInputImag(int slotIndex, int localDim) const
    {
        if (IsSplitSpillPanelSlot(slotIndex, localDim))
        {
            return planar.scratch1[PrimaryPanelTailBegin(localDim)];
        }
        return PanelSlotImag(slotIndex, localDim);
    }

    __aicore__ inline GlobalTensor<float> PanelSlotOutputReal(int slotIndex, int localDim) const
    {
        if (IsSplitSpillPanelSlot(slotIndex, localDim))
        {
            const int regularSlots = RegularPanelServiceSlotCount(localDim);
            return planar.scratch2[PackedUnitaryStorageEnd() + regularSlots * PanelBufferElems(localDim)];
        }
        return PanelSlotReal(slotIndex, localDim)[PanelBufferElems(localDim)];
    }

    __aicore__ inline GlobalTensor<float> PanelSlotOutputImag(int slotIndex, int localDim) const
    {
        if (IsSplitSpillPanelSlot(slotIndex, localDim))
        {
            const int regularSlots = RegularPanelServiceSlotCount(localDim);
            return planar.scratch2[PackedUnitaryStorageEnd() + (regularSlots + 1) * PanelBufferElems(localDim)];
        }
        return PanelSlotImag(slotIndex, localDim)[PanelBufferElems(localDim)];
    }

    __aicore__ inline GlobalTensor<float> RightPanelBufferReal(int slotIndex, int localDim, int bufferIndex) const
    {
        return bufferIndex == 0 ? PanelSlotInputReal(slotIndex, localDim) : PanelSlotOutputReal(slotIndex, localDim);
    }

    __aicore__ inline GlobalTensor<float> RightPanelBufferImag(int slotIndex, int localDim, int bufferIndex) const
    {
        return bufferIndex == 0 ? PanelSlotInputImag(slotIndex, localDim) : PanelSlotOutputImag(slotIndex, localDim);
    }

    __aicore__ inline int RightPanelPackReadyFlag(int bufferIndex) const
    {
        return bufferIndex == 0 ? CHEEVJ_STAGE_PANEL_PACK_READY_FLAG : CHEEVJ_STAGE_PANEL_PACK_READY_FLAG_PONG;
    }

    __aicore__ inline int RightPanelGemmDoneFlag(int bufferIndex) const
    {
        return bufferIndex == 0 ? CHEEVJ_STAGE_PANEL_GEMM_DONE_FLAG : CHEEVJ_STAGE_PANEL_GEMM_DONE_FLAG_PONG;
    }

    __aicore__ inline int PackedUnitaryStorageEnd() const
    {
        return schedule.PairCountPerStage() * 3 * UnitaryStorageElems();
    }

    __aicore__ inline int Scratch2ImagNegSlotCapacity(int localDim) const
    {
        const int available = planar.layout.planeElems - PackedUnitaryStorageEnd();
        return available > 0 ? available / PanelBufferElems(localDim) : 0;
    }

    __aicore__ inline int PrimaryPanelTailBegin(int localDim) const
    {
        return PanelStorageBegin() + PrimaryPanelSlotCapacity(localDim) * 2 * PanelBufferElems(localDim);
    }

    __aicore__ inline int ExtraImagNegSlotCapacity(int localDim) const
    {
        if (planar.layout.scratchPlanes < 5)
        {
            return 0;
        }
        const int panelElems = PanelBufferElems(localDim);
        const int tailElems = planar.layout.planeElems - PrimaryPanelTailBegin(localDim);
        const int slotsPerTail = tailElems > 0 ? tailElems / (panelElems > 0 ? panelElems : 1) : 0;
        return 2 * slotsPerTail;
    }

    __aicore__ inline GlobalTensor<float> LeftPanelImagNeg(int slotIndex, int localDim) const
    {
        const int panelElems = PanelBufferElems(localDim);
        if (IsSplitSpillPanelSlot(slotIndex, localDim))
        {
            const int regularSlots = RegularPanelServiceSlotCount(localDim);
            return planar.scratch2[PackedUnitaryStorageEnd() + (regularSlots + 2) * panelElems];
        }
        const int scratch2Slots = Scratch2ImagNegSlotCapacity(localDim);
        if (slotIndex < scratch2Slots)
        {
            return planar.scratch2[PackedUnitaryStorageEnd() + slotIndex * panelElems];
        }
        const int tailBegin = PrimaryPanelTailBegin(localDim);
        const int tailElems = planar.layout.planeElems - tailBegin;
        const int slotsPerTail = tailElems > 0 ? tailElems / (panelElems > 0 ? panelElems : 1) : 0;
        const int extraSlot = slotIndex - scratch2Slots;
        if (extraSlot < slotsPerTail)
        {
            return planar.scratch0[tailBegin + extraSlot * panelElems];
        }
        return planar.scratch1[tailBegin + (extraSlot - slotsPerTail) * panelElems];
    }

    __aicore__ inline int PanelBufferElems(int localDim) const { return planar.layout.n * localDim; }

    __aicore__ inline int PanelTaskSliceCount() const
    {
        return (planar.layout.n + CHEEVJ_STAGE_BLOCK_JACOBI16_TILE - 1) / CHEEVJ_STAGE_BLOCK_JACOBI16_TILE;
    }

    __aicore__ inline int ComputePanelServiceSlotCount(int localDim) const
    {
        if (localDim <= 0)
        {
            return 0;
        }
        const int regularSlots = RegularPanelServiceSlotCount(localDim);
        return regularSlots + (HasSplitSpillPanelSlot(localDim) ? 1 : 0);
    }
