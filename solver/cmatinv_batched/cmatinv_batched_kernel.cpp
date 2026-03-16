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
 * \file cmatinv_batched_kernel.cpp
 * \brief
 */

#ifndef CMATINV_BATCHED_AIV_H
#define CMATINV_BATCHED_AIV_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

namespace CmatinvBatched {

constexpr uint32_t BYTENUM_BLOCK = 32;
constexpr uint32_t BYTENUM_REPEAT = 256;

constexpr uint32_t COMPLEX_ELENUM = 2;
constexpr uint32_t MAX_CORE_CNT = 40;

constexpr uint32_t ELENUM_BLOCK_FP32 = 8;
constexpr uint32_t ELENUM_REPEAT_FP32 = 64;
constexpr uint32_t ELENUM_BLOCK_FP16 = 16;
constexpr uint32_t ELENUM_REPEAT_FP16 = 128;

constexpr uint32_t BASE_SIZE_ONE_ITERATION = 64 * 1024; // 64 kb
constexpr uint32_t BASE_COMPLEX64_NUM = BASE_SIZE_ONE_ITERATION / sizeof(float) / COMPLEX_ELENUM; // 8192
constexpr uint32_t MAX_BATCH_NUM = 128;
constexpr uint32_t MAX_STRIDE_NUM = 255;
constexpr uint32_t MAX_REPEAT_NUM = 255;
constexpr uint32_t MAX_UBUF_SIZE = 192 * 1024; // 192kb

template <typename T>
class CmatinvBatchedAIV {
public:
    __aicore__ inline CmatinvBatchedAIV(){};
    __aicore__ inline void Init(GM_ADDR dA, GM_ADDR dUniReal, GM_ADDR dUniImag,
        GM_ADDR dOffset, GM_ADDR dAinv, GM_ADDR workSpace, GM_ADDR tilingGm);

    __aicore__ inline void InitUbuf();
    __aicore__ inline void InitTempUbuf(LocalTensor<float> ubufLocal, uint32_t allocOffset);
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);

    __aicore__ inline void Process();
    __aicore__ inline void SingleProcess();

    __aicore__ inline void CopyCmatBatchGmToUb(LocalTensor<T> dst,
        GlobalTensor<T> src, uint32_t matColSize, uint32_t batchSize);
    __aicore__ inline void CopyUniMatBatchGmToUb(LocalTensor<float> dst, GlobalTensor<float> src,
        uint32_t matColSize, uint32_t batchSize, uint32_t isComplex);
    __aicore__ inline void CopyVecGmToUbUint32(LocalTensor<uint32_t> dst,
        GlobalTensor<uint32_t> src, uint32_t len);
    __aicore__ inline void CopyVecGmToUb(LocalTensor<float> dst,
        GlobalTensor<float> src, uint32_t len);
    __aicore__ inline void CopyVecUbToGm(GlobalTensor<T> dst,
        LocalTensor<T> src, uint32_t len);

    __aicore__ inline void ComputeComplexScalarOffset();
    __aicore__ inline void PrepareDataForHalf(uint32_t matColSize, uint32_t batchSize);
    __aicore__ inline void SeparateComplexMat();
    __aicore__ inline void CollectScalarSingle(LocalTensor<float> collectDst, uint32_t sOffset);
    __aicore__ inline void ComplexScalarDiv();

    __aicore__ inline void RealScalarVecBatchMul(LocalTensor<float> dst,
        LocalTensor<float> src0, LocalTensor<float> src1, uint32_t oneMatOffset);
    __aicore__ inline void RealScalarVecBatchSub(LocalTensor<float> dst,
        LocalTensor<float> src, uint32_t oneMatOffset);
    __aicore__ inline void RealVecScalarBatchDiv(LocalTensor<float> dst,
        LocalTensor<float> src0, LocalTensor<float> src1, uint32_t oneMatOffset);

    __aicore__ inline void ComplexScalarVecBatchMul(LocalTensor<float> matReal,
        LocalTensor<float> matImag, uint32_t oneMatOffset);
    __aicore__ inline void ComputeAndUpdateAj(LocalTensor<float> AjReal,
        LocalTensor<float> AjImag, uint32_t oneMatOffset);
    __aicore__ inline void ComplexVecScalarBatchDiv(LocalTensor<float> vecReal,
        LocalTensor<float> vecImag, uint32_t oneMatOffset);
    __aicore__ inline void ComplexCombinationAndCopyout(uint32_t outOffset, uint32_t batchSize);

private:
    static constexpr bool IS_FLOAT = IsSameType<T, float>::value;  // half or float

    TPipe pipe;

    GlobalTensor<T> inAGM;
    GlobalTensor<float> uniRealGM;
    GlobalTensor<float> uniImagGM;
    GlobalTensor<uint32_t> offsetGM;
    GlobalTensor<T> outGM;
    GlobalTensor<float> workGM;

    TBuf<QuePosition::VECCALC> uBuf;

    // main buf
    LocalTensor<float> invSepLocal;
    LocalTensor<float> invRealLocal;
    LocalTensor<float> invImagLocal;
    LocalTensor<float> matSepLocal;
    LocalTensor<float> matRealLocal;
    LocalTensor<float> matImagLocal;
    LocalTensor<float> matInLocal;

    // temp buf
    LocalTensor<float> preLocal;
    LocalTensor<float> preRealLocal;
    LocalTensor<float> preImagLocal;
    LocalTensor<float> afterLocal;
    LocalTensor<float> afterRealLocal;
    LocalTensor<float> afterImagLocal;
    LocalTensor<float> resLocal;
    LocalTensor<float> resRealLocal;
    LocalTensor<float> resImagLocal;

    LocalTensor<float> scalarTempLocal;
    LocalTensor<float> scalarBrcbLocal;
    LocalTensor<float> scalarBrcbRealLocal;
    LocalTensor<float> scalarBrcbImagLocal;
    LocalTensor<float> scalarBrcbAugLocal;

    LocalTensor<float> updateTempRealLocal;
    LocalTensor<float> updateTempImagLocal;
    LocalTensor<float> updateTempAugRealLocal;
    LocalTensor<float> updateTempAugImagLocal;

    LocalTensor<uint32_t> collectScalarOffsetLocal;
    LocalTensor<uint32_t> complexCombineOffsetLocal;

    uint32_t n = 0;
    uint32_t offset = 0;
    uint32_t calNum = 0;

    uint32_t alignedComplexMatSize = 0;
    uint32_t batchNumPerRepeat = 0;
    uint32_t oneMatUbOffsetFp32 = 0;

    int32_t coreIdx;
};

template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::Init(
    GM_ADDR dA, GM_ADDR dUniReal, GM_ADDR dUniImag,
    GM_ADDR dOffset, GM_ADDR dAinv, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    SetMaskNorm();
    SetAtomicNone();

    this->coreIdx = GetBlockIdx();

    ParseTilingData(tilingGm);

    this->inAGM.SetGlobalBuffer((__gm__ T *)dA);
    this->uniRealGM.SetGlobalBuffer((__gm__ float *)dUniReal);
    this->uniImagGM.SetGlobalBuffer((__gm__ float *)dUniImag);
    this->offsetGM.SetGlobalBuffer((__gm__ uint32_t *)dOffset);
    this->outGM.SetGlobalBuffer((__gm__ T *)dAinv);

    this->workGM.SetGlobalBuffer((__gm__ float *)workSpace);

    InitUbuf();
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::InitUbuf()
{
    /*
        can proces 64kb one time,
        can be 8 * 32 * 32 ~ 128 * 8 * 8 ~ 512 * 2 * 8 (complex64 num)
        the batch size ranges [8, 512]
        the size of matrix ranges [2, 32]
    */
    this->alignedComplexMatSize = (this->n + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32 * ELENUM_BLOCK_FP32;

    this->batchNumPerRepeat = BASE_COMPLEX64_NUM / (this->n * this->alignedComplexMatSize); // [8, 512]
    if (this->batchNumPerRepeat > MAX_BATCH_NUM) {
        this->batchNumPerRepeat = MAX_BATCH_NUM;
    }
    this->batchNumPerRepeat = this->batchNumPerRepeat / ELENUM_BLOCK_FP32 * ELENUM_BLOCK_FP32;

    // offset of one matrix in ub, mat_size x matAlignedSize, 8 * 8 = 64
    this->oneMatUbOffsetFp32 = this->n * this->alignedComplexMatSize;

    this->pipe.InitBuffer(this->uBuf, MAX_UBUF_SIZE);
    LocalTensor<float> ubufLocal = uBuf.Get<float>();

    // init main buf
    uint32_t allocOffset = 0;
    this->invSepLocal = ubufLocal[allocOffset]; // 64kb
    this->invRealLocal = ubufLocal[allocOffset]; // 32kb
    allocOffset += BASE_COMPLEX64_NUM;
    this->invImagLocal = ubufLocal[allocOffset]; // 32kb
    allocOffset += BASE_COMPLEX64_NUM;
    this->matSepLocal = ubufLocal[allocOffset]; // 64kb
    this->matRealLocal = ubufLocal[allocOffset]; // 32kb
    allocOffset += BASE_COMPLEX64_NUM;
    this->matImagLocal = ubufLocal[allocOffset]; // 32kb
    allocOffset += BASE_COMPLEX64_NUM;
    this->matInLocal = ubufLocal[allocOffset]; // 64kb

    // reuse mat sep addr for saving complex combination offset
    this->complexCombineOffsetLocal = this->matSepLocal.template ReinterpretCast<uint32_t>();

    // init temp buf, reuse mat in buf
    InitTempUbuf(ubufLocal, allocOffset);
}

template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::InitTempUbuf(LocalTensor<float> ubufLocal, uint32_t allocOffset)
{
    uint32_t alignedBatchSize = (this->batchNumPerRepeat +
        ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32 * ELENUM_REPEAT_FP32; // 256 bytes

    this->collectScalarOffsetLocal = ubufLocal[allocOffset].template ReinterpretCast<uint32_t>();
    allocOffset += alignedBatchSize * COMPLEX_ELENUM; // 512 bytes
    this->preLocal = ubufLocal[allocOffset]; // 1kb ~ 4kb, 1kb
    this->preRealLocal = ubufLocal[allocOffset]; // num of alignedBatchSize scalar
    allocOffset += alignedBatchSize; // 256 bytes
    this->preImagLocal = ubufLocal[allocOffset]; // num of alignedBatchSize scalar
    allocOffset += alignedBatchSize; // 256 bytes

    this->afterLocal = ubufLocal[allocOffset]; // 1kb ~ 4kb, 1kb
    this->afterRealLocal = ubufLocal[allocOffset];
    allocOffset += alignedBatchSize; // 256 bytes
    this->afterImagLocal = ubufLocal[allocOffset];
    allocOffset += alignedBatchSize; // 256 bytes

    this->resLocal = ubufLocal[allocOffset]; // 1kb ~ 4kb, 1kb
    this->resRealLocal = ubufLocal[allocOffset];
    allocOffset += alignedBatchSize; // 256 bytes
    this->resImagLocal = ubufLocal[allocOffset];
    allocOffset += alignedBatchSize; // 256 bytes
    this->scalarTempLocal = ubufLocal[allocOffset]; // 512bytes ~ 2kb, 512 bytes
    allocOffset += alignedBatchSize; // 256 bytes

    uint32_t brcbEleNum = alignedBatchSize * ELENUM_BLOCK_FP32 * COMPLEX_ELENUM; // 64 * 32 * 2 = 4kb
    this->scalarBrcbLocal = ubufLocal[allocOffset]; // 4 ~ 32 kb, 4kb
    this->scalarBrcbRealLocal = ubufLocal[allocOffset]; // 4 ~ 32 kb, 4kb
    allocOffset += brcbEleNum; // 4kb
    this->scalarBrcbImagLocal = ubufLocal[allocOffset]; // 4 ~ 32 kb, 4kb
    allocOffset += brcbEleNum; // 4kb
    this->scalarBrcbAugLocal = ubufLocal[allocOffset]; // 4 ~ 32 kb, 4kb
    allocOffset += brcbEleNum; // 4kb

    uint32_t updateTempEleNum = this->alignedComplexMatSize * this->batchNumPerRepeat; // 8kb
    this->updateTempRealLocal = ubufLocal[allocOffset]; // 4kb
    allocOffset += updateTempEleNum; // 8kb
    this->updateTempImagLocal = ubufLocal[allocOffset]; // 4kb
    allocOffset += updateTempEleNum; // 8kb
    this->updateTempAugRealLocal = ubufLocal[allocOffset]; // 4kb
    this->updateTempAugImagLocal = ubufLocal[allocOffset]; // 4kb
}

template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::ParseTilingData(GM_ADDR tilingGm)
{
    // skip dtype
    uint32_t locId = 1;
    auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);
    this->n = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t))); // num of complex elements
    locId++;
    this->offset = (*(
        __gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t) + sizeof(uint32_t) * this->coreIdx));
    locId += MAX_CORE_CNT;
    this->calNum = (*(
        __gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t) + sizeof(uint32_t) * this->coreIdx));
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::Process()
{
    auto eventId = EVENT_ID0;
    uint32_t overallRepeatTime = this->calNum / this->batchNumPerRepeat;
    uint32_t remainBatchNum = this->calNum % this->batchNumPerRepeat;
    if (remainBatchNum > 0) {
        overallRepeatTime++;
    }

    uint32_t batchOffset = this->batchNumPerRepeat * this->n * this->n * COMPLEX_ELENUM;
    uint32_t currentBatchSize = this->batchNumPerRepeat;
    uint32_t currOffset = this->offset;
    for (uint32_t i = 0; i < overallRepeatTime; i++) {
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);

        if (remainBatchNum > 0 && i == overallRepeatTime - 1) {
            currentBatchSize = remainBatchNum;
        }

        // copy A batch
        if constexpr (IS_FLOAT) {
            LocalTensor<T> inLocal = this->matInLocal.template ReinterpretCast<T>();
            CopyCmatBatchGmToUb(inLocal, this->inAGM[currOffset], this->n * COMPLEX_ELENUM, currentBatchSize);
        } else {
            LocalTensor<T> inLocal = this->matSepLocal.template ReinterpretCast<T>();
            CopyCmatBatchGmToUb(inLocal, this->inAGM[currOffset], this->n * COMPLEX_ELENUM, currentBatchSize);

            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);

            PrepareDataForHalf(this->n * COMPLEX_ELENUM, currentBatchSize);
        }

        // uni real part, uni imag part
        CopyUniMatBatchGmToUb(this->invRealLocal, this->uniRealGM, this->alignedComplexMatSize, currentBatchSize, 0);
        CopyUniMatBatchGmToUb(this->invImagLocal, this->uniImagGM, this->alignedComplexMatSize, currentBatchSize, 0);

        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        SingleProcess();

        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        // copy out
        ComplexCombinationAndCopyout(currOffset, currentBatchSize);
        currOffset += batchOffset;
    }
    PipeBarrier<PIPE_ALL>();
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::SingleProcess()
{
    // complex separation
    SeparateComplexMat();
    PipeBarrier<PIPE_ALL>();

    ComputeComplexScalarOffset();
    PipeBarrier<PIPE_ALL>();
    for (uint16_t elementIdx = 0; elementIdx < this->n; elementIdx++) {
        // iterate diagnal elements
        // collect diagnal element, e.g., aii
        uint32_t preOffset = elementIdx * this->alignedComplexMatSize + elementIdx;
        CollectScalarSingle(this->preLocal, preOffset);

        uint32_t eleOffset = elementIdx * this->alignedComplexMatSize;
        LocalTensor<float> tmpAiReal = this->matRealLocal[eleOffset];
        LocalTensor<float> tmpAiRmag = this->matImagLocal[eleOffset];
        LocalTensor<float> tmpAiInvReal = this->invRealLocal[eleOffset];
        LocalTensor<float> tmpAiInvImag = this->invImagLocal[eleOffset];

        for (uint16_t rowIdx = 0; rowIdx < this->n; rowIdx++) {
            // iterate elements in i-th column except aii
            if (rowIdx == elementIdx) {
                continue;
            }

            uint32_t rowOffset = rowIdx * this->alignedComplexMatSize;
            LocalTensor<float> tmpAjReal = this->matRealLocal[rowOffset];
            LocalTensor<float> tmpAjImag = this->matImagLocal[rowOffset];
            LocalTensor<float> tmpAjInvReal = this->invRealLocal[rowOffset];
            LocalTensor<float> tmpAjInvImag = this->invImagLocal[rowOffset];

            // suppose the collected element is aji
            uint32_t afterOffset = rowIdx * this->alignedComplexMatSize + elementIdx;
            CollectScalarSingle(this->afterLocal, afterOffset);
            PipeBarrier<PIPE_V>();

            ComplexScalarDiv();
            PipeBarrier<PIPE_V>();

            // compute and update A[j,:] = A[j,:] - (aji / aii) * A[i,:]
            // (aji / aii) * A[i,:]
            ComplexScalarVecBatchMul(tmpAiReal, tmpAiRmag, this->oneMatUbOffsetFp32);
            PipeBarrier<PIPE_V>();

            ComputeAndUpdateAj(tmpAjReal, tmpAjImag, this->oneMatUbOffsetFp32);
            PipeBarrier<PIPE_V>();

            // update inv
            ComplexScalarVecBatchMul(tmpAiInvReal, tmpAiInvImag, this->oneMatUbOffsetFp32);
            PipeBarrier<PIPE_V>();

            ComputeAndUpdateAj(tmpAjInvReal, tmpAjInvImag, this->oneMatUbOffsetFp32);
            PipeBarrier<PIPE_V>();
        }

        // update A[i,:] = A[i,:] / aii
        ComplexVecScalarBatchDiv(tmpAiReal, tmpAiRmag, this->oneMatUbOffsetFp32);
        PipeBarrier<PIPE_V>();

        ComplexVecScalarBatchDiv(tmpAiInvReal, tmpAiInvImag, this->oneMatUbOffsetFp32);
        PipeBarrier<PIPE_V>();
    }
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::CopyCmatBatchGmToUb(LocalTensor<T> dst,
    GlobalTensor<T> src, uint32_t matColSize, uint32_t batchSize)
{
    uint32_t matRowSize = this->n;
    uint16_t blockCnt = matRowSize * batchSize;
    uint32_t blockLen = matColSize * sizeof(T);
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    if constexpr (IS_FLOAT) {
        uint32_t paddingBlockNum = (blockLen + BYTENUM_BLOCK - 1) / BYTENUM_BLOCK;
        if (paddingBlockNum % COMPLEX_ELENUM != 0) {
            dstStride = 1;
        }
    }

    DataCopyExtParams copyParams{blockCnt, blockLen, srcStride, dstStride, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, copyParams, padParams);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::CopyUniMatBatchGmToUb(LocalTensor<float> dst, GlobalTensor<float> src,
    uint32_t matColSize, uint32_t batchSize, uint32_t isComplex)
{
    /*
        GM matrix shape:
            | a11, a12, ..., a1n|
            | a21, a12, ..., a1n|
            | ..., ..., ..., ...|
            | an1, ..., ..., ann|
        UB matrix shape:
            | a11, a12, ..., a1n, 0, 0, ..., 0| // padding
            | a21, a12, ..., a1n, 0, 0, ..., 0|
            | ..., ..., ..., ..., 0, 0, ..., 0|
            | an1, ..., ..., ann, 0, 0, ..., 0|
    */
    // copy matrix row by row

    uint32_t matRowSize = this->n;
    uint16_t blockCnt = matRowSize * batchSize;
    uint32_t blockLen = matColSize * sizeof(float);
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    uint32_t paddingBlockNum = (matColSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32;
    if (isComplex && (paddingBlockNum % COMPLEX_ELENUM != 0)) {
        dstStride = 1;
    }

    DataCopyExtParams copyParams{blockCnt, blockLen, srcStride, dstStride, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, copyParams, padParams);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::CopyVecGmToUbUint32(LocalTensor<uint32_t> dst,
    GlobalTensor<uint32_t> src, uint32_t len)
{
    uint16_t blockCnt = 1;
    uint32_t blockLen = len * sizeof(uint32_t);

    DataCopyExtParams copyParams{blockCnt, blockLen, 0, 0, 0};
    DataCopyPadExtParams<uint32_t> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, copyParams, padParams);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::CopyVecGmToUb(LocalTensor<float> dst,
    GlobalTensor<float> src, uint32_t len)
{
    uint16_t blockCnt = 1;
    uint32_t blockLen = len * sizeof(float);

    DataCopyExtParams copyParams{blockCnt, blockLen, 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, copyParams, padParams);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::CopyVecUbToGm(GlobalTensor<T> dst,
    LocalTensor<T> src, uint32_t len) {
    uint16_t blockCnt = 1;
    uint32_t blockLen = len * sizeof(T);

    DataCopyExtParams copyParams{blockCnt, blockLen, 0, 0, 0};
    DataCopyPad(dst, src, copyParams);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::ComputeComplexScalarOffset()
{
    LocalTensor<uint32_t> offsetLocal = this->collectScalarOffsetLocal;
    uint32_t oneMatOffset = this->oneMatUbOffsetFp32; // offset of one matrix in ub, matSize x matAlignedSize
    uint32_t batchSize = this->batchNumPerRepeat;
    uint32_t imagOffset = BASE_SIZE_ONE_ITERATION / COMPLEX_ELENUM; // 32kb

    uint32_t alignedBatchSize = (batchSize + ELENUM_REPEAT_FP32 - 1)
        / ELENUM_REPEAT_FP32 * ELENUM_REPEAT_FP32; // 128

    // scalar real part
    for (uint32_t j = 0; j < alignedBatchSize; j++) {
        uint32_t offsetVal = 0;
        if (j < batchSize) {
            offsetVal = oneMatOffset * j * sizeof(float); // real part
        } else {
            offsetVal = 0; // useless num, set to 0
        }
        offsetLocal.SetValue(j, offsetVal);
    }
    // scalar imag part
    for (uint32_t i = 0; i < alignedBatchSize; i++) {
        uint32_t offsetVal = 0;
        if (i < batchSize) {
            offsetVal = sizeof(float) * (i * oneMatOffset) + imagOffset; // imag part
        } else {
            offsetVal = 0; // useless num, set to 0
        }
        offsetLocal.SetValue(alignedBatchSize + i, offsetVal);
    }
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::PrepareDataForHalf(uint32_t matColSize, uint32_t batchSize)
{
    LocalTensor<half> src = this->matSepLocal.template ReinterpretCast<half>();
    LocalTensor<float> dst = this->matInLocal;

    uint32_t matRowSize = this->n;
    uint32_t blockCnt = matRowSize * batchSize;
    uint32_t blockLen = matColSize * sizeof(half);
    uint32_t paddingBlockNum = (blockLen + BYTENUM_BLOCK - 1) / BYTENUM_BLOCK;
    uint32_t vecCastRepeatNum = (blockCnt * paddingBlockNum * BYTENUM_BLOCK - 1) / BYTENUM_REPEAT + 1;

    Cast(dst, src, RoundMode::CAST_NONE, vecCastRepeatNum * ELENUM_REPEAT_FP16);
    PipeBarrier<PIPE_V>();
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::SeparateComplexMat()
{
    LocalTensor<float> matInner = this->matInLocal;
    LocalTensor<float> realInner = this->matRealLocal;
    LocalTensor<float> imagInner = this->matImagLocal;
    uint32_t matSize = this->n;
    uint32_t batchSize = this->batchNumPerRepeat;

    uint32_t alignedFpMatSize = (matSize * COMPLEX_ELENUM + ELENUM_BLOCK_FP32 - 1)
        / ELENUM_BLOCK_FP32 * ELENUM_BLOCK_FP32;
    uint32_t dstGap = 0;
    uint32_t paddingBlockNum = (matSize * COMPLEX_ELENUM + ELENUM_BLOCK_FP32 - 1)
        / ELENUM_BLOCK_FP32 * ELENUM_BLOCK_FP32;
    if ((paddingBlockNum / ELENUM_BLOCK_FP32) % COMPLEX_ELENUM != 0) {
        dstGap = 1;
    }

    uint32_t matAlignedSize = alignedFpMatSize + ELENUM_BLOCK_FP32 * dstGap;
    uint16_t repeatTime = (matSize * batchSize * matAlignedSize +
        ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32;

    uint64_t rsvdCnt = 0;
    GatherMask(realInner, matInner, 1, false, 0, {1, repeatTime, 8, 8}, rsvdCnt);
    GatherMask(imagInner, matInner, 2, false, 0, {1, repeatTime, 8, 8}, rsvdCnt);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::CollectScalarSingle(LocalTensor<float> collectDst,
    uint32_t sOffset)
{
    /*
        collect scalar from each matrix
    */
    LocalTensor<float> matInner = this->matSepLocal;
    LocalTensor<uint32_t> offsetInner = this->collectScalarOffsetLocal;
    uint32_t batchSize = this->batchNumPerRepeat;

    uint8_t repeatTime = (batchSize + ELENUM_REPEAT_FP32 - 1) /
        ELENUM_REPEAT_FP32 * COMPLEX_ELENUM;
    int32_t calCount = repeatTime * ELENUM_REPEAT_FP32;

    Gather(collectDst, matInner, offsetInner, sOffset * sizeof(float), calCount);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::ComplexScalarDiv()
{
    /*
        given (a + bi) / (c + di)
        result real part: (ac + bd) / (c^2 + d^2)
        result imag part: (bc - ad) / (c^2 + d^2)
    */
    uint32_t len = this->batchNumPerRepeat;
    uint32_t repeatTime = (len + ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32;

    // compute c^2 + d^2
    Mul(this->scalarTempLocal, this->preRealLocal, this->preRealLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8});
    Mul(this->resRealLocal, this->preImagLocal, this->preImagLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    Add(this->scalarTempLocal, this->scalarTempLocal, this->resRealLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8}); // c^2 + d^2
    PipeBarrier<PIPE_V>();

    // compute ac, ad, bd
    Mul(this->resRealLocal, this->afterRealLocal, this->preRealLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8}); // ac
    PipeBarrier<PIPE_V>();
    Mul(this->afterRealLocal, this->afterRealLocal, this->preImagLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8}); // ad
    PipeBarrier<PIPE_V>();
    Mul(this->resImagLocal, this->afterImagLocal, this->preImagLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8}); // bd
    PipeBarrier<PIPE_V>();

    // compute ac + bd
    Add(this->resRealLocal, this->resRealLocal, this->resImagLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();

    // compute (ac + bd) / (c^2 + d^2)
    Div(this->resRealLocal, this->resRealLocal, this->scalarTempLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();

    // compute bc
    Mul(this->resImagLocal, this->afterImagLocal, this->preRealLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8}); // bc
    PipeBarrier<PIPE_V>();
    // compute bc - ad
    Sub(this->resImagLocal, this->resImagLocal, this->afterRealLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    // compute (bc - ad) / (c^2 + d^2)
    Div(this->resImagLocal, this->resImagLocal, this->scalarTempLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8});
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::RealScalarVecBatchMul(LocalTensor<float> dst,
    LocalTensor<float> src0, LocalTensor<float> src1, uint32_t oneMatOffset)
{
    uint32_t matAlignedSize = this->alignedComplexMatSize;
    uint32_t batchSize = this->batchNumPerRepeat; // 64 aligned, 32
    uint8_t oneRowBlockStride = matAlignedSize / ELENUM_BLOCK_FP32;
    uint8_t oneMatBlockStride = oneMatOffset / ELENUM_BLOCK_FP32;
    uintptr_t oneMatBlockStrideFlag = oneMatOffset / ELENUM_BLOCK_FP32;
    uint32_t singleInstrRepeatTime = (matAlignedSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32;

    if (oneMatBlockStrideFlag * ELENUM_BLOCK_FP32 > MAX_STRIDE_NUM) {
        for (uint32_t j = 0; j < batchSize / ELENUM_BLOCK_FP32; j++) {
            Mul(dst[ELENUM_BLOCK_FP32 * matAlignedSize * j], src0[ELENUM_BLOCK_FP32 * oneMatOffset * j],
                src1[ELENUM_REPEAT_FP32 * j], ELENUM_REPEAT_FP32, (uint8_t)(singleInstrRepeatTime),
                {oneRowBlockStride, oneMatBlockStride, 1, 1, 1, 0});
        }
    } else {
        for (uint32_t i = 0; i < singleInstrRepeatTime; i++) {
            Mul(dst[ELENUM_BLOCK_FP32 * i], src0[ELENUM_BLOCK_FP32 * i],
                src1, ELENUM_REPEAT_FP32, (uint8_t)(batchSize / ELENUM_BLOCK_FP32),
                {oneRowBlockStride, oneMatBlockStride, 1,
                static_cast<uint8_t>(ELENUM_BLOCK_FP32 * oneRowBlockStride),
                static_cast<uint8_t>(ELENUM_BLOCK_FP32 * oneMatBlockStride), 8}); // vec.R x scalar.R
        }
    }
    PipeBarrier<PIPE_V>();
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::ComplexScalarVecBatchMul(LocalTensor<float> matReal,
    LocalTensor<float> matImag, uint32_t oneMatOffset)
{
    LocalTensor<float> scalarRealInner = this->resRealLocal;
    LocalTensor<float> scalarImagInner = this->resImagLocal;
    LocalTensor<float> scalarRealBrcbInner = this->scalarBrcbRealLocal;
    LocalTensor<float> scalarImagBrcbInner = this->scalarBrcbImagLocal;
    LocalTensor<float> resRealInner = this->updateTempRealLocal;
    LocalTensor<float> resImagInner = this->updateTempImagLocal;
    LocalTensor<float> resTempRealInner = this->updateTempAugRealLocal;
    LocalTensor<float> resTempImagInner = this->updateTempAugImagLocal;

    uint32_t matAlignedSize = this->alignedComplexMatSize;
    uint32_t batchSize = this->batchNumPerRepeat; // 64 aligned, 32
    /*
        given complex scalar and vec, compute scalar x vec
    */
    // ub_scalar_brcb stores: | R1, R1, ..., R1 | R2, .., R2 | ... | I1, ..., I1 | ... In |
    // real part: 64 * 8 nums, imag part 64 * 8 nums

    uint8_t brcbRepeatTime = (batchSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32; // 4
    Brcb(scalarRealBrcbInner, scalarRealInner, brcbRepeatTime, {1, 8});
    Brcb(scalarImagBrcbInner, scalarImagInner, brcbRepeatTime, {1, 8});
    PipeBarrier<PIPE_V>();

    /*
        need level-0 api to compute
            real part: vec.R x scalar.R - vec.I x scalar.I
            imag part: vec.I x scalar.R + vec.R x scalar.I
    */
    uint32_t resComputeRepeat = (matAlignedSize * batchSize + ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32; // 2

    // vec.R x scalar.R
    RealScalarVecBatchMul(resRealInner, matReal, scalarRealBrcbInner, oneMatOffset);
    // vec.I x scalar.I
    RealScalarVecBatchMul(resTempRealInner, matImag, scalarImagBrcbInner, oneMatOffset);

    // vec.R x scalar.R - vec.I x scalar.I
    Sub(resRealInner, resRealInner, resTempRealInner, ELENUM_REPEAT_FP32, resComputeRepeat, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();

    // vec.I x scalar.R
    RealScalarVecBatchMul(resImagInner, matImag, scalarRealBrcbInner, oneMatOffset);
    // vec.I x scalar.R
    RealScalarVecBatchMul(resTempRealInner, matReal, scalarImagBrcbInner, oneMatOffset);

    // vec.I x scalar.R + vec.R x scalar.I
    Add(resImagInner, resImagInner, resTempRealInner, ELENUM_REPEAT_FP32, resComputeRepeat, {1, 1, 1, 8, 8, 8});
}


// dst = dst - src
template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::RealScalarVecBatchSub(LocalTensor<float> dst,
    LocalTensor<float> src, uint32_t oneMatOffset)
{
    uint32_t matAlignedSize = this->alignedComplexMatSize;
    uint32_t batchSize = this->batchNumPerRepeat;
    uint8_t oneRowBlockStride = matAlignedSize / ELENUM_BLOCK_FP32; // 8
    uintptr_t oneMatBlockStrideFlag = oneMatOffset / ELENUM_BLOCK_FP32;
    uint8_t oneMatBlockStride = oneMatOffset / ELENUM_BLOCK_FP32; // 64 / 8 = 8, 256 / 8 = 64
    uint32_t singleInstrRepeatTime = (matAlignedSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32;

    if (oneMatBlockStrideFlag * ELENUM_BLOCK_FP32 > MAX_STRIDE_NUM) {
        for (uint32_t j = 0; j < batchSize / ELENUM_BLOCK_FP32; j++) {
            Sub(dst[ELENUM_BLOCK_FP32 * oneMatOffset * j], dst[ELENUM_BLOCK_FP32 * oneMatOffset * j],
                src[ELENUM_BLOCK_FP32 * matAlignedSize * j], ELENUM_REPEAT_FP32, (uint8_t)(singleInstrRepeatTime),
                {oneMatBlockStride, oneMatBlockStride, oneRowBlockStride, 1, 1, 1});
        }
    } else {
        for (uint32_t i = 0; i < singleInstrRepeatTime; i++) {
            Sub(dst[ELENUM_BLOCK_FP32 * i], dst[ELENUM_BLOCK_FP32 * i],
                src[ELENUM_BLOCK_FP32 * i], ELENUM_REPEAT_FP32, (uint8_t)(batchSize / ELENUM_BLOCK_FP32),
                {oneMatBlockStride, oneMatBlockStride, oneRowBlockStride,
                static_cast<uint8_t>(oneMatBlockStride * ELENUM_BLOCK_FP32),
                static_cast<uint8_t>(oneMatBlockStride * ELENUM_BLOCK_FP32),
                static_cast<uint8_t>(oneRowBlockStride * ELENUM_BLOCK_FP32)});
        }
    }
    PipeBarrier<PIPE_V>();
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::ComputeAndUpdateAj(LocalTensor<float> AjReal,
    LocalTensor<float> AjImag, uint32_t oneMatOffset)
{
    /*
        compute and update A[j,:] = A[j,:] - scalar * A[i,:]
    */
    LocalTensor<float> tempRealInner = this->updateTempRealLocal; // size: matAlignedSize * batchSize
    LocalTensor<float> tempImagInner = this->updateTempImagLocal; // size: matAlignedSize * batchSize

    // compuate and update A[j,:] real part
    RealScalarVecBatchSub(AjReal, tempRealInner, oneMatOffset);

    // compuate and update A[j,:] imag part
    RealScalarVecBatchSub(AjImag, tempImagInner, oneMatOffset);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::RealVecScalarBatchDiv(LocalTensor<float> dst,
    LocalTensor<float> src0, LocalTensor<float> src1, uint32_t oneMatOffset)
{
    uint32_t matAlignedSize = this->alignedComplexMatSize;
    uint32_t batchSize = this->batchNumPerRepeat; // 64 aligned, 32
    uint8_t oneRowBlockStride = matAlignedSize / ELENUM_BLOCK_FP32;
    uint8_t oneMatBlockStride = oneMatOffset / ELENUM_BLOCK_FP32;
    uintptr_t oneMatBlockStrideFlag = oneMatOffset / ELENUM_BLOCK_FP32;
    uint32_t singleInstrRepeatTime = (matAlignedSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32;

    if (oneMatBlockStrideFlag * ELENUM_BLOCK_FP32 > MAX_STRIDE_NUM) {
        for (uint32_t j = 0; j < batchSize / ELENUM_BLOCK_FP32; j++) {
            Div(dst[ELENUM_BLOCK_FP32 * oneMatOffset * j], src0[ELENUM_BLOCK_FP32 * matAlignedSize * j],
                src1[ELENUM_REPEAT_FP32 * j], ELENUM_REPEAT_FP32, (uint8_t)(singleInstrRepeatTime),
                {oneMatBlockStride, oneRowBlockStride, 1, 1, 1, 0});
        }
    } else {
        for (uint32_t i = 0; i < singleInstrRepeatTime; i++) {
            Div(dst[ELENUM_BLOCK_FP32 * i], src0[ELENUM_BLOCK_FP32 * i],
                src1, ELENUM_REPEAT_FP32, (uint8_t)(batchSize / ELENUM_BLOCK_FP32),
                {oneMatBlockStride, oneRowBlockStride, 1, static_cast<uint8_t>(ELENUM_BLOCK_FP32 * oneMatBlockStride),
                static_cast<uint8_t>(ELENUM_BLOCK_FP32 * oneRowBlockStride), 8}); // vec.R / scalar.R
        }
    }
    PipeBarrier<PIPE_V>();
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::ComplexVecScalarBatchDiv(LocalTensor<float> vecReal,
    LocalTensor<float> vecImag, uint32_t oneMatOffset)
{
    /*
        update A[i,:] = A[i,:] / aii
        given (vec.R + vec.I) / (scalar.R + scalar.I)
        result real part: (vec.R * scalar.R + vec.I * scalar.I) / (scalar.R^2 + scalar.I^2)
        result imag part: (vec.I * scalar.R - vec.R * scalar.I) / (scalar.R^2 + scalar.I^2)
    */
    LocalTensor<float> scalarRealInner = this->preRealLocal; // should be ub_scalar + 0, read only
    LocalTensor<float> scalarImagInner = this->preImagLocal; // should be ub_scalar + 64, read only
    LocalTensor<float> scalarTempRealInner = this->afterRealLocal;
    LocalTensor<float> scalarTempImagInner = this->afterImagLocal;
    LocalTensor<float> scalarRealBrcbInner = this->scalarBrcbRealLocal;
    LocalTensor<float> scalarImagBrcbInner = this->scalarBrcbImagLocal;
    LocalTensor<float> scalarTempBrcbInner = this->scalarBrcbAugLocal;
    LocalTensor<float> tempRealInner = this->updateTempRealLocal; // size: matAlignedSize * batchSize
    LocalTensor<float> tempImagInner = this->updateTempImagLocal; // size: matAlignedSize * batchSize
    uint32_t matAlignedSize = this->alignedComplexMatSize;
    uint32_t batchSize = this->batchNumPerRepeat;
    uint32_t squareSumRepeat = (batchSize + ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32;
    uint32_t resComputeRepeat = (matAlignedSize * batchSize + ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32;
    uint8_t brcbRepeatTime = (batchSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32;

    // compute scalar.R^2 + scalar.I^2
    // scalar.R^2
    Mul(scalarTempRealInner, scalarRealInner, scalarRealInner,
        ELENUM_REPEAT_FP32, squareSumRepeat, {1, 1, 1, 8, 8, 8});
    // scalar.I^2
    Mul(scalarTempImagInner, scalarImagInner, scalarImagInner,
        ELENUM_REPEAT_FP32, squareSumRepeat, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    // scalar.R^2 + scalar.I^2
    Add(scalarTempRealInner, scalarTempRealInner, scalarTempImagInner,
        ELENUM_REPEAT_FP32, squareSumRepeat, {1, 1, 1, 8, 8, 8});
    // broadcast scalar.R
    Brcb(scalarRealBrcbInner, scalarRealInner, brcbRepeatTime, {1, 8});
    // broadcast scalar.I
    Brcb(scalarImagBrcbInner, scalarImagInner, brcbRepeatTime, {1, 8});
    PipeBarrier<PIPE_V>();
    // broadcast scalar.R^2 + scalar.I^2
    Brcb(scalarTempBrcbInner, scalarTempRealInner, brcbRepeatTime, {1, 8});

    // compute vec.R * scalar.R
    RealScalarVecBatchMul(tempRealInner, vecReal, scalarRealBrcbInner, oneMatOffset);
    // compute vec.I * scalar.I
    RealScalarVecBatchMul(tempImagInner, vecImag, scalarImagBrcbInner, oneMatOffset);
    // compute vec.R * scalar.R + vec.I * scalar.I
    Add(tempRealInner, tempRealInner, tempImagInner, ELENUM_REPEAT_FP32, resComputeRepeat, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    // compute vec.R * scalar.I
    RealScalarVecBatchMul(tempImagInner, vecReal, scalarImagBrcbInner, oneMatOffset);
    // (vec.R * scalar.R + vec.I * scalar.I) / (scalar.R^2 + scalar.I^2)
    RealVecScalarBatchDiv(vecReal, tempRealInner, scalarTempBrcbInner, oneMatOffset);
    // vec.I * scalar.R
    RealScalarVecBatchMul(tempRealInner, vecImag, scalarRealBrcbInner, oneMatOffset);
    // vec.I * scalar.R - vec.R * scalar.I
    Sub(tempRealInner, tempRealInner, tempImagInner, ELENUM_REPEAT_FP32, resComputeRepeat, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    // (vec.I * scalar.R - vec.R * scalar.I) / (scalar.R^2 + scalar.I^2)
    RealVecScalarBatchDiv(vecImag, tempRealInner, scalarTempBrcbInner, oneMatOffset);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::ComplexCombinationAndCopyout(uint32_t outOffset, uint32_t batchSize)
{
    GlobalTensor<uint32_t> gmOffsetInner = this->offsetGM;
    GlobalTensor<T> gmMatOutInner = this->outGM[outOffset];
    LocalTensor<float> matInInner = this->invSepLocal;
    LocalTensor<uint32_t> offsetInner = this->complexCombineOffsetLocal;
    LocalTensor<float> matOutInner = this->matInLocal;

    uint32_t matRowSize = this->n; // complex
    uint32_t matColSize = this->n; // complex
    uint32_t matColFpAlignedSize = this->alignedComplexMatSize; // fp32, single mat, 8
    auto eventId = EVENT_ID0;

    uint32_t repeatTime = (matRowSize * matColFpAlignedSize * batchSize * COMPLEX_ELENUM +
        ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32;

    PipeBarrier<PIPE_ALL>();

    // copy offset in
    uint32_t copyinSize = matRowSize * matColFpAlignedSize * batchSize * COMPLEX_ELENUM;
    copyinSize = (copyinSize + ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32 * ELENUM_REPEAT_FP32;
    CopyVecGmToUbUint32(offsetInner, gmOffsetInner, copyinSize);

    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);

    uint32_t cmdRepeats = repeatTime / MAX_REPEAT_NUM; // 255 repeats per time
    uint32_t remainRepeats = repeatTime % MAX_REPEAT_NUM;
    uint32_t currOffset = 0;
    for (uint32_t i = 0; i < cmdRepeats; i++) {
        Gather(matOutInner[currOffset], matInInner, offsetInner[currOffset], 0, MAX_REPEAT_NUM * ELENUM_REPEAT_FP32);
        currOffset += MAX_REPEAT_NUM * ELENUM_REPEAT_FP32;
        PipeBarrier<PIPE_ALL>();
    }
    if (remainRepeats > 0) {
        Gather(matOutInner[currOffset], matInInner, offsetInner[currOffset], 0, remainRepeats * ELENUM_REPEAT_FP32);
        PipeBarrier<PIPE_ALL>(); // necessary
    }

    if constexpr (IS_FLOAT) {
        LocalTensor<T> outLocal = matOutInner.template ReinterpretCast<T>();
        CopyVecUbToGm(gmMatOutInner, outLocal, matRowSize * matColSize * COMPLEX_ELENUM * batchSize);
    } else {
        LocalTensor<half> castDstLocal = matInInner.template ReinterpretCast<half>();
        uint32_t castRepeatNum = (matRowSize * matColSize * COMPLEX_ELENUM * batchSize - 1) / ELENUM_REPEAT_FP32 + 1;

        Cast(castDstLocal, matOutInner, RoundMode::CAST_NONE, castRepeatNum * ELENUM_REPEAT_FP32);

        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        LocalTensor<T> outLocal = matInInner.template ReinterpretCast<T>();
        CopyVecUbToGm(gmMatOutInner, outLocal, matRowSize * matColSize * COMPLEX_ELENUM * batchSize);
    }
}
}  // namespace CmatinvBatched


__global__ __aicore__ void
cmatinv_batched_kernel(GM_ADDR dA, GM_ADDR dUniReal, GM_ADDR dUniImag,
                       GM_ADDR dOffset, GM_ADDR dAinv, GM_ADDR workSpace,
                       GM_ADDR tilingGm) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);
    uint32_t dtype = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf)); // 0-c32; 1-c64

    if (dtype > 0) {
        CmatinvBatched::CmatinvBatchedAIV<float> op;
        op.Init(dA, dUniReal, dUniImag, dOffset, dAinv, workSpace, tilingGm);
        op.Process();
    } else {
        CmatinvBatched::CmatinvBatchedAIV<half> op;
        op.Init(dA, dUniReal, dUniImag, dOffset, dAinv, workSpace, tilingGm);
        op.Process();
    }
}

void cmatinv_batched_kernel_do(GM_ADDR dA, GM_ADDR dUniReal, GM_ADDR dUniImag,
                               GM_ADDR dOffset, GM_ADDR dAinv,
                               GM_ADDR workSpace, GM_ADDR tilingGm,
                               uint32_t numBlocks, void *stream) {
    cmatinv_batched_kernel<<<numBlocks, nullptr, stream>>>(dA, dUniReal, dUniImag, dOffset, dAinv, workSpace, tilingGm);
}
#endif  // CMATINV_BATCHED_AIV_H