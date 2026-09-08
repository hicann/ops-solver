/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if defined(CHEEVJ_HOUSEHOLDER_COMMON_SECTION) && CHEEVJ_HOUSEHOLDER_COMMON_SECTION == 1
    __aicore__ inline void BindGlobals(GM_ADDR matrixReal, GM_ADDR matrixImag, GM_ADDR panelVReal, GM_ADDR panelVImag,
                                       GM_ADDR panelWReal, GM_ADDR panelWImag, GM_ADDR wHReal, GM_ADDR wHImag,
                                       GM_ADDR wHImagNeg, GM_ADDR diagonal, GM_ADDR offDiagonal, GM_ADDR tauReal,
                                       GM_ADDR tauImag, GM_ADDR panelWorkspace, GM_ADDR barrierWorkspace)
    {
        matrixRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matrixReal));
        matrixImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matrixImag));
        panelVRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(panelVReal));
        panelVImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(panelVImag));
        panelWRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(panelWReal));
        panelWImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(panelWImag));
        wHRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wHReal));
        wHImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wHImag));
        wHImagNegGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wHImagNeg));
        diagonalGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(diagonal));
        offDiagonalGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(offDiagonal));
        tauRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tauReal));
        tauImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tauImag));
        internalVRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(panelWorkspace) +
                                            CHEEVJ_HH_COMMON_WS_V_REAL);
        internalVImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(panelWorkspace) +
                                            CHEEVJ_HH_COMMON_WS_V_IMAG);
        internalWRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(panelWorkspace) +
                                            CHEEVJ_HH_COMMON_WS_W_REAL);
        internalWImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(panelWorkspace) +
                                            CHEEVJ_HH_COMMON_WS_W_IMAG);
        productRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(panelWorkspace) +
                                          CHEEVJ_HH_COMMON_WS_Y_REAL);
        productImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(panelWorkspace) +
                                          CHEEVJ_HH_COMMON_WS_Y_IMAG);
        barrierGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(barrierWorkspace));
    }
#elif defined(CHEEVJ_HOUSEHOLDER_COMMON_SECTION) && CHEEVJ_HOUSEHOLDER_COMMON_SECTION == 2
        AscendC::PipeBarrier<PIPE_V>();

        const float diagonal = ApplyPriorReflectors(step, begin, active, vectorReal, vectorImag, tmp0, tmp1);
        float beta = 0.0f;
        float tauR = 0.0f;
        float tauI = 0.0f;
        FormReflector(vectorReal, vectorImag, tmp0, tmp1, result, beta, tauR, tauI);

        uint64_t firstMask[2] = {1ULL, 0ULL};
        AscendC::Duplicate(vectorReal, 1.0f, firstMask, static_cast<uint8_t>(1), 1, 8);
        AscendC::Duplicate(vectorImag, 0.0f, firstMask, static_cast<uint8_t>(1), 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
#elif defined(CHEEVJ_HOUSEHOLDER_COMMON_SECTION) && CHEEVJ_HOUSEHOLDER_COMMON_SECTION == 3
        StoreContiguous(internalVRealGlobal, columnOffset + begin, vectorReal, active);
        StoreContiguous(internalVImagGlobal, columnOffset + begin, vectorImag, active);
        ScatterPanelColumn(panelVRealGlobal, step, begin, vectorReal);
        ScatterPanelColumn(panelVImagGlobal, step, begin, vectorImag);
        StoreScalar(diagonalGlobal, step, tmp0, diagonal);
        StoreScalar(offDiagonalGlobal, step, tmp0, beta);
        StoreScalar(tauRealGlobal, step, tmp0, tauR);
        StoreScalar(tauImagGlobal, step, tmp0, tauI);
#elif defined(CHEEVJ_HOUSEHOLDER_COMMON_SECTION) && CHEEVJ_HOUSEHOLDER_COMMON_SECTION == 15
    __aicore__ inline float ApplyPriorReflectors(int step, int begin, int active,
                                                 const AscendC::LocalTensor<float> &vectorReal,
                                                 const AscendC::LocalTensor<float> &vectorImag,
                                                 const AscendC::LocalTensor<float> &tmp0,
                                                 const AscendC::LocalTensor<float> &tmp1)
    {
        auto previousReal = control2Buf.Get<float>();
        auto previousImag = control3Buf.Get<float>();
        float diagonal = LoadScalar(matrixRealGlobal, step * CHEEVJ_HH_VARIANT_N + step, tmp0);
        for (int previous = 0; previous < step; ++previous)
        {
            const int previousOffset = previous * CHEEVJ_HH_VARIANT_N;
            const float vjr = LoadScalar(internalVRealGlobal, previousOffset + step, tmp0);
            const float vji = LoadScalar(internalVImagGlobal, previousOffset + step, tmp0);
            const float wjr = LoadScalar(internalWRealGlobal, previousOffset + step, tmp0);
            const float wji = LoadScalar(internalWImagGlobal, previousOffset + step, tmp0);
            diagonal -= 2.0f * (vjr * wjr + vji * wji);

            ClearDense(previousReal);
            ClearDense(previousImag);
            LoadContiguous(previousReal, internalVRealGlobal, previousOffset + begin, active);
            LoadContiguous(previousImag, internalVImagGlobal, previousOffset + begin, active);
            ClearActivePadding(previousReal, active);
            ClearActivePadding(previousImag, active);
            SubtractScaled(vectorReal, vectorImag, previousReal, previousImag, wjr, -wji, tmp0, tmp1);

            ClearDense(previousReal);
            ClearDense(previousImag);
            LoadContiguous(previousReal, internalWRealGlobal, previousOffset + begin, active);
            LoadContiguous(previousImag, internalWImagGlobal, previousOffset + begin, active);
            ClearActivePadding(previousReal, active);
            ClearActivePadding(previousImag, active);
            SubtractScaled(vectorReal, vectorImag, previousReal, previousImag, vjr, -vji, tmp0, tmp1);
        }
        return diagonal;
    }

    __aicore__ inline void FormReflector(const AscendC::LocalTensor<float> &vectorReal,
                                         const AscendC::LocalTensor<float> &vectorImag,
                                         const AscendC::LocalTensor<float> &tmp0,
                                         const AscendC::LocalTensor<float> &tmp1,
                                         const AscendC::LocalTensor<float> &result, float &beta, float &tauR,
                                         float &tauI)
    {
        const float alphaReal = ReadLocalControlLane(vectorReal, 0, result);
        const float alphaImag = ReadLocalControlLane(vectorImag, 0, result);
        AscendC::Mul(tmp0, vectorReal, vectorReal, CHEEVJ_HH_VARIANT_N);
        AscendC::Mul(tmp1, vectorImag, vectorImag, CHEEVJ_HH_VARIANT_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(tmp0, tmp0, tmp1, CHEEVJ_HH_VARIANT_N);
        AscendC::PipeBarrier<PIPE_V>();
        const float totalNorm2 = CHEEVJ_HH_VARIANT_REDUCE(tmp0, tmp1, result);
        const float alphaNorm2 = alphaReal * alphaReal + alphaImag * alphaImag;
        const float rawTailNorm2 = totalNorm2 - alphaNorm2;
        const float tailNorm2 = rawTailNorm2 > 0.0f ? rawTailNorm2 : 0.0f;

        beta = alphaReal;
        tauR = 0.0f;
        tauI = 0.0f;
        if (tailNorm2 == 0.0f && alphaImag == 0.0f)
        {
            ClearDense(vectorReal);
            ClearDense(vectorImag);
        }
        else
        {
            ScaleReflector(alphaReal, alphaImag, alphaNorm2, tailNorm2, vectorReal, vectorImag, tmp0, tmp1, result,
                           beta, tauR, tauI);
        }
    }
#elif defined(CHEEVJ_HOUSEHOLDER_COMMON_SECTION) && CHEEVJ_HOUSEHOLDER_COMMON_SECTION == 4
        auto vectorReal = vectorRealBuf.Get<float>();
        auto vectorImag = vectorImagBuf.Get<float>();
        auto yReal = control0Buf.Get<float>();
        auto yImag = control1Buf.Get<float>();
        auto tmp0 = accumulatorRealBuf.Get<float>();
        auto tmp1 = accumulatorImagBuf.Get<float>();
        auto result = resultBuf.Get<float>();

        ClearDense(vectorReal);
        ClearDense(vectorImag);
        LoadContiguous(vectorReal, internalVRealGlobal, columnOffset + begin, active);
        LoadContiguous(vectorImag, internalVImagGlobal, columnOffset + begin, active);
        ClearActivePadding(vectorReal, active);
        ClearActivePadding(vectorImag, active);
        ClearDense(yReal);
        ClearDense(yImag);
        LoadContiguous(yReal, productRealGlobal, begin, active);
        LoadContiguous(yImag, productImagGlobal, begin, active);
        ClearActivePadding(yReal, active);
        ClearActivePadding(yImag, active);

        ApplyPreviousWCorrections(step, begin, active, vectorReal, vectorImag, yReal, yImag, tmp0, tmp1, result);
        const float tauR = LoadScalar(tauRealGlobal, step, tmp0);
        const float tauI = LoadScalar(tauImagGlobal, step, tmp0);
        MultiplyInPlace(yReal, yImag, tauR, tauI, tmp0, tmp1);

        float dotReal = 0.0f;
        float dotImag = 0.0f;
        DotConjugate(yReal, yImag, vectorReal, vectorImag, tmp0, tmp1, result, dotReal, dotImag);
        const float gammaReal = -0.5f * (tauR * dotReal - tauI * dotImag);
        const float gammaImag = -0.5f * (tauR * dotImag + tauI * dotReal);
        AddScaled(yReal, yImag, vectorReal, vectorImag, gammaReal, gammaImag, tmp0, tmp1);

        StoreContiguous(internalWRealGlobal, columnOffset + begin, yReal, active);
        StoreContiguous(internalWImagGlobal, columnOffset + begin, yImag, active);
        StoreContiguous(wHRealGlobal, columnOffset + begin, yReal, active);
#elif defined(CHEEVJ_HOUSEHOLDER_COMMON_SECTION) && CHEEVJ_HOUSEHOLDER_COMMON_SECTION == 5
        AscendC::PipeBarrier<PIPE_V>();
        StoreContiguous(wHImagGlobal, columnOffset + begin, tmp0, active);
        StoreContiguous(wHImagNegGlobal, columnOffset + begin, yImag, active);
#elif defined(CHEEVJ_HOUSEHOLDER_COMMON_SECTION) && CHEEVJ_HOUSEHOLDER_COMMON_SECTION == 16
    __aicore__ inline void ApplyPreviousWCorrections(int step, int begin, int active,
                                                     const AscendC::LocalTensor<float> &vectorReal,
                                                     const AscendC::LocalTensor<float> &vectorImag,
                                                     const AscendC::LocalTensor<float> &yReal,
                                                     const AscendC::LocalTensor<float> &yImag,
                                                     const AscendC::LocalTensor<float> &tmp0,
                                                     const AscendC::LocalTensor<float> &tmp1,
                                                     const AscendC::LocalTensor<float> &result)
    {
        auto previousReal = control2Buf.Get<float>();
        auto previousImag = control3Buf.Get<float>();
        for (int previous = 0; previous < step; ++previous)
        {
            const int previousOffset = previous * CHEEVJ_HH_VARIANT_N;
            ClearDense(previousReal);
            ClearDense(previousImag);
            LoadContiguous(previousReal, internalWRealGlobal, previousOffset + begin, active);
            LoadContiguous(previousImag, internalWImagGlobal, previousOffset + begin, active);
            ClearActivePadding(previousReal, active);
            ClearActivePadding(previousImag, active);
            float dotWReal = 0.0f;
            float dotWImag = 0.0f;
            DotConjugate(previousReal, previousImag, vectorReal, vectorImag, tmp0, tmp1, result, dotWReal, dotWImag);

            ClearDense(previousReal);
            ClearDense(previousImag);
            LoadContiguous(previousReal, internalVRealGlobal, previousOffset + begin, active);
            LoadContiguous(previousImag, internalVImagGlobal, previousOffset + begin, active);
            ClearActivePadding(previousReal, active);
            ClearActivePadding(previousImag, active);
            float dotVReal = 0.0f;
            float dotVImag = 0.0f;
            DotConjugate(previousReal, previousImag, vectorReal, vectorImag, tmp0, tmp1, result, dotVReal, dotVImag);
            SubtractScaled(yReal, yImag, previousReal, previousImag, dotWReal, dotWImag, tmp0, tmp1);

            ClearDense(previousReal);
            ClearDense(previousImag);
            LoadContiguous(previousReal, internalWRealGlobal, previousOffset + begin, active);
            LoadContiguous(previousImag, internalWImagGlobal, previousOffset + begin, active);
            ClearActivePadding(previousReal, active);
            ClearActivePadding(previousImag, active);
            SubtractScaled(yReal, yImag, previousReal, previousImag, dotVReal, dotVImag, tmp0, tmp1);
        }
    }
#elif defined(CHEEVJ_HOUSEHOLDER_COMMON_SECTION) && CHEEVJ_HOUSEHOLDER_COMMON_SECTION == 18
    __aicore__ inline void ApplyPreviousWCorrectionsN1024(
        int step, int begin, int active, const AscendC::LocalTensor<float> &vectorReal,
        const AscendC::LocalTensor<float> &vectorImag, const AscendC::LocalTensor<float> &yReal,
        const AscendC::LocalTensor<float> &yImag, const AscendC::LocalTensor<float> &tmp0,
        const AscendC::LocalTensor<float> &tmp1, const AscendC::LocalTensor<float> &result)
    {
        auto previousImag = control3Buf.Get<float>();
        auto previousReal = control2Buf.Get<float>();
        for (int previous = 0; previous < step; ++previous)
        {
#elif defined(CHEEVJ_HOUSEHOLDER_COMMON_SECTION) && CHEEVJ_HOUSEHOLDER_COMMON_SECTION == 8
    __aicore__ inline void AccumulateGemvSegments(const AscendC::LocalTensor<float> &matrixReal,
                                                  const AscendC::LocalTensor<float> &matrixImag,
                                                  const AscendC::LocalTensor<float> &vectorReal,
                                                  const AscendC::LocalTensor<float> &vectorImag,
                                                  const AscendC::LocalTensor<float> &vectorImagNeg,
                                                  const AscendC::LocalTensor<float> &accumulatorReal,
                                                  const AscendC::LocalTensor<float> &accumulatorImag)
    {
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
#elif defined(CHEEVJ_HOUSEHOLDER_COMMON_SECTION) && CHEEVJ_HOUSEHOLDER_COMMON_SECTION == 9
    __aicore__ inline void LoadContiguous(const AscendC::LocalTensor<float> &destination,
                                          const AscendC::GlobalTensor<float> &source, int offset, int count)
    {
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(destination, source[offset], copy, pad);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StoreContiguous(AscendC::GlobalTensor<float> destination, int offset,
                                           const AscendC::LocalTensor<float> &source, int count)
    {
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(destination[offset], source, copy);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline float LoadScalar(const AscendC::GlobalTensor<float> &source, int offset,
                                       const AscendC::LocalTensor<float> &staging)
    {
        AscendC::DataCopyExtParams copy{1, sizeof(float), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(staging, source[offset], copy, pad);
        AscendC::PipeBarrier<PIPE_ALL>();
        return staging.GetValue(0);
    }

    __aicore__ inline float ReadLocalControlLane(const AscendC::LocalTensor<float> &source, int offset,
                                                 const AscendC::LocalTensor<float> &staging)
    {
        // Scalar-unit access is confined to this control staging lane.  The
        // matrix/vector source itself remains on the vector path.
        const int alignedOffset = offset & ~7;
        const int lane = offset & 7;
        AscendC::Muls(staging, source[alignedOffset], 1.0f, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        return staging.GetValue(lane);
    }

    __aicore__ inline void StoreScalar(AscendC::GlobalTensor<float> destination, int offset,
                                       const AscendC::LocalTensor<float> &staging, float value)
    {
        AscendC::Duplicate(staging, value, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams copy{1, sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(destination[offset], staging, copy);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
#ifdef CHEEVJ_HH_SCATTER_INDEX_COUNT
    __aicore__ inline void PrepareScatterIndex()
    {
        auto signedIndex = scatterIndexBuf.Get<int32_t>();
        auto index = signedIndex.template ReinterpretCast<uint32_t>();
        AscendC::CreateVecIndex(signedIndex, static_cast<int32_t>(0),
                                static_cast<uint32_t>(CHEEVJ_HH_SCATTER_INDEX_COUNT));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(index, index, static_cast<uint32_t>(3), CHEEVJ_HH_SCATTER_INDEX_COUNT);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(index, index, static_cast<uint32_t>(2), CHEEVJ_HH_SCATTER_INDEX_COUNT);
        AscendC::PipeBarrier<PIPE_V>();
    }
#endif
#elif defined(CHEEVJ_HOUSEHOLDER_COMMON_SECTION) && CHEEVJ_HOUSEHOLDER_COMMON_SECTION == 10
    AscendC::GlobalTensor<float> matrixRealGlobal;
    AscendC::GlobalTensor<float> matrixImagGlobal;
    AscendC::GlobalTensor<float> panelVRealGlobal;
    AscendC::GlobalTensor<float> panelVImagGlobal;
    AscendC::GlobalTensor<float> panelWRealGlobal;
    AscendC::GlobalTensor<float> panelWImagGlobal;
    AscendC::GlobalTensor<float> wHRealGlobal;
    AscendC::GlobalTensor<float> wHImagGlobal;
    AscendC::GlobalTensor<float> wHImagNegGlobal;
    AscendC::GlobalTensor<float> diagonalGlobal;
    AscendC::GlobalTensor<float> offDiagonalGlobal;
    AscendC::GlobalTensor<float> tauRealGlobal;
    AscendC::GlobalTensor<float> tauImagGlobal;
    AscendC::GlobalTensor<float> internalVRealGlobal;
    AscendC::GlobalTensor<float> internalVImagGlobal;
    AscendC::GlobalTensor<float> internalWRealGlobal;
    AscendC::GlobalTensor<float> internalWImagGlobal;
    AscendC::GlobalTensor<float> productRealGlobal;
    AscendC::GlobalTensor<float> productImagGlobal;
    AscendC::GlobalTensor<int32_t> barrierGlobal;

    AscendC::TBuf<AscendC::TPosition::VECCALC> matrixRealBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matrixImagBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> vectorRealBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> vectorImagBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> vectorImagNegBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumulatorRealBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumulatorImagBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> control0Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> control1Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> control2Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> control3Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scatterBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scatterIndexBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> barrierBuf;
#elif defined(CHEEVJ_HOUSEHOLDER_COMMON_SECTION) && CHEEVJ_HOUSEHOLDER_COMMON_SECTION == 11
    __aicore__ inline void ProcessPanelStep(int block, int step, const AscendC::LocalTensor<float> &matrixReal,
                                           const AscendC::LocalTensor<float> &matrixImag,
                                           const AscendC::LocalTensor<int32_t> &barrierLocal)
    {
        if (block == 0)
        {
            BuildReflector(step);
        }

        PanelBarrier(barrierLocal);
        if (block != 0)
        {
#elif defined(CHEEVJ_HOUSEHOLDER_COMMON_SECTION) && CHEEVJ_HOUSEHOLDER_COMMON_SECTION == 13
    __aicore__ inline void ComputeReflectorScalars(float alphaReal, float alphaImag, float alphaNorm2,
                                                   float tailNorm2,
                                                   const AscendC::LocalTensor<float> &result, float &beta,
                                                   float &tauR, float &tauI, float &scaleReal, float &scaleImag)
    {
        const float norm = CheevjHouseholderSqrt(alphaNorm2 + tailNorm2, result);
        beta = alphaReal >= 0.0f ? -norm : norm;
        tauR = (beta - alphaReal) / beta;
        tauI = -alphaImag / beta;
        const float denominatorReal = alphaReal - beta;
        const float denominatorImag = alphaImag;
        const float denominatorNorm = denominatorReal * denominatorReal + denominatorImag * denominatorImag;
        scaleReal = denominatorReal / denominatorNorm;
        scaleImag = denominatorImag / denominatorNorm;
    }
#elif defined(CHEEVJ_HOUSEHOLDER_COMMON_SECTION) && CHEEVJ_HOUSEHOLDER_COMMON_SECTION == 19
    __aicore__ inline void ScaleReflector(float alphaReal, float alphaImag, float alphaNorm2, float tailNorm2,
                                          const AscendC::LocalTensor<float> &vectorReal,
                                          const AscendC::LocalTensor<float> &vectorImag,
                                          const AscendC::LocalTensor<float> &tmp0,
                                          const AscendC::LocalTensor<float> &tmp1,
                                          const AscendC::LocalTensor<float> &result, float &beta, float &tauR,
                                          float &tauI)
    {
        float scaleReal = 0.0f;
        float scaleImag = 0.0f;
        ComputeReflectorScalars(alphaReal, alphaImag, alphaNorm2, tailNorm2, result, beta, tauR, tauI, scaleReal,
                                scaleImag);
        AscendC::Muls(tmp0, vectorReal, scaleReal, CHEEVJ_HH_VARIANT_N);
        AscendC::Muls(tmp1, vectorImag, scaleImag, CHEEVJ_HH_VARIANT_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(tmp0, tmp0, tmp1, CHEEVJ_HH_VARIANT_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(tmp1, vectorImag, scaleReal, CHEEVJ_HH_VARIANT_N);
        auto previousReal = control2Buf.Get<float>();
        AscendC::Muls(previousReal, vectorReal, scaleImag, CHEEVJ_HH_VARIANT_N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(vectorImag, tmp1, previousReal, CHEEVJ_HH_VARIANT_N);
        AscendC::Muls(vectorReal, tmp0, 1.0f, CHEEVJ_HH_VARIANT_N);
        AscendC::PipeBarrier<PIPE_V>();
    }
#endif
