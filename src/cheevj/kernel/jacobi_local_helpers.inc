/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    __aicore__ inline void InitBuffers()
    {
        pipe.InitBuffer(inputBuf, CHEEVJ_COMPLEX_FLOATS * sizeof(float));
        pipe.InitBuffer(outputBuf, CHEEVJ_COMPLEX_FLOATS * sizeof(float));
        pipe.InitBuffer(matrixRealBuf, CHEEVJ_MATRIX_ELEMS * sizeof(float));
        pipe.InitBuffer(matrixImagBuf, CHEEVJ_MATRIX_ELEMS * sizeof(float));
        pipe.InitBuffer(vectorRealBuf, CHEEVJ_MATRIX_ELEMS * sizeof(float));
        pipe.InitBuffer(vectorImagBuf, CHEEVJ_MATRIX_ELEMS * sizeof(float));
        pipe.InitBuffer(wBuf, CHEEVJ_W_FLOATS * sizeof(float));
        pipe.InitBuffer(temp0Buf, CHEEVJ_DEVICE_MAX_N * sizeof(float));
        pipe.InitBuffer(temp1Buf, CHEEVJ_DEVICE_MAX_N * sizeof(float));
        pipe.InitBuffer(temp2Buf, CHEEVJ_DEVICE_MAX_N * sizeof(float));
        pipe.InitBuffer(temp3Buf, CHEEVJ_DEVICE_MAX_N * sizeof(float));
        pipe.InitBuffer(temp4Buf, CHEEVJ_DEVICE_MAX_N * sizeof(float));
    }

    __aicore__ inline int PlanarOffset(int row, int col, int n) { return row + col * n; }

    __aicore__ inline int ComplexOffset(int row, int col, int n) { return PlanarOffset(row, col, n) * 2; }

    __aicore__ inline Complex32 GetInputComplex(LocalTensor<float> inputLocal, int row, int col, int n)
    {
        const int offset = ComplexOffset(row, col, n);
        return MakeComplex(inputLocal.GetValue(offset), inputLocal.GetValue(offset + 1));
    }

    __aicore__ inline Complex32 GetPlanarComplex(LocalTensor<float> realLocal, LocalTensor<float> imagLocal, int row,
                                                 int col, int n)
    {
        const int offset = PlanarOffset(row, col, n);
        return MakeComplex(realLocal.GetValue(offset), imagLocal.GetValue(offset));
    }

    __aicore__ inline void SetPlanarComplex(LocalTensor<float> realLocal, LocalTensor<float> imagLocal, int row,
                                            int col, int n, Complex32 value)
    {
        const int offset = PlanarOffset(row, col, n);
        realLocal.SetValue(offset, value.real);
        imagLocal.SetValue(offset, value.imag);
    }

    __aicore__ inline Complex32 LoadHermitianValue(LocalTensor<float> inputLocal, int row, int col, int n, bool lower)
    {
        if (row == col)
        {
            Complex32 diag = GetInputComplex(inputLocal, row, col, n);
            return MakeComplex(diag.real, 0.0f);
        }
        if (lower)
        {
            if (row > col)
            {
                return GetInputComplex(inputLocal, row, col, n);
            }
            return ComplexConj(GetInputComplex(inputLocal, col, row, n));
        }
        if (row < col)
        {
            return GetInputComplex(inputLocal, row, col, n);
        }
        return ComplexConj(GetInputComplex(inputLocal, col, row, n));
    }

    __aicore__ inline void CopyInputToLocal(LocalTensor<float> inputLocal, int n)
    {
        const uint32_t bytes = static_cast<uint32_t>(n * n * 2 * sizeof(float));
        AscendC::DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(inputLocal, aGlobal[0], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyEigenvaluesToGlobal(LocalTensor<float> wLocal, int n)
    {
        PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(wGlobal[0], wLocal, copyParams);
    }

    __aicore__ inline void CopyTinyVectorsToGlobal(LocalTensor<float> outLocal, int n)
    {
        PipeBarrier<PIPE_ALL>();
        const uint32_t bytes = static_cast<uint32_t>(n * n * 2 * sizeof(float));
        AscendC::DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        AscendC::DataCopyPad(aGlobal[0], outLocal, copyParams);
    }

    __aicore__ inline void SetOutputComplex(LocalTensor<float> outLocal, int row, int col, int n, Complex32 value)
    {
        const int offset = ComplexOffset(row, col, n);
        outLocal.SetValue(offset, value.real);
        outLocal.SetValue(offset + 1, value.imag);
    }

    __aicore__ inline void SolveTinyHermitian(LocalTensor<float> inputLocal, LocalTensor<float> outLocal,
                                              LocalTensor<float> wLocal, int n, bool computeVectors, int uplo)
    {
        const float a00 = GetInputComplex(inputLocal, 0, 0, n).real;
        if (n == 1)
        {
            wLocal.SetValue(0, a00);
            if (computeVectors)
            {
                SetOutputComplex(outLocal, 0, 0, n, MakeComplex(1.0f, 0.0f));
            }
            return;
        }

        const bool lower = IsLowerMode(uplo);
        const float a11 = GetInputComplex(inputLocal, 1, 1, n).real;
        Complex32 upper01 =
            lower ? ComplexConj(GetInputComplex(inputLocal, 1, 0, n)) : GetInputComplex(inputLocal, 0, 1, n);
        const float absB2 = upper01.real * upper01.real + upper01.imag * upper01.imag;
        if (absB2 <= CHEEVJ_MIN_PIVOT)
        {
            if (a00 <= a11)
            {
                wLocal.SetValue(0, a00);
                wLocal.SetValue(1, a11);
                if (computeVectors)
                {
                    WriteTinyIdentity(outLocal, n);
                }
            }
            else
            {
                wLocal.SetValue(0, a11);
                wLocal.SetValue(1, a00);
                if (computeVectors)
                {
                    WriteTinySwap(outLocal, n);
                }
            }
            return;
        }

        const float traceHalf = 0.5f * (a00 + a11);
        const float diffHalf = 0.5f * (a00 - a11);
        const float radius = SqrtApprox(diffHalf * diffHalf + absB2);
        const float lambda0 = traceHalf - radius;
        const float lambda1 = traceHalf + radius;
        wLocal.SetValue(0, lambda0);
        wLocal.SetValue(1, lambda1);

        if (computeVectors)
        {
            WriteTinyEigenvectors(outLocal, n, upper01, a00, lambda0, lambda1);
        }
    }

    __aicore__ inline void WriteTinyIdentity(LocalTensor<float> outLocal, int n)
    {
        for (int idx = 0; idx < n * n * 2; ++idx)
        {
            outLocal.SetValue(idx, 0.0f);
        }
        SetOutputComplex(outLocal, 0, 0, n, MakeComplex(1.0f, 0.0f));
        SetOutputComplex(outLocal, 1, 1, n, MakeComplex(1.0f, 0.0f));
    }

    __aicore__ inline void WriteTinySwap(LocalTensor<float> outLocal, int n)
    {
        for (int idx = 0; idx < n * n * 2; ++idx)
        {
            outLocal.SetValue(idx, 0.0f);
        }
        SetOutputComplex(outLocal, 1, 0, n, MakeComplex(1.0f, 0.0f));
        SetOutputComplex(outLocal, 0, 1, n, MakeComplex(1.0f, 0.0f));
    }

    __aicore__ inline void WriteTinyEigenvectors(LocalTensor<float> outLocal, int n, Complex32 upper01, float a00,
                                                 float lambda0, float lambda1)
    {
        for (int idx = 0; idx < n * n * 2; ++idx)
        {
            outLocal.SetValue(idx, 0.0f);
        }
        WriteTinyEigenvector(outLocal, n, 0, upper01, lambda0 - a00);
        WriteTinyEigenvector(outLocal, n, 1, upper01, lambda1 - a00);
    }

    __aicore__ inline void WriteTinyEigenvector(LocalTensor<float> outLocal, int n, int col, Complex32 upper01,
                                                float secondReal)
    {
        const float norm =
            SqrtApprox(upper01.real * upper01.real + upper01.imag * upper01.imag + secondReal * secondReal);
        const float invNorm = norm > 0.0f ? 1.0f / norm : 1.0f;
        SetOutputComplex(outLocal, 0, col, n, MakeComplex(upper01.real * invNorm, upper01.imag * invNorm));
        SetOutputComplex(outLocal, 1, col, n, MakeComplex(secondReal * invNorm, 0.0f));
    }

    __aicore__ inline void MergeVectorsAndCopyOut(LocalTensor<float> vReal, LocalTensor<float> vImag,
                                                  LocalTensor<float> outLocal, int n)
    {
        for (int col = 0; col < n; ++col)
        {
            for (int row = 0; row < n; ++row)
            {
                const int planar = PlanarOffset(row, col, n);
                const int complex = ComplexOffset(row, col, n);
                outLocal.SetValue(complex, vReal.GetValue(planar));
                outLocal.SetValue(complex + 1, vImag.GetValue(planar));
            }
        }
        PipeBarrier<PIPE_ALL>();
        const uint32_t bytes = static_cast<uint32_t>(n * n * 2 * sizeof(float));
        AscendC::DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        AscendC::DataCopyPad(aGlobal[0], outLocal, copyParams);
    }

    __aicore__ inline void ReconstructHermitian(LocalTensor<float> inputLocal, LocalTensor<float> aReal,
                                                LocalTensor<float> aImag, int n, int uplo)
    {
        const bool lower = IsLowerMode(uplo);
        for (int col = 0; col < n; ++col)
        {
            for (int row = 0; row < n; ++row)
            {
                SetPlanarComplex(aReal, aImag, row, col, n, LoadHermitianValue(inputLocal, row, col, n, lower));
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitIdentity(LocalTensor<float> vReal, LocalTensor<float> vImag, int n)
    {
        Duplicate(vReal, 0.0f, n * n);
        Duplicate(vImag, 0.0f, n * n);
        PipeBarrier<PIPE_V>();
        for (int idx = 0; idx < n; ++idx)
        {
            vReal.SetValue(PlanarOffset(idx, idx, n), 1.0f);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline float EstimateNorm(LocalTensor<float> aReal, LocalTensor<float> aImag, int n)
    {
        float norm = 1.0f;
        for (int row = 0; row < n; ++row)
        {
            float rowSum = 0.0f;
            for (int col = 0; col < n; ++col)
            {
                rowSum += ComplexAbs(GetPlanarComplex(aReal, aImag, row, col, n));
            }
            norm = MaxFloat(norm, rowSum);
        }
        return norm;
    }

    __aicore__ inline void RotateColumns(LocalTensor<float> realLocal, LocalTensor<float> imagLocal, int n, int p,
                                         int q, float c, float sr, float si)
    {
        if (n % 8 != 0)
        {
            RotateColumnsScalar(realLocal, imagLocal, n, p, q, c, sr, si);
            return;
        }

        LocalTensor<float> newPReal = temp0Buf.Get<float>();
        LocalTensor<float> newPImag = temp1Buf.Get<float>();
        LocalTensor<float> newQReal = temp2Buf.Get<float>();
        LocalTensor<float> newQImag = temp3Buf.Get<float>();
        LocalTensor<float> scratch = temp4Buf.Get<float>();

        const int pOffset = PlanarOffset(0, p, n);
        const int qOffset = PlanarOffset(0, q, n);

        Muls(newPReal, realLocal[pOffset], c, n);
        Muls(scratch, realLocal[qOffset], sr, n);
        Sub(newPReal, newPReal, scratch, n);
        Muls(scratch, imagLocal[qOffset], si, n);
        Sub(newPReal, newPReal, scratch, n);

        Muls(newPImag, imagLocal[pOffset], c, n);
        Muls(scratch, imagLocal[qOffset], sr, n);
        Sub(newPImag, newPImag, scratch, n);
        Muls(scratch, realLocal[qOffset], si, n);
        Add(newPImag, newPImag, scratch, n);

        Muls(newQReal, realLocal[pOffset], sr, n);
        Muls(scratch, imagLocal[pOffset], si, n);
        Sub(newQReal, newQReal, scratch, n);
        Muls(scratch, realLocal[qOffset], c, n);
        Add(newQReal, newQReal, scratch, n);

        Muls(newQImag, imagLocal[pOffset], sr, n);
        Muls(scratch, realLocal[pOffset], si, n);
        Add(newQImag, newQImag, scratch, n);
        Muls(scratch, imagLocal[qOffset], c, n);
        Add(newQImag, newQImag, scratch, n);
        PipeBarrier<PIPE_V>();

        Muls(realLocal[pOffset], newPReal, 1.0f, n);
        Muls(imagLocal[pOffset], newPImag, 1.0f, n);
        Muls(realLocal[qOffset], newQReal, 1.0f, n);
        Muls(imagLocal[qOffset], newQImag, 1.0f, n);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RotateColumnsScalar(LocalTensor<float> realLocal, LocalTensor<float> imagLocal, int n,
                                               int p, int q, float c, float sr, float si)
    {
        const int pOffset = PlanarOffset(0, p, n);
        const int qOffset = PlanarOffset(0, q, n);
        for (int row = 0; row < n; ++row)
        {
            const float pReal = realLocal.GetValue(pOffset + row);
            const float pImag = imagLocal.GetValue(pOffset + row);
            const float qReal = realLocal.GetValue(qOffset + row);
            const float qImag = imagLocal.GetValue(qOffset + row);
            realLocal.SetValue(pOffset + row, c * pReal - sr * qReal - si * qImag);
            imagLocal.SetValue(pOffset + row, c * pImag - sr * qImag + si * qReal);
            realLocal.SetValue(qOffset + row, sr * pReal - si * pImag + c * qReal);
            imagLocal.SetValue(qOffset + row, sr * pImag + si * pReal + c * qImag);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ApplyMatrixRotation(LocalTensor<float> aReal, LocalTensor<float> aImag, int n, int p, int q,
                                               float absPq, float t, float c, Complex32 s)
    {
        const float app = aReal.GetValue(PlanarOffset(p, p, n));
        const float aqq = aReal.GetValue(PlanarOffset(q, q, n));

        RotateColumns(aReal, aImag, n, p, q, c, s.real, s.imag);

        for (int k = 0; k < n; ++k)
        {
            if (k == p || k == q)
            {
                continue;
            }
            const int kp = PlanarOffset(k, p, n);
            const int kq = PlanarOffset(k, q, n);
            const int pk = PlanarOffset(p, k, n);
            const int qk = PlanarOffset(q, k, n);
            aReal.SetValue(pk, aReal.GetValue(kp));
            aImag.SetValue(pk, -aImag.GetValue(kp));
            aReal.SetValue(qk, aReal.GetValue(kq));
            aImag.SetValue(qk, -aImag.GetValue(kq));
        }

        aReal.SetValue(PlanarOffset(p, p, n), app - t * absPq);
        aImag.SetValue(PlanarOffset(p, p, n), 0.0f);
        aReal.SetValue(PlanarOffset(q, q, n), aqq + t * absPq);
        aImag.SetValue(PlanarOffset(q, q, n), 0.0f);
        SetPlanarComplex(aReal, aImag, p, q, n, MakeComplex(0.0f, 0.0f));
        SetPlanarComplex(aReal, aImag, q, p, n, MakeComplex(0.0f, 0.0f));
    }

    __aicore__ inline void ApplyVectorRotation(LocalTensor<float> vReal, LocalTensor<float> vImag, int n, int p, int q,
                                               float c, Complex32 s)
    {
        RotateColumns(vReal, vImag, n, p, q, c, s.real, s.imag);
    }
