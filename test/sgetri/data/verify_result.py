#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import numpy as np

# for float32
relative_tol = 5e-3
absolute_tol = 5e-3
error_tol = 1e-4


def verify_result(A, invA, N):
    A = np.fromfile(A, dtype=np.float32).reshape(N, N)
    invA = np.fromfile(invA, dtype=np.float32).reshape(N, N)

    output = np.dot(A, invA).reshape(-1)
    golden = np.eye(N).astype(np.float32).reshape(-1)
    different_element_results = np.isclose(
        output, golden, rtol=relative_tol, atol=absolute_tol, equal_nan=True
    )
    different_element_indexes = np.where(different_element_results == False)[0]
    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(
            "data index: %08d, expected: %.9f, actual: %.9f, rdiff: %.6f"
            % (
                real_index,
                golden_data,
                output_data,
                abs(output_data - golden_data) / abs(golden_data),
            )
        )
        if index == 32:
            break
    error_ratio = float(different_element_indexes.size) / golden.size
    print("error ratio: %.4f, tolrence: %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol


if __name__ == "__main__":
    try:
        res = (
            verify_result(
                "./test/sgetri/data/input/A_gm.bin",
                "./test/sgetri/data/output/A_gm.bin",
                int(sys.argv[1]),
            )
            if len(sys.argv) > 1
            else 32
        )
        if not res:
            print("[Failed] Case accuracy verification failed.")
            sys.exit(1)
        else:
            print("[Success] Case accuracy is verification passed.")
            sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)