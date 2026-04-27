## CmatinvBatched

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |


## 功能说明

- 接口功能\
    aclsolverCmatinvBatched：计算批量复数矩阵的逆矩阵。

- 计算公式
    $$
    A^{-1}A = I
    $$
    其中$A$为$n \times n$阶非奇异复数方阵，$I$为$n$阶单位矩阵。

- 示例 \
    输入"A"为：
    ```
    [2-2i ,1-i ,1-i, 1-i
    1-i, 2-2i, 1-i ,1-i
    1-i, 1-i ,2-2i ,1-i
    1-i ,1-i, 1-i, 2-2i]

    [3-3i ,1-i, 1-i, 1-i
    1-i, 3-3i, 1-i ,1-i
    1-i, 1-i ,3-3i ,1-i
    1-i, 1-i, 1-i, 3-3i]
    ```
    输入"n"为： 4\
    输入"batchSize"为：2\
    调用"aclsolverCmatinvBatched"算子后，\
    输出"Ainv"为：
    ```
    [0.4+0.4i, -0.1-0.1i,-0.1-0.1i ,-0.1-0.1i
    -0.1-0.1i ,0.4+0.4i ,-0.1-0.1i ,-0.1-0.1i
    -0.1-0.1i ,-0.1-0.1i,0.4+0.4i ,-0.1-0.1i
    -0.1-0.1i ,-0.1-0.1i,-0.1-0.1i,0.4+0.4i]

    [0.208+0.208i,-0.0417-0.0417i,-0.0417-0.0417i,-0.0417-0.0417i
    -0.0417-0.0417i,0.208+0.208i,-0.0417-0.0417i,-0.0417-0.0417i
    -0.0417-0.0417i,-0.0417-0.0417i,0.208+0.208i,-0.0417-0.0417i
    -0.0417-0.0417i,-0.0417-0.0417i,-0.0417-0.0417i,0.208+0.208i]
    ```

## 函数原型
- 函数定义
    ```
    aclError aclsolverCmatinvBatched(
        aclsolverHandle_t handle,
        const int64_t n, 
        std::complex<float> *A, 
        const int64_t lda, 
        std::complex<float> *Ainv, 
        const int64_t lda_inv, 
        int32_t *info, 
        int64_t batchSize);
    ```
- 参数说明：
    <table>
    <tr>
        <td align="center">参数名</td>
        <td align="center">输入输出</td>
        <td align="center">描述</td>
    </tr>
    <tr>
        <td align="center">handle</td>
        <td align="center">输入</td>
        <td align="left">solver handle，通过aclsolverCreate创建</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">单个矩阵A的行数</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">输入</td>
        <td align="left">公式中的矩阵A，行主序<br>数据类型仅支持COMPLEX64，数据格式支持ND，shape为[batch, n, n]</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">A左右相邻元素间的内存地址偏移量（当前约束为n）</td>
    </tr>
    <tr>
        <td align="center">Ainv</td>
        <td align="center">输入</td>
        <td align="left">输出的逆矩阵<br>数据类型仅支持COMPLEX64，数据格式支持ND，shape为[batch, n, n]</td>
    </tr>
    <tr>
        <td align="center">lda_inv</td>
        <td align="center">输入</td>
        <td align="left">输出的逆矩阵的左右相邻元素间的内存地址偏移量（当前约束为n）</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">输入</td>
        <td align="left">每个batch矩阵的求逆结果信息<br>数据类型支持int32_t，数据格式支持ND，shape为[batch, 1]</td>
    </tr>
    <tr>
        <td align="center">batchSize</td>
        <td align="center">输入</td>
        <td align="left">复数矩阵求逆中的矩阵数量</td>
    </tr>
    </table>

- 算子约束： 
  - lda、lda_inv、info参数在当前版本实际未启用。
  - 入参n小于等于256。
  - 入参batchSize小于等于3000。

- 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 调用示例
- 完整代码示例：[aclsolverCmatinvBatched批量矩阵求逆示例](../../test/cmatinv_batched/cmatinv_batched_test.cpp)
- 核心调用步骤：

    ```
    #include <vector>
    #include <complex>
    #include "acl/acl.h"
    #include "cann_ops_solver.h"

    int32_t main(int32_t argc, char *argv[])
    {
        // 固定写法，acl初始化
        int32_t deviceId = 0;
        aclrtStream stream = nullptr;
        aclInit(nullptr);
        aclrtSetDevice(deviceId);
        aclrtCreateStream(&stream);

        // 创建solver handle并设置stream
        aclsolverHandle_t handle = nullptr;
        aclsolverCreate(&handle);
        aclsolverSetStream(handle, stream);

        // 构造输入数据
        int64_t batchSize = 2;
        int64_t n = 3;
        int64_t tensorASize = batchSize * n * n;
        std::vector<int64_t> shape = {batchSize, n, n };
        std::vector<std::complex<float>> tensorInAData;
        std::vector<std::complex<float>> tensorInAinvData;
        std::vector<int32_t> tensorInInfoData;
        tensorInAData.resize(tensorASize);
        tensorInAinvData.resize(tensorASize);
        tensorInInfoData.resize(batchSize);
        for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            for (int32_t i = 0; i < n; i++) {
                for (int32_t j = 0; j < n; j++) {
                    if (i == j) {
                        tensorInAData[n * n * batchIdx + n * i + j] = {2.0f + batchIdx, -2.0f - batchIdx};
                    } else {
                        tensorInAData[n * n * batchIdx + n * i + j] = {1.0f, -1.0f};
                    }
                }
            }
        }
        for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            for (int32_t i = 0; i < n; i++) {
                for (int32_t j = 0; j < n; j++) {
                    tensorInAinvData[n * n * batchIdx + n * i + j] = {-1.0f, -1.0f};
                }
            }
        }
        for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            tensorInInfoData[batchIdx] = 0;
        }

        // 调用 aclsolverCmatinvBatched
        auto ret = aclsolverCmatinvBatched(handle, n, tensorInAData.data(), n, tensorInAinvData.data(), n,tensorInInfoData.data(), batchSize);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCmatinvBatched failed. ERROR: %d\n", ret); return ret);

        // 释放资源
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 1;
    }
    ```