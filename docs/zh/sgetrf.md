## Sgetrf

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
    aclsolverSgetrf：对实数矩阵进行LU分解。

- 计算公式
    $$
    A = P \cdot L \cdot U
    $$
    其中$A$为$m \times n$阶实数矩阵，$P$为置换矩阵，$L$为单位下三角矩阵，$U$为上三角矩阵。

- 示例 \
    输入"A"为：
    ```
    [4.0, 3.0, 2.0, 1.0
     3.0, 4.0, 3.0, 2.0
     2.0, 3.0, 4.0, 3.0
     1.0, 2.0, 3.0, 4.0]
    ```
    输入"m"为： 4\
    输入"n"为： 4\
    调用"aclsolverSgetrf"算子后，\
    输出"A"为L和U合并存储：
    ```
    [4.0, 3.0, 2.0, 1.0
     0.75, 1.75, 1.5, 1.25
     0.5, 0.4286, 2.0, 2.0
     0.25, 0.2857, 0.5, 2.0]
    ```
    输出"ipiv"为pivot信息：
    ```
    [1, 2, 3, 4]
    ```

## 函数原型
- 函数定义
    ```
    aclError aclsolverSgetrf(
        const int64_t m,
        const int64_t n,
        float *A,
        const int64_t lda,
        int32_t *ipiv,
        int32_t *info,
        void *stream);
    ```
- 参数说明：
    <table>
    <tr>
        <td align="center">参数名</td>
        <td align="center">输入输出</td>
        <td align="center">描述</td>
    </tr>
    <tr>
        <td align="center">m</td>
        <td align="center">输入</td>
        <td align="left">矩阵A的行数</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">矩阵A的列数</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">输入/输出</td>
        <td align="left">输入为矩阵A，输出为L和U矩阵<br>数据类型仅支持FLOAT32，数据格式支持ND，shape为[m, n]</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">A左右相邻元素间的内存地址偏移量（当前约束为n）</td>
    </tr>
    <tr>
        <td align="center">ipiv</td>
        <td align="center">输出</td>
        <td align="left">置换矩阵的pivot信息<br>数据类型支持int32_t，数据格式支持ND，shape为[min(m,n), 1]</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">输出</td>
        <td align="left">分解结果信息<br>数据类型支持int32_t</td>
    </tr>
    </table>

- 算子约束： 
  - lda、info参数在当前版本实际未启用。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 调用示例
- 完整代码示例：[aclsolverSgetrf矩阵LU分解示例](../../test/sgetrf/sgetrf_test.cpp)
- 核心调用步骤：

    ```
    #include <vector>
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

        // 构造输入数据
        int64_t m = 4;
        int64_t n = 4;
        size_t aMatrixFileSize = m * n * sizeof(float);
        float* A;
        aclrtMallocHost((void**)(&A), aMatrixFileSize);
        // 填充A矩阵数据...

        int32_t *ipiv = new int32_t[std::min(m, n)];
        int32_t *info;

        // 调用 aclsolverSgetrf
        auto ret = aclsolverSgetrf(m, n, A, n, ipiv, info, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverSgetrf failed. ERROR: %d\n", ret); return ret);

        // 释放资源
        delete[] ipiv;
        aclrtFreeHost(A);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }
    ```
