# Atlas 950 单精度实数QR分解与最小二乘 任务书

## 1. 任务概述

在昇腾 NPU 上基于 AscendC/CATLASS，开发稠密 QR 分解、正交变换与最小二乘求解功能，具体 API 包括 `aclsolverSgeqrf`、`aclsolverSorgqr`、`aclsolverSormqr`、`aclsolverXlarft`、`aclsolverSSgels`。接口参考 NVIDIA cuSolver DN 稠密 QR 接口（`geqrf`、`orgqr`、`ormqr`）、块反射器三角因子 `Xlarft`，以及单精度最小二乘 `SSgels`（任务书中记为 Xgels(S)）。要求接口、功能、参数、约束与对标 CUDA 接口对齐，精度达到 CANN 生态算子开源标准。完成设计、开发、测试全流程，验收通过后合入 https://gitcode.com/cann/ops-solver 。

本任务适配硬件为 **Ascend 950PR（Atlas 950）**。五个接口作为**同一社区任务**一并交付，不允许拆分验收。ops-solver 仓库当前无对应 QR / gels 原型，须全新实现。`aclsolverXlarft` 本任务仅验收 FLOAT32 组合（语义对齐 SLARFT）。

## 2. 核心开发要求

### 2.1 功能实现要求

对标基线如下：

| NPU 交付接口 | 对标 CUDA 接口 | 对标文档 |
|---|---|---|
| `aclsolverSgeqrf` / `aclsolverSgeqrf_bufferSize` | `cusolverDnSgeqrf` / `cusolverDnSgeqrf_bufferSize` | https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-legacy-api |
| `aclsolverSorgqr` / `aclsolverSorgqr_bufferSize` | `cusolverDnSorgqr` / `cusolverDnSorgqr_bufferSize` | https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-orgqr |
| `aclsolverSormqr` / `aclsolverSormqr_bufferSize` | `cusolverDnSormqr` / `cusolverDnSormqr_bufferSize` | https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-ormqr |
| `aclsolverXlarft` / `aclsolverXlarft_bufferSize` | `cusolverDnXlarft` / `cusolverDnXlarft_bufferSize` | https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxlarft |
| `aclsolverSSgels` / `aclsolverSSgels_bufferSize` | `cusolverDnSSgels` / `cusolverDnSSgels_bufferSize` | https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-gels |

#### 数学公式

1. **aclsolverSgeqrf（QR 分解）**

```
A = Q · R
```

其中 A 为 m × n 单精度实矩阵。分解以 Householder 反射器形式原地写入 A：上三角（或梯形）为 R，严格下三角为反射器向量；`TAU` 长度为 min(m,n)，存放各反射器系数。语义对齐 `cusolverDnSgeqrf`。

2. **aclsolverSorgqr（由 Householder 生成显式 Q）**

输入为 `aclsolverSgeqrf` 输出的 A 与 `tau`，生成前 k 列正交矩阵 Q（m × n，且 m ≥ n ≥ k ≥ 0）。结果原地覆盖 A。语义对齐 `cusolverDnSorgqr`。

3. **aclsolverSormqr（用 Q 左乘或右乘）**

```
C := op(Q) · C     （side = LEFT）
C := C · op(Q)     （side = RIGHT）
```

其中 `op(Q)` 由 `trans` 指定：`N` 为不转置，`T` / `C` 对实数均为转置。A、`tau` 为 `aclsolverSgeqrf` 输出。C 原地更新。语义对齐 `cusolverDnSormqr`。

4. **aclsolverXlarft（块反射器三角因子 T）**

由存储在 V 中的 k 个初等反射器及 `tau` 生成三角因子 T（k × k）：

```
H = I - V · T · V^T
```

`direct = FORWARD` 时 T 为上三角；`direct = BACKWARD` 时 T 为下三角。本任务仅支持 `storev = COLUMNWISE`，且 n ≥ k。本任务仅验收 dataTypeV / dataTypeTau / dataTypeT / computeType 全为 FLOAT32 的组合（对齐 CUDA_R_32F，即 SLARFT）。其他 dtype 组合返回 `NOT_SUPPORTED`。

5. **aclsolverSSgels（单精度最小二乘，Xgels(S)）**

求解超定或方阵最小二乘问题（须 n ≤ m）：

```
min || A · X - B ||_F
```

其中 A 为 m × n，B 为 m × nrhs，解 X 为 n × nrhs，**out-of-place** 写入独立缓冲区。语义对齐 `cusolverDnSSgels`（主精度与内部精度均为 FLOAT32）。`niter` 报告迭代次数（可为 0 表示直接主精度分解求解）；`dinfo` 为 Device 信息。

#### 算法说明

1. 数据布局必须为 **列主序（column-major）**。leading dimension 须满足对标约束，禁止要求 `lda == m` 或 `lda == n`。
2. 矩阵、tau、T、info、workspace 均为 **Device 指针**（`Xlarft` 的 Host workspace 除外，对齐 cuSolver）；Host 侧仅传入标量维数、枚举、handle 和 params。
3. `geqrf` / `orgqr` / `ormqr` / `Xlarft` / `SSgels` 均须提供 `_bufferSize`。`geqrf`/`orgqr`/`ormqr` 计算接口带 `Lwork`（元素个数）；`Xlarft` / `SSgels` 的 workspace 以 **字节** 计，对齐 cuSolver。
4. `devInfo` / `dinfo` 必须真实写入：`0` 成功；`-i` 表示第 i 个参数非法（不计 handle）。
5. 须复用已有 handle 管理接口：`aclsolverCreate`、`aclsolverDestroy`、`aclsolverSetStream`、`aclsolverGetStream`。计算走调用方 stream，禁止无必要的 Host 同步。
6. 核心计算必须在 NPU AI Core 上完成，不允许 CPU fallback 代替 NPU 实现。
7. **确定性计算要求**：相同输入、相同 stream 串行多次执行，输出（含 `tau` / `info`）bit-wise 一致。Householder 符号约定须确定（例如对角元取使 R 对角为正或对齐 LAPACK 的确定规则）。

### 2.2 算子工程模式

使用 **ops-solver Host C API + AscendC/CATLASS Kernel 直调** 工程模式（非 aclnn 两段式、非 PyTorch 接口）。

- 公开头文件：https://gitcode.com/cann/ops-solver/blob/master/include/cann_ops_solver.h 、https://gitcode.com/cann/ops-solver/blob/master/include/cann_ops_solver_common.h
- Host：参数校验、workspace 计算、kernel 下发
- Kernel：基于 AscendC/CATLASS 实现 GEQRF / ORGQR / ORMQR / LARFT / GELS
- 返回值：计算接口统一返回 `aclsolverStatus_t`，状态码语义对齐 cuSolver。若需短暂兼容现存 `aclError` 原型，须在头文件与文档中给出对照表，且最终验收以 `aclsolverStatus_t` 公开接口为准

### 2.3 接口定义

须在 `cann_ops_solver_common.h` 中补充：

```
typedef void* aclsolverParams_t;

typedef enum {
    ACLSOLVER_OP_N = 0,
    ACLSOLVER_OP_T = 1,
    ACLSOLVER_OP_C = 2
} aclsolverOperation_t;

typedef enum {
    ACLSOLVER_SIDE_LEFT  = 0,  /* 对齐 CUBLAS_SIDE_LEFT */
    ACLSOLVER_SIDE_RIGHT = 1   /* 对齐 CUBLAS_SIDE_RIGHT */
} aclsolverSideMode_t;

typedef enum {
    ACLSOLVER_DIRECT_FORWARD  = 0,  /* 对齐 CUBLAS_DIRECT_FORWARD */
    ACLSOLVER_DIRECT_BACKWARD = 1   /* 对齐 CUBLAS_DIRECT_BACKWARD */
} aclsolverDirectMode_t;

typedef enum {
    ACLSOLVER_STOREV_COLUMNWISE = 0  /* 对齐 CUBLAS_STOREV_COLUMNWISE；本任务不要求 ROWWISE */
} aclsolverStorevMode_t;

typedef enum {
    ACLSOLVER_R_32F = 0,  /* 对齐 CUDA_R_32F */
    ACLSOLVER_C_32F = 1   /* 对齐 CUDA_C_32F；本任务不验收 */
} aclsolverDataType_t;
```

`aclsolverParams_t` 允许传入 `nullptr` 表示默认参数。若实现需要显式 params 对象，须同时交付 `aclsolverCreateParams` / `aclsolverDestroyParams`，语义对齐 `cusolverDnCreateParams` / `cusolverDnDestroyParams`。

公开计算接口如下。

```
aclsolverStatus_t aclsolverSgeqrf_bufferSize(
    aclsolverHandle_t handle, int m, int n, float *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverSgeqrf(
    aclsolverHandle_t handle, int m, int n, float *A, int lda,
    float *TAU, float *Workspace, int Lwork, int *devInfo);

aclsolverStatus_t aclsolverSorgqr_bufferSize(
    aclsolverHandle_t handle, int m, int n, int k,
    const float *A, int lda, const float *tau, int *lwork);

aclsolverStatus_t aclsolverSorgqr(
    aclsolverHandle_t handle, int m, int n, int k,
    float *A, int lda, const float *tau, float *work, int lwork, int *devInfo);

aclsolverStatus_t aclsolverSormqr_bufferSize(
    aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
    int m, int n, int k, const float *A, int lda, const float *tau,
    const float *C, int ldc, int *lwork);

aclsolverStatus_t aclsolverSormqr(
    aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
    int m, int n, int k, const float *A, int lda, const float *tau,
    float *C, int ldc, float *work, int lwork, int *devInfo);

aclsolverStatus_t aclsolverXlarft_bufferSize(
    aclsolverHandle_t handle, aclsolverParams_t params,
    aclsolverDirectMode_t direct, aclsolverStorevMode_t storev,
    int64_t n, int64_t k,
    aclsolverDataType_t dataTypeV, const void *V, int64_t ldv,
    aclsolverDataType_t dataTypeTau, const void *tau,
    aclsolverDataType_t dataTypeT, void *T, int64_t ldt,
    aclsolverDataType_t computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost);

aclsolverStatus_t aclsolverXlarft(
    aclsolverHandle_t handle, aclsolverParams_t params,
    aclsolverDirectMode_t direct, aclsolverStorevMode_t storev,
    int64_t n, int64_t k,
    aclsolverDataType_t dataTypeV, const void *V, int64_t ldv,
    aclsolverDataType_t dataTypeTau, const void *tau,
    aclsolverDataType_t dataTypeT, void *T, int64_t ldt,
    aclsolverDataType_t computeType,
    void *bufferOnDevice, size_t workspaceInBytesOnDevice,
    void *bufferOnHost, size_t workspaceInBytesOnHost);

aclsolverStatus_t aclsolverSSgels_bufferSize(
    aclsolverHandle_t handle, int m, int n, int nrhs,
    float *dA, int ldda, float *dB, int lddb, float *dX, int lddx,
    void *dwork, size_t *lwork_bytes);

aclsolverStatus_t aclsolverSSgels(
    aclsolverHandle_t handle, int m, int n, int nrhs,
    float *dA, int ldda, float *dB, int lddb, float *dX, int lddx,
    void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo);
```

参数名、顺序、含义、Device/Host 归属不得擅自增删或调换。`geqrf`/`orgqr`/`ormqr`/`SSgels` 维数类型为 `int`；`Xlarft` 维数为 `int64_t`，对齐 cuSolver 64-bit API。

### 2.4 参数说明

非法类型 / shape / dtype / 排布 / 值域须返回 `ACLSOLVER_STATUS_INVALID_VALUE`，并在 info 可写时写入 `-i`。空问题（m=0 / n=0 / k=0 / nrhs=0）成功返回。

#### 2.4.1 aclsolverSgeqrf / aclsolverSgeqrf_bufferSize

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文 | scalar | 句柄 | - | - | 非空 | 空句柄报错 |
| m | 输入 | 矩阵 A 的行数 | scalar | int | - | - | m >= 0 | m < 0 报错 |
| n | 输入 | 矩阵 A 的列数 | scalar | int | - | - | n >= 0 | n < 0 报错 |
| A | 输入/输出（原地） | 输入 A，输出 Householder 形式的 QR | tensor | FLOAT32 | 列主序 ND | lda × n，有效区域 m × n | 有限浮点 | 空指针且 m,n>0 时报错；lda < max(1,m) 报错 |
| lda | 输入 | A 的 leading dimension | scalar | int | - | - | lda >= max(1,m) | 不满足报错 |
| TAU | 输出（仅 geqrf） | Householder 系数 | tensor | FLOAT32 | ND | [min(m,n)] | 有限浮点 | 空指针且 min(m,n)>0 时报错 |
| Lwork / Workspace | 见 2.3 | workspace 元素个数 / Device 缓冲 | scalar / tensor | int* / FLOAT32 | - | [Lwork] | Lwork >= 0 | 指针非法报错 |
| devInfo | 输出 | 分解信息 | tensor | INT32 | ND | [1] | 0 / -i | 空指针报错 |

#### 2.4.2 aclsolverSorgqr / aclsolverSorgqr_bufferSize

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文 | scalar | 句柄 | - | - | 非空 | 空句柄报错 |
| m | 输入 | Q 的行数 | scalar | int | - | - | m >= 0 | m < 0 报错 |
| n | 输入 | Q 的列数 | scalar | int | - | - | 0 <= n <= m | n > m 或 n < 0 报错 |
| k | 输入 | 使用的反射器个数 | scalar | int | - | - | 0 <= k <= n | 不满足报错 |
| A | 输入/输出（原地） | geqrf 输出；返回显式 Q | tensor | FLOAT32 | 列主序 ND | lda × n | 有限浮点 | 空指针且 m,n>0 时报错；lda < max(1,m) 报错 |
| tau | 输入 | geqrf 输出的系数，长度 k | tensor | FLOAT32 | ND | [k] | 有限浮点 | k>0 时为空指针报错 |
| work / lwork | 见 2.3 | Device workspace | tensor / scalar | FLOAT32 / int | - | [lwork] | lwork >= 查询值 | 不足报错 |
| devInfo | 输出 | 信息 | tensor | INT32 | ND | [1] | 0 / -i | 空指针报错 |

#### 2.4.3 aclsolverSormqr / aclsolverSormqr_bufferSize

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文 | scalar | 句柄 | - | - | 非空 | 空句柄报错 |
| side | 输入 | LEFT / RIGHT | attr | aclsolverSideMode_t | - | - | LEFT、RIGHT | 非法枚举报错 |
| trans | 输入 | op(Q)：N / T / C | attr | aclsolverOperation_t | - | - | N、T、C | 非法枚举报错 |
| m, n | 输入 | 矩阵 C 的行数、列数 | scalar | int | - | - | m,n >= 0 | 负值报错 |
| k | 输入 | 反射器个数 | scalar | int | - | - | 0 <= k <= nq，nq = (side==LEFT ? m : n) | 不满足报错 |
| A | 输入 | geqrf 输出的反射器 | tensor | FLOAT32 | 列主序 ND | lda × k | 有限浮点 | lda < max(1, nq) 报错 |
| tau | 输入 | 系数，长度 k | tensor | FLOAT32 | ND | [k] | 有限浮点 | k>0 时为空报错 |
| C | 输入/输出（原地） | 被 Q 作用的矩阵 | tensor | FLOAT32 | 列主序 ND | ldc × n | 有限浮点 | ldc < max(1,m) 报错 |
| work / lwork / devInfo | 见 2.3 | workspace 与 info | - | - | - | - | - | 同 geqrf |

#### 2.4.4 aclsolverXlarft / aclsolverXlarft_bufferSize

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文 | scalar | 句柄 | - | - | 非空 | 空句柄报错 |
| params | 输入 | 允许 nullptr | scalar | 句柄 | - | - | nullptr 或合法 params | 非法指针报错 |
| direct | 输入 | FORWARD / BACKWARD | attr | aclsolverDirectMode_t | - | - | FORWARD、BACKWARD | 非法枚举报错 |
| storev | 输入 | 仅 COLUMNWISE | attr | aclsolverStorevMode_t | - | - | COLUMNWISE | 其他值报错 |
| n | 输入 | 块反射器阶数 | scalar | int64_t | - | - | n >= k >= 1；允许 n=0 空问题 | n < k 报错 |
| k | 输入 | 反射器个数 / T 的阶 | scalar | int64_t | - | - | k >= 0 | k < 0 报错 |
| dataType* | 输入 | 本任务仅 FLOAT32 四元组 | attr | aclsolverDataType_t | - | - | 全为 ACLSOLVER_R_32F | 其他组合返回 NOT_SUPPORTED |
| V | 输入 | 列存反射器，ldv × k，有效 n × k | tensor | FLOAT32 | 列主序 ND | ldv × k | 有限浮点 | ldv < max(1,n) 报错 |
| tau | 输入 | 长度 k | tensor | FLOAT32 | ND | [k] | 有限浮点 | k>0 时为空报错 |
| T | 输出 | k × k 三角因子 | tensor | FLOAT32 | 列主序 ND | ldt × k | 有限浮点 | ldt < max(1,k) 报错 |
| bufferOnDevice / bufferOnHost | 输入 | Device / Host workspace | buffer | void | - | 查询字节数 | - | 长度不足报错 |

#### 2.4.5 aclsolverSSgels / aclsolverSSgels_bufferSize

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| handle | 输入 | aclsolver 上下文 | scalar | 句柄 | - | - | 非空 | 空句柄报错 |
| m | 输入 | A 的行数 | scalar | int | - | - | m >= n >= 0 | m < n 或负值报错 |
| n | 输入 | A 的列数 | scalar | int | - | - | n >= 0 | n < 0 报错 |
| nrhs | 输入 | 右端项列数 | scalar | int | - | - | nrhs >= 0 | nrhs < 0 报错 |
| dA | 输入/输出 | m × n 矩阵；求解后内容可被破坏 | tensor | FLOAT32 | 列主序 ND | ldda × n | 有限浮点 | ldda < max(1,m) 报错 |
| dB | 输入 | m × nrhs 右端项 | tensor | FLOAT32 | 列主序 ND | lddb × nrhs | 有限浮点 | lddb < max(1,m) 报错 |
| dX | 输出 | n × nrhs 解 | tensor | FLOAT32 | 列主序 ND | lddx × nrhs | 有限浮点 | lddx < max(1,n) 报错 |
| dWorkspace / lwork_bytes | 见 2.3 | Device workspace（字节） | buffer / size_t | void | - | [lwork_bytes] | >= 查询值 | 不足报错 |
| niter | 输出 | 迭代次数，Host 标量 | scalar | int* | - | [1] | >= 0 | 空指针报错 |
| dinfo | 输出 | Device 信息 | tensor | INT32 | ND | [1] | 0 / -i | 空指针报错 |

**规模与泛化（验收下限，允许实现支持更大范围并在文档中声明）**：

| 接口 | 维数 | 其他 |
|---|---|---|
| aclsolverSgeqrf | m, n ∈ [0, 4096]，须覆盖方阵与矩形 | lda padding |
| aclsolverSorgqr | m ∈ [0, 4096]，n,k ≤ m | 覆盖 k=n 与 k<n |
| aclsolverSormqr | m,n ∈ [0, 4096]，k 合法 | side 覆盖 LEFT/RIGHT；trans 覆盖 N/T/C |
| aclsolverXlarft | n ∈ [1, 4096]，k ∈ [1, min(n, 256)] | 仅 FLOAT32；仅 COLUMNWISE；FORWARD 与 BACKWARD |
| aclsolverSSgels | m ∈ [n, 4096]，n ∈ [0, 4096]，nrhs ∈ [0, 128] | 须 n ≤ m；m=n 与 m>n |

不支持 broadcast，不要求图融合。dynamic shape：Host 按本次调用维数生成 tiling。

## 3. 验收标准

### 3.1 软硬件环境要求

- **适配硬件**：Ascend 950PR（Atlas 950）
- **CANN 版本**：CANN 9.0.0 及以上，须与 https://gitcode.com/cann/ops-solver 仓库 README 已验证配套版本一致
- **对标软件（性能金标采集）**：CUDA Toolkit 中的 cuSolver，GPU 为 NVIDIA A100
- **其他依赖**：ops-solver 仓编译所需 ACL / AscendC / CATLASS 工具链；精度 golden 可使用 NumPy / SciPy LAPACK（float64）或 CPU 参考实现
- **版本记录**：自测报告须记录 CANN 版本、NPU 具体型号（950PR）、驱动版本、CUDA / cuSolver 版本

功能和精度、性能均须提交 950PR 测试结果。

### 3.2 精度要求

1. 对标接口见 2.1 节。数值正确性以更高精度 CPU 参考为单标杆：FLOAT32 计算的 golden 使用 FLOAT64（NumPy/SciPy `float64` 的 `qr` / `ormqr` 等价操作 / `lstsq` / `larft`，或等价 LAPACK `dgeqrf`/`dorgqr`/`dormqr`/`dlarft`/`dgels`）。
2. NPU 输出须满足生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
   - geqrf：由 Householder 还原 Q、R 后比较 A 与 Q R，或比较 R 与 golden 的上三角
   - orgqr：QᵀQ ≈ I，且 A ≈ Q R（与 geqrf 联测）
   - ormqr：与显式构造 Q 再乘 C 的结果比较
   - Xlarft：由 T 还原 H 后与逐个反射器连乘比较
   - SSgels：比较 X 与 `lstsq` golden，以及残差范数
3. FLOAT32 混合容差如下：

    | 数据类型 | FLOAT32 |
    |----------|---------|
    | rtol | 2^-10 (9.77e-4) |
    | atol | 2^-16 (1.53e-5) |
    | required_matched_ratio | 0.99 |
    | max_abs_error_limit | 1e-2 or 32 * ULP |

    逐元素判定：|actual - golden| ≤ atol + rtol × |golden|。当用例同时满足 matched_ratio ≥ required_matched_ratio 且 max_abs_error ≤ max_abs_error_limit 时，判定该用例精度通过。

4. Householder 的符号/相位约定允许与 CUDA 不同，但必须满足还原后的 Q R（或作用在 C 上的结果）通过混合容差。
5. 覆盖普通值、小值、正负混合、高瘦/方阵/矮胖（geqrf 允许 m<n；SSgels 仅 m≥n）；INF/NAN 按精度标准文档对应规则验收。
6. 确定性：合法用例重复执行须 bit-wise 一致。

### 3.3 性能要求

1. 性能对标为 NVIDIA A100 上同 shape、同 dtype（FLOAT32）、同列主序、同等预热与计时口径下的 CUDA 接口：
   - aclsolverSgeqrf：`cusolverDnSgeqrf`
   - aclsolverSorgqr：`cusolverDnSorgqr`（geqrf 时间不计入）
   - aclsolverSormqr：`cusolverDnSormqr`（geqrf 时间不计入）
   - aclsolverXlarft：`cusolverDnXlarft`（FLOAT32 组合）
   - aclsolverSSgels：`cusolverDnSSgels`
2. 达标判据：对下表每一个 case，NPU kernel 耗时满足 T_NPU ≤ T_A100 / 0.8，即性能倍率 ≥ 0.8 倍 A100。须在 950PR 上达标。
3. 每个 case 至少预热 10 次、正式采样 30 次，报告中位数；每轮须 Device 同步后计时。不含首次编译、数据生成、H2D/D2H。workspace 在正式采样期间复用。
4. 开发者须在 A100 上实测填入 T_A100，并在报告中给出 CUDA 版本与采集脚本。下表 T_A100 列为待测占位。

| 编号 | 接口 | 规格 | GPU A100 性能 | NPU 耗时 | 目标 |
|---|---|---|---|---|---|
| P-01 | aclsolverSgeqrf | m=n=1024，lda=m | 待测 | 待测 | ≥ 0.8×A100 |
| P-02 | aclsolverSgeqrf | m=n=4096，lda=m | 待测 | 待测 | ≥ 0.8×A100 |
| P-03 | aclsolverSgeqrf | m=2048，n=1024，lda=m | 待测 | 待测 | ≥ 0.8×A100 |
| P-04 | aclsolverSorgqr | m=n=k=1024 | 待测 | 待测 | ≥ 0.8×A100 |
| P-05 | aclsolverSorgqr | m=n=k=4096 | 待测 | 待测 | ≥ 0.8×A100 |
| P-06 | aclsolverSormqr | m=n=2048，k=2048，side=LEFT，trans=N | 待测 | 待测 | ≥ 0.8×A100 |
| P-07 | aclsolverSormqr | m=n=2048，k=2048，side=LEFT，trans=T | 待测 | 待测 | ≥ 0.8×A100 |
| P-08 | aclsolverXlarft | n=4096，k=128，FORWARD，COLUMNWISE，FLOAT32 | 待测 | 待测 | ≥ 0.8×A100 |
| P-09 | aclsolverXlarft | n=2048，k=256，FORWARD，COLUMNWISE，FLOAT32 | 待测 | 待测 | ≥ 0.8×A100 |
| P-10 | aclsolverSSgels | m=n=1024，nrhs=1 | 待测 | 待测 | ≥ 0.8×A100 |
| P-11 | aclsolverSSgels | m=4096，n=1024，nrhs=8 | 待测 | 待测 | ≥ 0.8×A100 |
| P-12 | aclsolverSSgels | m=n=4096，nrhs=32 | 待测 | 待测 | ≥ 0.8×A100 |

全部 12 条在 950PR 上均须达标，未达标不予验收。

### 3.4 内存要求

对标 cuSolver 同等规格：

1. 各接口 Device workspace 不超过 `bufferSize` 返回值；禁止额外分配未计入查询结果的隐蔽大块 Device 内存，并在设计文档中给出公式。`Xlarft` 的 Host workspace 同样不得超过查询值。
2. 同规格下 NPU 额外 workspace 不得超过 A100 对应接口 workspace 的 2 倍。
3. 无内存泄漏：连续创建 handle、查询 workspace、执行、销毁，重复 100 次后 Device 已用内存回到基线。

### 3.5 自验要求

本任务使用 ops-solver 仓测试工程（https://gitcode.com/cann/ops-solver/tree/master/test ）及 AscendOpTest 工具（https://gitcode.com/HIT1920/AscendOpTest ）进行自验。请根据本任务给出的自测用例和测试指导完成自测，并输出自测报告。自测用例目录：./sgeqrf_family_950_testCase/

NPU 接口与 CUDA 接口参数序列一致（见 2.3）。差异仅在命名前缀与枚举类型名。

功能自验至少覆盖：

| 类别 | 必测场景 |
|---|---|
| 基础功能 | geqrf；geqrf+orgqr 还原 Q；geqrf+ormqr；Xlarft；SSgels |
| 矩形 | m>n、m=n、m<n（仅 geqrf）；SSgels 仅 m≥n |
| side / trans | ormqr 的 LEFT/RIGHT 与 N/T/C |
| Xlarft | FORWARD/BACKWARD；非法 ROWWISE 报错；非 FLOAT32 返回 NOT_SUPPORTED |
| SSgels | nrhs=1 与多右端；m=n 与 m>n；m<n 报错 |
| padding | lda/ldc/ldv/ldt > 最小合法值 |
| info | 成功 info=0；非法参数 info=-i |
| 空问题 | m=0 或 n=0 或 k=0 或 nrhs=0 成功返回 |
| 确定性 | 同一输入重复执行 bit-wise 一致 |
| 流 | 非默认 stream 下结果正确 |

测试用例入参生成规则：

| 参数名 | Tensor值域分布 | Attr 覆盖规则 |
|---|---|---|
| A | 50% 均匀 [-5,5]；50% 正态 μ∈[-5,5]、σ∈[0.1,2] | m、n 覆盖 2 的幂与 2 的幂-1 |
| C / B | 与 A 相同分布 | ormqr 的 C；gels 的 B |
| side / trans | - | LEFT/RIGHT；N/T/C |
| direct / storev | - | FORWARD/BACKWARD；仅 COLUMNWISE |
| lda 等 | - | 最小合法值及 +8 / +32 padding |
| nrhs | - | 1、8、32、128 及 0 |

## 4. 验收交付件

在社区任务IT系统中提交验收时， 需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|------|-----------|------------|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ；<br> 2. 在cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以PR形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md|
| 2 | 自测用例及测试代码 |1. 需要清晰列出精度测试case和性能测试case；<br> 2. 测试代码中的readme文件需要说明测试步骤，保证验收人可以复现测试结果|
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ；<br> 2. 需要包含用例参数、精度对比结果及截图、性能数据及截图 |
| 4 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录；需要在个人仓邀请账号Ascend-CANN作为开发者，如下图所示； <br> 2. 需要根据每个仓库的规范提供算子readme文档 <br> |

![邀请示意](./pics/invite.jpeg)

## 5. PR 申请合入

测试通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的接口合入 https://gitcode.com/cann/ops-solver 。建议目录：

```
include/cann_ops_solver.h
include/cann_ops_solver_common.h
src/sgeqrf/
src/sorgqr/
src/sormqr/
src/xlarft/
src/ssgels/
test/sgeqrf/
test/sorgqr/
test/sormqr/
test/xlarft/
test/ssgels/
docs/zh/sgeqrf.md
docs/zh/sorgqr.md
docs/zh/sormqr.md
docs/zh/xlarft.md
docs/zh/ssgels.md
docs/api_list.md
```

须同步更新 README 与接口列表。五个接口作为同一 PR 或一组关联 PR 合入，不得只合入其中部分接口。

## 6. 参考资料

1. NVIDIA cuSolver 文档（geqrf / orgqr / ormqr / Xlarft / SSgels）：https://docs.nvidia.com/cuda/cusolver/index.html ；
2. ops-solver 仓库：https://gitcode.com/cann/ops-solver ；
3. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ；
4. AscendOpTest：https://gitcode.com/HIT1920/AscendOpTest ；
5. Ascend C算子开发文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html ；
6. 算子开发接口文档：https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html ；
7. Ascend C在线课程：https://www.hiascend.com/developer/courses/detail/1691696509765107713 ；
8. CATLASS：https://gitcode.com/cann/catlass 。

## 7. 特别注意事项

1. 本任务五个接口必须全部交付：aclsolverSgeqrf、aclsolverSorgqr、aclsolverSormqr、aclsolverXlarft、aclsolverSSgels。缺一不可。
2. 文档中的 cuSolver 仅作为接口、功能、约束和性能标杆，NPU 实现统一使用 aclsolver + AscendC/CATLASS，禁止依赖 CUDA。
3. 必须列主序、Device 指针、真实写入 info、支持合法 lda padding。
4. `aclsolverXlarft` 本任务仅验收 FLOAT32 组合；`storev` 仅 COLUMNWISE；n ≥ k。
5. `aclsolverSSgels` 须 n ≤ m；解 X 为独立输出，不得要求 X 与 B 共用存储。任务书中的 Xgels(S) 即本接口。
6. `aclsolverSorgqr` / `aclsolverSormqr` / `aclsolverXlarft` 的输入依赖 `aclsolverSgeqrf` 的 Householder 表示，不是未分解的原始 A。
7. 性能门禁为 0.8 倍 A100，须在 950PR 上达标。
8. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请。
9. 开发前请务必阅读【社区任务】流程及注意事项：https://gitcode.com/org/cann/discussions/39 。

## 8. 环境获取（无需修改，使用模板原始内容）

 1. 使用 hidevlab webIDE 算力：https://hidevlab.huawei.com/online-develop-intro?from=hiascend 。
 - **【补充说明】填写示例：本人gitcode账号是 yolo，现在参与社区任务"7月社区任务-aclnnRoll算子开发"，需要申请A2/A3算力进行任务开发。**
 	 
 	![环境截图](./pics/zaixiankaifa1.png)  
 	![环境截图](./pics/apply.png)  
 	 
2. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。
 	 
 	![环境截图](./pics/yunkaifa.png)
 	 
3. 如需额外环境资源，请联系昇腾CANN小助手。
