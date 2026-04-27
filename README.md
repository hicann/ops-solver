# ops-solver

## 🚀概述
ops-solver 是基于昇腾NPU芯片的高级数值求解算子库，提供包括矩阵分解、矩阵求逆、特征值计算、特征向量计算、线性方程组求解等功能。  

## ⚡️快速入门
若您希望**从零到一快速体验**项目能力，请访问下述简易教程。

1. **[环境部署](#环境部署)**：介绍基础环境搭建，包括软件包和三方依赖的获取和安装、源码下载等。

   >  **说明**：本步骤是QuickStart和各类教程的操作前提，请先完成基础环境搭建。

2. **[QuickStart](docs/从开发一个简单算子出发.md)**：提供快速上手本项目能力的指南，包括算子开发流程等。

## 📂目录结构
ops-solver库的目录结构如下：

```
ops-solver
├── build                        //可存放构建生成的文件
├── docs                         //文档文件
├── example                      //算子调用示例代码，包含可直接运行的Demo
├── include                      //存放公共头文件
├── scripts                      //脚本文件存放目录
├── solver                       //主体源代码目录
│   ├── utils                    //公共函数
│   ├── cmatinv_batched          //单精度复数批量矩阵求逆
│   ├──  ...                     //其他算子实现
│   └── CMakeLists.txt
├── tests                        //测试代码
```

## 🔧环境部署

本节提供基础环境搭建说明，更多安装步骤请参考[详细安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/softwareinst/instg/instg_0001.htm)。

### 安装前准备

请先确保编译环境的基础库依赖已安装，注意满足版本号要求。

- python >= 3.7.0（建议版本 <= 3.10）
- gcc >= 7.3.0
- cmake >= 3.16.0
- pigz（可选，安装后可提升打包速度，建议版本 >= 2.4）
- dos2unix
- gawk
- make
- patch

### CANN软件安装
 **场景1：体验master版本能力或基于master版本进行开发**

   1. **安装驱动与固件（运行态依赖）**

      下载和安装操作请参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》中“准备软件包”和“安装NPU驱动和固件”章节。驱动与固件是运行态依赖，若仅编译算子，可以不安装。

   2. **安装CANN包**

      请单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)，选择最新时间版本，并根据产品型号和环境架构下载对应包。安装命令如下，更多指导参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》。

      **安装CANN toolkit包**

      ```bash
      # 确保安装包具有可执行权限
      chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
      # 安装命令
      ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
      ```

      **安装CANN ops包（运行态依赖）**

      ops包是运行态依赖，若仅编译算子，可不安装此包。

      ```bash
      # 确保安装包具有可执行权限
      chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
      # 安装命令
      ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
      ```

      - \$\{cann\_version\}：表示CANN包版本号。
      - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
      - \$\{soc\_name\}：表示NPU型号名称。
      - \$\{install\_path\}：表示指定安装路径，ops包需与toolkit包安装在相同路径，root用户默认安装在`/usr/local/Ascend`目录。

**场景2：体验已发布版本能力或基于已发布版本进行开发**

   - 请访问[CANN官网下载中心](https://www.hiascend.com/cann/download)，选择发布版本（仅支持CANN 8.5.0及后续版本），并根据产品型号和环境架构下载对应包，最后参考网页提供的命令完成安装。

### 环境变量配置

按需选择合适的命令使环境变量生效。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
source ${install_path}/cann/set_env.sh
```
### ⬇️ 源码下载
下载仓库，您可自行选择需要的分支。
```sh
git clone https://gitcode.com/cann/ops-solver.git
```
>  **快速体验项目标准流程**
```sh
cd ops-solver
bash build.sh --ops=cmatinv_batched --run  # --ops=<算子名> --run可选参数，执行测试样例
```

## 💬相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-linear-algebra)

## 🤝联系我们

本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/ops-solver/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/ops-solver/discussions)参与交流。
- **技术专栏**：通过GitCode[【Wiki】](https://gitcode.com/cann/ops-solver/wiki)获取技术文章，如系列化教程、优秀实践等。