# ops-solver

## 🚀概述
ops-solver 是基于昇腾NPU芯片的高级数值求解算子库，提供包括矩阵分解、矩阵求逆、特征值计算、特征向量计算、线性方程组求解等功能。  

## ⚡️快速入门
若您希望**从零到一快速体验**项目能力，请访问下述简易教程。

1. **[环境部署](docs/zh/install/quick_install.md)**：介绍基础环境搭建，包括软件包和三方依赖的获取和安装、源码下载等。

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
├── src                          //主体源代码目录
│   ├── utils                    //公共函数
│   ├── cmatinv_batched          //单精度复数批量矩阵求逆
│   ├──  ...                     //其他算子实现
│   └── CMakeLists.txt
├── tests                        //测试代码
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