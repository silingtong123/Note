# [一文带你从零认识什么是XLA123](https://zhuanlan.zhihu.com/p/445994865)

- CPU和内存的交互方式
  - 内存分为DRAM, SRAM, 前者为主存，后者为L1,L2，L3高速缓存
  - 一个CPU核心有一块自己的L1，一块？L2,但不在CPU核心，所以比L1慢，L3是多个CPU共有 
  - 读取内存数据到cache -->  CPU读取cache/寄存器  -->  CPU的计算  -->  将结果写入cache/寄存器  -->  写回数据到内存
- 内存墙：该问题主要是访存瓶颈引起。算子融合主要通过对计算图上存在数据依赖的“生产者-消费者”算子进行融合，从而提升中间Tensor数据的访存局部性，以此来解决内存墙问题，主要通过手工方式实现固定Pattern的Buffer融合。代表：手工融合，XLA， TVM
  - 手工融合：无法泛化
  - XLA的全称是Accelerated Linear Algebra，即加速线性代数，XLA也是基于LLVM框架开发的
  - LLVM主要组成:
    - 前端：将前端语言转换为LLVM-IR
    - 优化器：负责优化LLVM-IR
    - 后端： 生成可执行机器码的模块
  - TVM：主要用于推理，能够支持自动对算子tuning,
- 并行墙：该问题主要是由于芯片多核增加与单算子多核并行度不匹配引起
