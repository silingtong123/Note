#### 常见结构
TF会给每一个核心概念设计一个构建类


- NodeDef
  - NodeDef只有repeated string input = 3，格式形如node:src_output;没有output
  - 前端
  - 后端：NodeProperties 联系opDef, Nodedef
- OpDef


#### 如何查看一个op

- 查看前端
- 查看后端
  - 查看op注册：REGISTER_OP("Select")
  - 查看kernel注册：REGISTER_XLA_OP(Name("Select"), SelectOp);

#### 常见结构
- LocalExecutorParams
  - "params" 为excecutor提供了一组上下文。我们期望不同的上下文会提供不同的实现。
- Executor 
  - Executor运行一个图的计算