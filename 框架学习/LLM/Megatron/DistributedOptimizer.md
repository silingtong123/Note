# MegatronLM

## DistributedOptimizer

### DistributedOptimizer.\__init__

- 创建params buffer，和grad_buffer共用一份内存，但是有自己的视图和数据类型
- 更新优化器group, 并重新构建优化器状态
- 问题：
  - grad_buffer是什么？grad_buff是在DistributeDateParaller，详情见DistributeDateParaller

### reduce_model_grads

- 在train.py 中被调用， 用于reduce_scatter 梯度
- reduce顺序为：allReuce layer-norm, allReduce emb_grad, 再对all grad进行redcue-scatter
- DDP的grad_buffer主要用于reduce_scatter，所以这里没有动态分配张量  
- 问题：
  - 最后all grad包含前两步的吗？

## DistributeDateParaller

### DistributeDateParaller.\__init__

- overlap_grad_reduce 为False时，bucket_size 为无穷大，此时只有一个bucket，bucket_size默认4M
- 根据梯度的数据类型将params分组，记录每个param的数据类型和元素个数，将同一梯度数据类型对应的参数个数累加记录
- 根据上面记录的信息，申请grad_buffer,此时申请的grad_buffer的内存空间是完整的grad，而不是分片的
- 参数以相反的顺序放置在相应的grad_buffer中（所以param和grad共用了内存）
- grad_buffer_param_index_map 记录了dtype对应的每个参数在buffer中的起点和终点
- 注册了反向的hook：主要是将grad添加到main_grad（像是做梯度累计）
- 问题：
  - grad_buffer是什么？ 是GradBuffer对象，详情见GradBuffer

### GradBuffer

- 在构造函数中传入了numel和numel_padded，前者为真实个数，后者是pad后的大小，有一定的冗余，在父类MemoryBuffer，使用后者初始化一个zero tensor，并存入self.data
- 提供get函数，将buffer一部分切分出来给param使用
- 反向遍历params使其和反向的顺序大致一致，并对每个param.main_grad使用get函数申请内存使用， 当bucket满了之后我们会创建新的bucket
- set_bucket_: 也调用了get函数，但是他和前面param.main_grad的内存空间是同一个，这里只是使用了一个一维的视图，前者为真实的shape
- 问题： bucket是什么？ bucket是Bucket， 详情见Bucket

### Bucket

- bucket以异步的方式all reducee一组参数,当bucket中所有元素都有grad时就会调用self.all_reduce
- Bucket.all_reduce: 当overlap_grad_reduce为false时， 直接all reduce，反之则进行异步all reduce

## train.py

### train_step

- reduce_model_grads: 将梯度scatter到每个rank上面
- gather_model_params: 当自己拥有的部分参数更新完毕后，收集所有rank更新后的，所有rank得到完整的更新后的参数
- 更新学习率
- 调用顺序：pretrain - > train -> train_step, 所使用的优化器为DistributedOptimizer（一般会通过参数指定）
- 问题：
  - 这个优化器的更新方式是zero2吗？（我认为是的）

### get_model

- wrap_with_ddp默认为True，即获取模型时一般都会调用DDP(DistributeDateParaller)
- 问题：
  - DistributeDateParaller有什么特点？详情见DistributeDateParaller