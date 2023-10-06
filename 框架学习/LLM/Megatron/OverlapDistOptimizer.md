# MagatronLLaMA

## _grad_hook

- 在没有使用SP的时候，需要将layerNorm进行allReduce

## hook机制
- register_hook 注册一个反向hook，每次计算Node的梯度时被调用
   - 签名：hook(grad_inputs: Tuple[Tensor], grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None
- register_forward_pre_hook 在forward开始执行前执行hook
   - 签名：hook(module, args, kwargs) -> None or a tuple of modified input and kwargs
- torch.nn.Module.register_forward_hook  在forward之后执行
   -签名：hook(module, args, kwargs, output) -> None or modified output
- register_full_backward_hook       计算得到梯度后调研hook
   - 签名：hook(module, grad_input, grad_output) -> tuple(Tensor) or None

### train.py

- train_step: 调用optimizer.reduce_model_grads，overlap优化器这个函数里啥也不做，只保留了接口内容为pass，梯度分发主要靠overlap里注册的hook实现，再调用optimizer.step更新参数，最后调用optimizer.gather_model_params收集更新后的参数


## grad

- main_grad：这是PyTorch中的一个特殊属性，它表示的是当前计算图中最主要的梯度。也就是说，如果你有多个损失函数，并且它们都使用了同一个变量进行计算，那么这个变量的main_grad属性将包含所有这些损失函数的梯度。
- grad：这个属性表示的是当前变量在所有输入上的梯度。如果有多个损失函数使用了同一个变量，那么它们的梯度会被合并到这个变量的grad属性中。

### OverlappedDistributedOptimizer

- 遍历optimizer.param_groups创建ParameterSchedule记录参数顺序， 根据记录的顺序创建ParameterBuffer， 根据记录的顺序创建BucketAssignment， 接着调param_buffer.init_partitioned_buffer初始化_partitioned_param（partation_param_buffer)，调用_allocate_other_buffers初始化_partitioned_grad(partation_grad_buffer), 元素个数和partation_param一致，保证切片的param和grad数据类型转换为fp32
- _partitioned_grad主要是用来做GA的，会把partitioned_grad_receiving_buffer的Add进去，partitioned_grad_receiving_buffer主要是scatter grad时的target
- _allocate_other_buffers: 初始化_partitioned_grad，并初始化_partitioned_grad_receiving_buffer开辟新空间，和_partitioned_grad和数据类型一致
- _register_hooks： 调用register_hook注册钩子到grad_acc，每个param会其hook函数会调用_grad_hook，当param对应的bucket已满时，会scatter 梯度
- _grad_hook： 会调用_all_reduce_layer_norm_grads和_collect_grad
- _all_reduce_layer_norm_grads：当没有使用SP时，layerNorm的grad会直接进行all_reducee;否则，这里什么也不做
- _collect_grad：从get_bucket_receiving_buffer获取梯度参数，再调用Bucket.reduce_scatter_grad scatter梯度, 回收对应的Bucket
- step: 调用原生optim的step，先调用copy_updated_parameters， 遍历_param_buffer时调用param_buffer.copy_updated_fp32_param， 接着遍历所有bucket调用gather_partitioned_param，将参数从partation_param gather到flatten_buffer中
- gather_model_params和reduce_model_grads里都是pass，只保留了接口
- Range记录起始和大小

- 问题：
   - optimizer.param_groups里面记录的什么？
   - ParameterSchedule是什么？如何记录顺序？详情见ParameterSchedule
   - ParameterBuffer他的作用是什么？详情见ParameterBuffer
   - BucketAssignment又是什么？ 详情见BucketAssignment
- 待优化：
   - gather参数时，能否保证更新一个bucket，就通信一个bucket（因为DP按照bucket分片，这样应该可以增加overlap）


### ParameterSchedule

- 跟踪param参数的顺序,

## ParameterBuffer

- 创建连续的内存，为了存放param和grad
- 构造函数调用_flatten_dense_tensors创建params_list所有参数大小的连续空间存到flatten_buffer，并调用_unflatten_dense_tensors获得tensors的视图，但内存空间不变
- init_partitioned_buffer：将每个bucket分片的起始和大小记录到Range中,接着调用_init_param_range，记录参数param在每个rank上分片后的的range,接着会申请所有bucket分片后在单个rank上总大小的连续内存，调用scatter_flatted_param_to_partitioned_param
- _init_param_range：当参数记录的起始和终点位置跨过了rank的切片，则会新起一个Range，替换原来param的value
- scatter_flatted_param_to_partitioned_param：将self._flatted_buffer中每个切片对应的内存空间的参数拷贝到self._partitioned_param
- copy_updated_fp32_param：将_partitioned_param更新为fp32
- gather_partitioned_param：

- 问题：
  - _flatten_dense_tensors是什么？通过torch中的C函数， 将制定tensors平铺在一维的连续空间中
  - _unflatten_dense_tensors是什么？ 通过torch中的C函数， 将_flatten_dense_tensors的视图转化为tensors，但是底层的数据存储没有改变

### BucketAssignment
- 每个parma顺序应该和获取grad的顺序一致
- 在构造函数中调_assign_buckets，
- _assign_buckets： 将paramlist放入bucket, 当前bucket满了后会申请下一个，其中调_create_bucket创建新的bucket
- _create_bucket：创建Bucket， 并保存param到bucket到映射，以及bucket到params的映射， 返回当前bucket的大小，更新max(return_val, largest_bucket_numel),并调Bucket.setup_grad_buffer_pool
- 问题：
  - Bucket是什么？Bucket.setup_grad_buffer_pool是在做什么？详情见Bucket


### Bucket

- self._total_size % self._num_partitions == 0， 要求能够被_num_partitions整除，为什么？ 按照_num_partitions的数目对bucket进行分片
- setup_grad_buffer_pool： 创建grad_buffer_pool = GradientBufferPool
- _assign_starting_position： 记录paramLists在bucket中的起始位置，创建param到起始位置的映射（map)
- collect_param_grad: 调用_init_grad_buffer，初始化_grad_buffer,当param.grad.dtype和bucket记录的dtype不一致时，会被强制转换，并将param.grad拷贝到grad_buffer中， _init_grad_buffer通过调grad_buffer_pool.get_buffer实现
- _init_grad_buffer：调grad_buffer_pool.get_buffer设置为self._borrowed_grad_buffer，调用get_real_buffer设置为self.grad_buffer
- reduce_scatter_grad: 将self.grad_buffer中的梯度scatter出去存入bucket_receiving_buffer，然后通过grad_buffer_pool.return_buffer回收self._borrowed_grad_buffer


### GradientBufferPool

- get_buffer： 通过调用_create_buffer创建GradientBuffer或者从self._buffer_pool.pop一个buffer使用
- return_buffer： 添加一个buffer到self._buffer_pool，一般时buffer通信完毕回收buffer，在下一次bucket申请时使用

### GradientBuffer
- 构造函数创建指定大小的empty tensor存放self._buffer， 该大小一般通过GradientBufferPool指定为largest_bucket_numel，除此之外他还指定dtype和device
- get_real_buffer：指定一定大小从self._buffer切分一部分使用， 通过tensor.narrow实现内存切分
- 问题：
   - Megatron-LM中：overlap使用一所有参数大小的内存切分使用，但是Megatron_LLaMA中使用一个Bucket指定bucket大小的内存块切分使用，且使用完毕后可以归还，重复使用
   - LM在DDP中：GradientBuffer拥有一批bucket, LLaMA中每个bucket拥有GradientBufferPool


