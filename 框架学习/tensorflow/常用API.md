# 常用API

####  **tf.summary.scalar**

```python
tf.summary.scalar(tags, values, collections=None, name=None) 
```

用来显示标量信息
- - -

#### **tf.slice**

```python
tf.slice(inputs, begin, size, name)
```

从列表、数组、张量等对象中抽取一部分数据:

- begin和size是两个多维列表，他们共同决定了要抽取的数据的开始和结束位置
- begin表示从inputs的哪几个维度上的哪个元素开始抽取
- size表示在inputs的各个维度上抽取的元素个数
- 若begin[]或size[]中出现-1,表示抽取对应维度上的所有元素
  
- - -

#### **tf.cast**

```python
tf.cast(x, dtype, name=None)
```

数据类型转换
- - -

#### **tf.clip_by_global_norm**

```python
tf.clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)
```

让权重的更新限制在一个合适的范围。避免梯度消失或者梯度爆炸

- t_list 是梯度张量， clip_norm 是截取的比率
- 计算公式t_list[i] * clip_norm / max(global_norm, clip_norm)

- - -

#### **tf.layers.dense**

```python
tf.layers.dense  
```

tf的全连接层函数
- - -

#### **tf.group**

```python
tf.group( *inputs, **kwargs)
```

输入为op

- var = tf.group(input1， input2)  得到var时,input1和input2也能得计算完毕
      return: sess.run返回为None
- 和tf.tuple类似  tuple = tf.tuple([input1， input2])， input计算完毕，并且sess,run返回为input1和input2 tensor
  
- - -

#### **z = tf.identity**

```python
z = tf.identity(x, name="z_add")
```

返回一个shape和数值都一样的tensor

1. 与control_dependencies配套使用
2. 为没有名称参数的op分配name
3. tf.identity在计算图内部创建了两个节点，send / recv节点，用来发送和接受两个变量，如果两个变量在不同的设备上，比如 CPU 和 GPU，那么将会复制变量，如果在一个设备上，将会只是一个引用。
4. 当我们的输入数据以字节为单位进行序列化时，我们想要从该数据集中提取特征。 我们可以关键字格式执行此操作，然后为其获取占位符

- - -

#### **tf.assign**

```python
tf.assign(
    ref,
    value,
    validate_shape=None,
    use_locking=None,
    name=None
）
```

通过赋值来更新“ref”。这个函数将需要被更新的变量 ref 赋值为新值 value。

eg. ref.assign(value)
- - -

#### **tf.gradients**

```python
tf.gradients(ys, xs, 
    grad_ys=None, 
    name='gradients',
    colocate_gradients_with_ops=False,
    gate_gradients=False,
    aggregation_method=None,
    stop_gradients=None)
```

tf.gradients()接受求导值ys和xs不仅可以是tensor，还可以是list，形如[tensor1, tensor2, …, tensorn]。当ys和xs都是list时，它们的求导关系为
![example]
- - -

#### **tf.tile**

```python
tf.tile(a, [2, 2])
```

按照矩阵维度复制数据 ，表示把a的第一个维度复制两次，第二个维度复制2次。
- - -

#### **tf.norm**

```python
tf.norm(
    tensor,
    ord='euclidean',
    axis=None,
    keepdims=None,
    name=None,
    keep_dims=None
)  
```

作用：计算向量，矩阵的范数。
- - -

#### **tf.reduce_sum**

```python
tf.reduce_sum(
    input_tensor, 
    axis=None, 
    keepdims=None,
    name=None,
    reduction_indices=None, 
    keep_dims=None) 
```

计算张量tensor沿着某一维度的和，可以在求和后降维。

- 再多的维只不过是是把上一个维度当作自己的元素
- 1维的元素是标量（0维），2维的元素是数组（1维），3维的元素是矩阵(2维),如2*3*4的tensor，调用tf.reduct_sum, 当axis =0, 结果维度为3*4,当axis =1, 结果维度为2*4,当axis =3, 结果维度为2*3

- - -

#### **tf.multiply**

```python
tf.multiply 
```

两个矩阵中对应元素各自相乘,笛卡尔积为A,B集合的所有元素组合的排列，比如A,B元素个数分别为3，5且元素不同，此时笛卡尔积集合个数为3*5， a1为例，(a1,b1) ... (a1, b5) 
- - -

#### **tf.matmul**

```python
tf.matmul
```

将矩阵a乘以矩阵b，生成a * b。
- - -

#### **tf.local_variables**

```python
tf.local_variables
```

返回 TensorFlow 的局部变量.

局部变量 - 每个进程变量,通常不保存或还原到检查点,并用于临时值或中间值
- - -

#### **tf.add_n**

```python
tf.add_n([p1, p2, p3....])
```

函数是实现一个列表的元素的相加。就是输入的对象是一个列表，列表里的元素可以是向量，矩阵，等,p1,p2,p3维度一致
- - -

#### **tf.negative**

```python
tf.negative(x, name=None)
```

取负(y = -x).
- - -

#### **tf.Variable**

```python
tf.Variable()
```

创建变量，初始化是直接传入initial_value，如果检测到命名冲突，系统会自己处理
- - -

#### **tf.get_variable**

```python
tf.get_variable
```

创建变量，初始化是传入一个initializer，如果检测到命名冲突，系统会不会处理，而是报错
- - -

#### **tf.feed_dict**

```python
tf.feed_dict 
```

feed的字典中的value应该为tensor或者numpy数组
- - -

#### **SparseTensor**

在TensorFlow中，SparseTensor对象表示稀疏矩阵。SparseTensor对象通过3个稠密矩阵indices, values及dense_shape来表示稀疏矩阵，这三个稠密矩阵的含义介绍如下：

1. indices：数据类型为int64的二维Tensor对象，它的Shape为[N, ndims]。indices保存的是非零值的索引，即稀疏矩阵中除了indices保存的位置之外，其他位置均为0。
2. values：一维Tensor对象，其Shape为[N]。它对应的是稀疏矩阵中indices索引位置中的值。
3. dense_shape：数据类型为int64的一维Tensor对象，其维度为[ndims]，用于指定当前稀疏矩阵对应的Shape。

```python
SparseTensor(indices=[[0, 0],[1, 1], [1,2] ], value=[1, 6,2], dense_shape=[3, 3])

[[1. 0. 0.]
 [0. 6. 2.]
 [0. 0. 0.]]
```

- - -

#### **tf.sparse.fill_empty_rows**

```python
tf.sparse.fill_empty_rows(
    sp_input, default_value, name=None
)  
```

将稀疏矩阵没有值的行，填充默认值

- - -

#### **tf.strided_slice**

```python
tf.strided_slice(
    input_, begin, end, strides=None, begin_mask=0, end_mask=0, ellipsis_mask=0,
    new_axis_mask=0, shrink_axis_mask=0, var=None, name=None
) 
```

将输入每个维度，按照[begien,end, stide)     进行切分得到新的tensor

- - -

#### **tf.gather**

```python
tf.gather(
    params, indices, validate_indices=None, name=None, axis=None, batch_dims=0
)
```

收集切片,指定维度收集切片

- - -

#### **tf.one_hot**

```python
 def one_hot(indices,
            depth,
            on_value=None,
            off_value=None,
            axis=None,
            dtype=None,
            name=None):
  """Returns a one-hot tensor.
```

对对应索引位置的值设置为on_value，其余位置为off_value， -1也是off_value

- - -

#### **tf.reshape**

```python
tf.reshape(
    tensor, shape, name=None
)
```

shape为-1，将tensor转换为1维的tensor,类似数组
- - -

### **tf.data.Dataset**
```python

# 将转换函数应用于此数据集, 
# apply 启用自定义数据集转换的链接，这些转换表示为采用一个数据集参数并返回转换后的数据集的函数
apply(
    transformation_func
)

# 将此数据集的连续元素组合成批次
# 结果元素的组件将有一个额外的外部维度，它将是 batch_size（如果 batch_size 没有将输入元素的数量 N 平均划分并且 drop_remainder 为 False，则最后一个元素的 N % batch_size）。 如果您的程序依赖于具有相同外部尺寸的批次，则应将 drop_remainder 参数设置为 True 以防止生成较小的批次
batch(
    batch_size, drop_remainder=False
)

# 缓存此数据集中的元素
cache(
    filename=''
)

# 通过将给定数据集与此数据集连接来创建数据集
concatenate(
    dataset
)

# 枚举此数据集的元素。类似于python的enumerate,enumerate多用于在for循环中得到计数，利用它可以同时获得索引和值
enumerate(
    start=0
)

# 根据谓词过滤此数据集
filter(
    predicate
)

# 在这个数据集中映射 map_func，并交错结果
interleave(
    map_func, cycle_length=AUTOTUNE, block_length=1, num_parallel_calls=None
)

# 匹配一个或多个 glob 模式的所有文件的数据集
@staticmethod
list_files(
    file_pattern, shuffle=None, seed=None
)

# 创建一个迭代器以枚举此数据集的元素
make_initializable_iterator(
    shared_name=None
)

# 创建一个迭代器以枚举此数据集的元素
make_one_shot_iterator()

# 跨该数据集的元素映射 map_func
# 此转换将 map_func 应用于此数据集的每个元素，并返回包含转换后元素的新数据集，顺序与它们在输入中出现的顺序相同
map(
    map_func, num_parallel_calls=None
)
# -------------------------------example-------------------------------------
sequence = np.array([[1, 3], [2, 3], [3, 4]])
def generator():
    for el in sequence:
        yield el

dataset = tf.data.Dataset.from_generator(generator,
                                         output_types=(tf.float32),
                                         output_shapes=(tf.TensorShape([2])))

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    print(sess.run(next_element))
```
数据集可用于将输入管道表示为元素的集合和作用于这些元素的转换的“逻辑计划”
--- 

### **tf.train.MonitoredTrainingSession**
```python
tf.train.MonitoredTrainingSession(
    master='', is_chief=True, checkpoint_dir=None, scaffold=None, hooks=None,
    chief_only_hooks=None, save_checkpoint_secs=USE_DEFAULT,
    save_summaries_steps=USE_DEFAULT, save_summaries_secs=USE_DEFAULT, config=None,
    stop_grace_period_secs=120, log_step_count_steps=100, max_wait_secs=7200,
    save_checkpoint_steps=USE_DEFAULT, summary_dir=None
)

tf.train.MonitoredSession(
    session_creator=None, hooks=None, stop_grace_period_secs=120
)

# MonitoredSession的method

close()

run(
    fetches, feed_dict=None, options=None, run_metadata=None
)

run_step_fn(
    step_fn
)

should_stop()


```
对于cheif，此实用程序设置适当的会话初始化器/恢复器。 它还创建与检查点和摘要保存相关的挂钩。 对于worker，此实用程序设置适当的会话创建者，等待主管初始化/恢复。 请查看 tf.compat.v1.train.MonitoredSession 了解更多信息

初始化：在创建时，受监视的会话按给定顺序执行以下操作：
- 为每个给定的钩子调用 hook.begin()
- 通过 scaffold.finalize() 完成图表
- 创建会话
- 通过脚手架提供的初始化操作初始化模型
- 如果检查点存在则恢复变量
- 启动队列运行器
- 调用 hook.after_create_session()

运行：调用 run() 时，受监视的会话会执行以下操作：
- 调用 hook.before_run()
- 使用合并的提取和 feed_dict 调用 TensorFlow session.run()
- 调用 hook.after_run()
- 返回用户询问的 session.run() 结果
- 如果发生 AbortedError 或 UnavailableError，它会在再次执行 run() 调用之前恢复或重新初始化会话

退出：在 close() 时，受监控的会话按顺序执行以下操作：
- 调用 hook.end()
- 关闭队列运行器和会话
- 如果将 monitored_session 用作上下文，则抑制表明所有输入均已处理的 OutOfRange 错误

注意：这不同于tf.compat.v1.Session。 例如，它不能执行以下操作
- 它不能设置为默认会话。
- 它不能发送到 saver.save。
- 它不能发送到 tf.train.start_queue_runners。

--- 

### **SessionRunHook**
```python
# 其method

#创建新的 TensorFlow 会话时调用
after_create_session(
    session, coord
)


# 在每次调用 run() 之后调用
after_run(
    run_context, run_values
)

# 在每次调用 run() 之前调用
# 您可以从此调用返回一个 SessionRunArgs 对象，指示要添加到即将到来的 run() 调用的操作或张量。 这些操作/张量将与最初传递给原始 run() 调用的操作/张量一起运行。 您返回的运行参数还可以包含要添加到 run() 调用的提要
# 此时图表已完成，您无法添加操作
before_run(
    run_context
)


# 在使用会话之前调用一次
# 调用时，默认图形是将在会话中启动的图形。 挂钩可以通过向其添加新操作来修改图形。 在 begin() 调用之后，图形将被最终确定，其他回调不能再修改图形。 在同一张图上第二次调用 begin() 不应更改图
begin()

# 在会话结束时调用
# 如果挂钩想要运行最终操作，例如保存最后一个检查点，则可以使用会话参数。
# 如果 session.run() 引发 OutOfRangeError 或 StopIteration 以外的异常，则不会调用 end()。 
# 当 session.run() 引发 OutOfRangeError 或 StopIteration 时，请注意 end() 和 after_run() 行为之间的区别。 在这种情况下，会调用 end() 但不会调用 after_run()
end(
    session
)

```

挂钩以扩展对 MonitoredSession.run() 的调用
--- 


### ****
```python

```

--- 
