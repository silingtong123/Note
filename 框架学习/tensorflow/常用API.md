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

两个矩阵中对应元素各自相乘
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

#### ****

```python
```

- - -
