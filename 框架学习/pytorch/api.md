# 常用API

### **torch.save**
- 可以保存模型结构和所有参数，此时加载模型比较方便，不不需要初始化模型结构；但是缺乏灵活性，如果想加载部分参数，需要先把保存的模型加载进来，再进行额外解析
- 也可以只保存模型参数，此时加载模型时需要先初始化模型结构
```py
torch.save(model, 'new_model.pth') #第一种
model = torch.load('my_model.pth')

model = MyModel()  # 第二种
torch.save(model.state_dict(), 'model_state_dict1.pth')
#state_dict = torch.load('model_state_dict.pth') 不确定是否需要这个
model.load_state_dict(state_dict)
```

### **model.load_state_dict**
- 该函数就是用于将预训练的参数权重加载到新的模型之中
- 当strict=True,要求预训练权重层数的键值与新构建的模型中的权重层数名称完全吻合；如果新构建的模型在层数上进行了部分微调，则上述代码就会报错：说key对应不上
- 当strict=false, 与训练权重中与新构建网络中匹配层的键值就进行使用，没有的就默认初始化
- 该函数通过调用每个子模块的_load_from_state_dict 函数来加载他们所需的权重，也说明了每个模块可以自行定义他们的 _load_from_state_dict 函数来满足特殊需求

### **torch.tensor([1.], requires_grad=True)**

- 需要设置tensor的requires_grad属性为True，才会进行梯度反传
### **torch.cuda.is_available()**
- torch.cuda.is_available() 检测GPU设备
- python -c 'import torch;print(torch.__version__);print(torch.version.cuda)' 检测cuda版本
### **torch.autograd.backward**

- gradient: 形状与tensor一致，可以理解为链式求导的中间结果，若tensor标量，可以省略（默认为1）
- retain_graph: 多次反向传播时梯度累加。反向传播的中间缓存会被清空，为进行多次反向传播需指定retain_graph=True来保存这些缓存
- create_graph: 为反向传播的过程同样建立计算图，可用于计算二阶导


### **tensor.is_leaf**

- 判断是否为叶子节点；

### **tensor.grad_fn**

- 每个变量的grad_fn指向产生其算子的backward function，叶节点的grad_fn为空

### **tensor.grad_fn.next_functions**

- 如果F = D + E，F.grad_fn.next_functions也存在两项，分别对应于D, E两个变量，每个元组中的第一项对应于相应变量的grad_fn，第二项指示相应变量是产生其op的第几个输出,即有多少个输入，梯度就有多少个输出

### **tensor.backward**

- 进行梯度反转

### **jacobian**
- 假设存在$y_1(x_1,...,x_n),...,y_m(x_1, ..., x_n)$, 偏导数矩阵为如下m * n 矩阵
$$
\begin{bmatrix}
\frac{\partial{y_1}}{\partial{x_1}} & ... & \frac{\partial{y_1}}{\partial{x_n}} \\
... & ... & ... \\
\frac{\partial{y_m}}{\partial{x_1}} & ... & \frac{\partial{y_m}}{\partial{x_n}}
\end{bmatrix} 
$$

- 雅克比矩阵：一阶偏导数矩阵

### **hessian**

- 海森矩阵：是一个多元函数的二阶偏导数构成的方阵，描述了函数的局部曲率， 二阶偏导数对一阶偏导再求导，类似$\dot{x}=\frac{dy}{dx},\ddot{x}=\frac{d\dot{x}}{dx}$

### **tensor.auto_grad**

### **autograd.backward**
- autograd.backward()为节约空间，仅会保存叶节点的梯度.若我们想得知输出关于某一中间结果的梯度，我们可以选择使用autograd.grad()接口

### **torch.utils.data.Dataset**
- Map-style dataset
- 它是一种通过实现 __getitem__() 和 __len()__ 来获取数据的 Dataset
```py
import datasets
class DatasetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.dataset[idx]["input_ids"]),
            torch.LongTensor(self.dataset[idx]["input_ids"]),
        )
# DatasetDataset自定义map style dataset, 按元组的方式返回数据
# torch.utils.data.DataLoader 通过DataLoader加载dataset指定batch_size和shuffle等参数
# RepeatingLoader 允许数据无限迭代下去        
dataset = datasets.load_from_disk(args.dataset_path)     
dataloader = RepeatingLoader(torch.utils.data.DataLoader(
    DatasetDataset(dataset),
    batch_size=args.batch_size,
    shuffle=True
))   

for s in ...:
    generator = iter(dataloader)
    input_ids, labels = next(generator) #获取数据的输入和输出
    
#数据格式，一个样本数据一般为1行，sent2一般为空
{"id":...,"sent1":...,"sent2":...,"ending1":...,"ending2":...,"ending3":...,"label":3}
# 将上述数据拼成一个句子(sentence)，通过tokenizer.encode(sentence)生成token    
```
### **torch.utils.data.IterableDataset**
- Iterable-style dataset
- 它是一种实现 __iter__() 来获取数据的 Dataset，这种类型的数据集特别适用于以下情况：随机读取代价很大甚至不大可能

### **其他Dataset**
- torch.utils.data.ConcatDataset: 用于连接多个 ConcatDataset 数据集
- torch.utils.data.ChainDataset : 用于连接多个 IterableDataset 数据集，在 IterableDataset 的 __add__() 方法中被调用
- torch.utils.data.Subset: 用于获取指定一个索引序列对应的子数据集

### **Sampler**
- torch.utils.data.SequentialSampler : 顺序采样样本，始终按照同一个顺序
- torch.utils.data.RandomSampler: 可指定有无放回地，进行随机采样样本元素
- torch.utils.data.SubsetRandomSampler: 无放回地按照给定的索引列表采样样本元素
- torch.utils.data.WeightedRandomSampler: 按照给定的概率来采样样本。样本元素来自 [0,…,len(weights)-1] ， 给定概率（权重）
- torch.utils.data.BatchSampler: 在一个batch中封装一个其他的采样器, 返回一个 batch 大小的 index 索引
- torch.utils.data.DistributedSample: 将数据加载限制为数据集子集的采样器。与 torch.nn.parallel.DistributedDataParallel 结合使用。 在这种情况下，每个进程都可以将 DistributedSampler 实例作为 DataLoader 采样器传递

### **torch.utils.data.DataLoader**
-  是 PyTorch 数据加载的核心，负责加载数据，同时支持 Map-style 和 Iterable-style Dataset，支持单进程/多进程，还可以设置 loading order, batch size, pin memory 等加载参数
```py
torch.utils.data.DataLoader(
    DatasetDataset(dataset),
    batch_size=args.batch_size,
    shuffle=True
)
```

### **torch.nn.Parameter**
- Parameters 是 torch.Tensor 的子类，当与 Modules 一起使用时具有一个非常特殊的属性 - 当它们被分配为 Module attributes 时，它们会自动添加到其参数列表中，并将 出现例如 在 parameters() 迭代器中。 分配张量没有这样的效果
- nn.Parameter的对象的requires_grad属性的默认值是True，即是可被训练的，这与torh.Tensor对象的默认值相反
- 将一个不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面。即在定义网络时这个tensor就是一个可以训练的参数了

### **torch.nn.Embedding**
```py
r"""
    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If given, pads the output with the embedding vector at :attr:`padding_idx`
                                         (initialized to zeros) whenever it encounters the index.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor.
                                 See Notes for more details regarding sparse gradients.
    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`
"""                         
```

### **torch.nn.Dropout**
```py
r"""
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)
"""
```

### tensor.view
- 类似reshape, 但是reshape后再修改 tensor 的值不确定是否会修改原始 tensor 的值，而view只是切换了视角，共享一份内存，修改view后的内容一定会将原tensor的内存内容修改。
-  tensor 连续条件的时候 tensor.reshape() 和 tensor.view() 效果相同; 当不满足时, tensor.reshape() 效果与 tensor.clone().view() 相同.

### tensor.contiguous
- Tensor多维数组底层实现是使用一块连续内存的1维数组,Tensor在元信息里保存了多维数组的形状，在访问元素时，通过多维度索引转化成1维数组相对于数组起始位置的偏移量即可找到对应的数据
- 某些Tensor操作（如transpose、permute、narrow、expand）与原Tensor是共享内存中的数据，不会改变底层数组的存储，但原来在语义上相邻、内存里也相邻的元素在执行这样的操作后，在语义上相邻，但在内存不相邻，即不连续了（is not contiguous）
- 如果Tensor不是连续的，调用tensor.contiguous则会重新开辟一块内存空间保证数据是在内存中是连续的，如果Tensor是连续的，则contiguous无操作

### torch.transpose(x, 0, 1)
- 交互两个维度，常用于矩阵转置，由于会引起tensor不连续，所以一般需要在其后调用contiguous

### **state_dict**
state_dict对象存储模型的可学习参数，即权重和偏差，并且可以非常容易地序列化和保存

### model.eval()
- 由于在验证或者测试时，我们不需要模型中的某些层起作用（比如:Dropout层），也不希望某些层的参数被改变（i.e. BatchNorm的参数），这时就需要设置成**model.eval()**模式
- 在模型测试或者验证时还需要使用到一个函数：torch.grad(),用来禁用梯度计算。此时即使设置了require_grad = True，也会被覆写为false
### model.train()
- 但是在训练模型的时候又希望这些层起作用，所以又要重新将这些层设置回来，这时候就需要用到**model.train()**模式

### nn.Linear(in_features, out_features) 
- (bs, in_features) * (in_features, out_features) +(1, out_features),对应$y= XW + b$

### arange(start=0, end, step=1,...)
- 返回大小为(end-start)/step)大小的一维张量，其值界于[start, end), 以step为步长

### nn.ModuleList
- 自动将每个module的parameters添加到网络之中的容器， 如果直接使用python list则不会添加到网络中，也不会更新参数
```py
self.linears = nn.ModuleList([nn.Linear(10,10) for i in range(2)] #正确，list元素顺序并不保证forward顺序，只是一般为了增强可读性，将二者保持一致
self.linears = [nn.Linear(10,10) for i in range(2)] #错误
```

### nn.Sequential
- 接受一个list的module， 而且list元素顺序并保证forward顺序一致，这是和ModuleList不一样的地方

### torch.squeeze(input, dim)
- 将张量shape[i]为1的维度压缩， (A×1×B×C×1×D)-> (A×B×C×D), 指定dim时，如果dim为1则将其压缩，否则不变

### torch.unsqueeze(input, dim)
- 将维度升维，将指定维度扩充为1
```py
x = torch.tensor([1, 2, 3, 4])
y = torch.unsqueeze(x, 0)#在第0维扩展，第0维大小为1

#输出结果如下：
#(tensor([[1, 2, 3, 4]]), torch.Size([1, 4]))
```

### model.apply(fn) ### 
- 会递归地将函数fn应用到父模块的每个子模块submodule，也包括model这个父模块自身
  
### torch.utils.checkpoint.checkpoint ### 
- transformer通过if self.gradient_checkpointing and self.training:调用

### torch.copy_
- 若采用直接赋值的方式，原来权重的存储指针会指向新得到的权重张量的存储区域,不会申请新空间，而是在原空间赋值，也可以给原空间指定区域赋值
```py
import torch
x = torch.tensor([[1,2], [3,4], [5,6]])
y = torch.rand((3,2)) # [0,1)之间均匀分布
print(y,id(y))
y = x #赋值操作，导致y的地址指向变了
print(y,id(y))

m = torch.rand((3,2)) # [0,1)之间均匀分布
print(m,id(m))
m.copy_(x) # copy_()操作，y的地址指向没变，只是重新赋值。
print(m,id(m))

z = torch.rand((4,2))
z[:x.shape[0],:x.shape[1]].copy_(x) #只拷贝x的大小区域
print(z[:x.shape[0],:x.shape[1]].copy_(x))
print(z)
```