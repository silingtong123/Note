# 常用API

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

- 如果F = D + E，F.grad_fn.next_functions也存在两项，分别对应于D, E两个变量，每个元组中的第一项对应于相应变量的grad_fn，第二项指示相应变量是产生其op的第几个输出

### **tensor.backward**

- 进行梯度反转

### **jacobian**

- 雅克比矩阵：一阶偏导数矩阵

### **hessian**

- 海森矩阵：是一个多元函数的二阶偏导数构成的方阵，描述了函数的局部曲率

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

### **state_dict**
state_dict对象存储模型的可学习参数，即权重和偏差，并且可以非常容易地序列化和保存