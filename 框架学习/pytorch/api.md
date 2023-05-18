# 常用API

### **torch.tensor([1.], requires_grad=True)**

- 需要设置tensor的requires_grad属性为True，才会进行梯度反传

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