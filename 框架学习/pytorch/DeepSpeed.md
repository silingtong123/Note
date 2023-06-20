# 大模型运行

### deepspeed指定显卡
```shell
# --include/--exclude/--num_gpus/--num_nodes,使用其中一种方式
--num_gpus=4 #默认使用0-3
--include="localhost:1,3,5,7" #使用1,3,5,7显卡
# CUDA_VISIBLE_DEVICES=0,2,4,6 torchrun --nproc_per_node=4指定显卡
#  torchrun --nproc_per_node=4 默认使用0-3

# 保证train_batch_size = train_micro_batch_size_per_gpu * gradient_accumulation * number of GPUs
# 可以在deepspeed.config配置train_micro_batch_size_per_gpu，gradient_accumulation_steps，train_batch_size为auto

```

### 3090
- 3090 不支持NCLL p2p,需要禁止: export NCCL_P2P_DISABLE=1

### 禁止wandb
- export WANDB_DISABLED=1

### 模型保存
- peft_model.save_pretrained(OUTPUT_DIR)

### gradio使用报错
- 报错信息： Something went wrong Expecting value: line 1 column 
- 服务启动环境关闭代理即可

### 节省显存
- model.gradient_checkpointing_enable(): 过用计算换取内存来工作。检查点部分不是存储整个计算图的所有中间激活以进行反向计算，而是不保存中间激活，而是在反向过程中重新计算它们。它可以应用于模型的任何部分
- ZeRO stage 2
- ZeRO stage 2 + `offload_optimizer`
- stage 3
- `offload_param` to `cpu`
- `offload_optimizer` to `cpu`


### 单节点
```shell
pip install deepspeed>=0.9.0

git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/applications/DeepSpeed-Chat/
pip install -r requirements.txt

python train.py --actor-model facebook/opt-13b --reward-model facebook/opt-350m --deployment-type single_node
```

### 多节点
```shell
pip install deepspeed>=0.9.0

git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/applications/DeepSpeed-Chat/
pip install -r requirements.txt

python train.py --actor-model facebook/opt-13b --reward-model facebook/opt-350m --deployment-type single_node

pip install accelerate==0.19.0 # 升级accelerate 0.18.0 -》0.19.0
```

### 流水线并行（Pipeline Parallelism)
https://zhuanlan.zhihu.com/p/613196255
- 单卡到多卡的期望
  - 能训练更大的模型。理想状况下，模型的大小和GPU的数量成线性关系。即GPU量提升x倍，模型大小也能提升x倍
  - 能更快地训练模型。理想状况下，训练的速度和GPU的数量成线性关系。即GPU量提升x倍，训练速度也能提升x倍
- 期望的难点：
  - 训练更大的模型时，每块GPU里不仅要存模型参数，还要存中间结果，此时GPU中的内存限制会成为瓶颈
  - 网络通讯开销。数据在卡之间进行传输，是需要通讯时间的，此时GPU间的带宽限制
-  模型并行：不同层放置于不同的GPU上
   -  GPU利用率不高：对于单个batch而言，假设$K$块GPU，而单块GPU上做一次forward和backward的时间为$t_{fb}=t_f+t_b$
      -  单个gpu一层网络的forward和backward时间：$t_{fb}$，等待其他GPU做任务时的空闲时间为$(K-1)*t_{fb}$，所以单块GPU的时间为空闲时间加上工作时间为$K*t_{fb}$, 对于k块GPU来说为$K*K*t_{fb}$(Note：假设网络每一层的前向反向耗时相等, 长为$K*t_{fb}$， 宽为$K$，面积为$K*K*t_{fb}$)
      -  整个网络的实际forward和backward时间为每个GPU的工作时间：$K*t_{fb}$
      -  则GPU的空闲时间为：$K*(K-1)*t_{fb}$, 空闲占比为$\frac{K-1}{K}$,K越大空闲时间占比约接近100%
   - 中间结果占据大量内存：只考虑中间结果，不考虑参数本身。每一层的中间结果z。假设我们的模型有L层，每一层的宽度为d
     - 空间复杂度$O(N*\frac{L}{K}*d)$, N,L,d可能会抵消K的收益（Note: N从何而来，是z吗）
- 流水线并行：基于Gpipe。核心思想是：在模型并行的基础上，进一步引入数据并行的办法，即把原先的数据再划分成若干个batch，送入GPU进行训练
  - 切分micro-batch，在mini-batch上再划分的数据，叫micro-batch，将mini-batch划分为M个，而单块GPU上个micro-batch做一次forward和backward的时间为$t_{fb}=t_f+t_b$：解决GPU空转的问题
    - 总时间计算K块GPU的面积，宽为$K$,长为$(K+M-1)t_{fb}$,面积为$K*(K+M-1)t_{fb}$
    - 单块实际运行时间为$M*f_{fb}$,$K$块GPU运行时间为$M*f_{fb}$
    - 空闲时间为$K*(K-1)t_{fb}$, 空闲占比$O(\frac{K-1}{K+M-1})$, 当$M>=4K$时，空转占比会小于等于20%
  - re-materialization（active checkpoint：解决GPU的内存问题。几乎不存中间结果，等到backward的时候，再重新算一遍forward，只保存来自上一块的最后一层输入z，其余的中间结果我们算完就废
    - 记起始输入为$N$（即mini-batch的大小),中间变量大小为$\frac{N}{M}*\frac{L}{K}*d$,每块GPU峰值时刻的空间复杂度为$O(N+\frac{N}{M}*\frac{L}{K}*d)$,和单独的模型并行$O(N*\frac{L}{K}*d)$比较，当L变大时，对GPU内存的压力显著减小

### 数据并行
水线并行并不特别流行，主要原因是模型能否均匀切割，影响了整体计算效率，这就需要算法工程师做手调
- 核心思想：在各个GPU上都拷贝一份完整模型，各自吃一份数据，算一份梯度，最后对梯度进行累加来更新整体模型
  - DP: 最早的数据并行模式，一般采用PS架构，实际中多用于单机多卡
  - DDP: 分布式数据并行，采用Ring AllReduce的通讯方式，实际中多用于多机场景
  - ZeRO：零冗余优化器。由微软推出并应用于其DeepSpeed框架中。严格来讲ZeRO采用数据并行+张量并行的方式，旨在降低存储
- DP (AllReduce): 同步更新和异步更新
  - 每个worker将自己的数据发给其他的所有worker，然而这种方式存在大量的浪费
  - 利用主从式架构，将一个worker设为master，其余所有worker把数据发送给master之后，由master进行整合元算，完成之后再分发给其余worker， master成为瓶颈
- DDP (ring-reduce): 模型参数大小为K,则梯度大小也为K,每个梯度块的大小为K/N
  - scatter-reduce: 会逐步交换彼此的梯度并融合，最后每个 GPU 都会包含完整融合梯度的一部分； 数组求和，分发N-1次
  - allgather：GPU 会逐步交换彼此不完整的融合梯度，最后所有 GPU 都会得到完整的融合梯度；数据交换， 分发N-1
  - 若显卡数为N,每个卡上数据为K，两个阶段的数据传输为2*(N-1)*K/N = 2*K*(1-1/N),基本可以看出跟显卡数并不是线性增长，可近似看作2K
```py
torch.nn.Parallel.DistributedDataParallel

#设置数据加载器
torch.utils.data.Distributed.DistributedSampler

#设置分布式后端以管理GPU的同步
torch.distributed.init_process_group（backend =‘nccl’）

#从所有设备收集指定的input_tensor并将它们放置在collect_list中的dst设备上
torch.distributed.gather（input_tensor，collect_list，dst）

#从所有设备收集指定的input_tensor并将其放置在所有设备上的tensor_list变量中。
torch.distributed.all_gather(tensor_list，input_tensor)

#收集所有设备的input_tensor并使用指定的reduce操作（例如求和，均值等）进行缩减。最终结果放置在dst设备上
torch.distributed.reduce（input_tensor，dst，reduce_op = ReduceOp.SUM）

#与reduce操作相同，但最终结果被复制到所有设备
torch.distributed.all_reduce（input_tensor，reduce_op = ReduceOp.SUM）

# Note:
# 1. GPU之间拆分模型，请将模型拆分为sub_modules，然后将每个sub_module推送到单独的GPU
# 2. GPU上拆分批次，请使用累积梯度nn.DataParallel或nn.DistributedDataParallel
# 3. 在使用nn.DistributedDataParallel时，用nn.SyncBatchNorm替换或包装nn.BatchNorm层
```

  
### ZERO技术 显存优化
- 核心思想：思想就是用通讯换显存。ZERO将模型训练阶段，每张卡中显存内容分为两类,模型状态为显存占用大头，其中Adam状态为模型状态的75%,假设参数数量为X:
  - 模型状态（model states）: 模型参数（fp16，中间值)，模型梯度（fp16，中间值）和Adam状态（fp32的parameter，fp32的momentum和fp32的variance,必存), fp16为2字节，fp32为4字节，所以总数据大小为2X+2X+(4X+4X+4X+4X)=16X byte
    - 暂不将activation纳入统计范围
    - 很多states并不会每时每刻都用到，举例来说
      - Adam优化下的optimizer states只在最终做update时才用到
      - 数据并行中，gradients只在最后做AllReduce和updates时才用到
      - 参数W只在做forward和backward的那一刻才用到
  - 剩余状态（residual states）: 除了模型状态之外的显存占用，包括激活值（activation）、各种临时缓冲区（buffer）以及无法使用的显存碎片（fragmentation）
- ZERO： 每个显卡，只存1/N的数据
  - ZERO-1: $P_{os}$ 将模型状态中的Adam状态分片->单个显卡 4X + 12X/N,N无限大，4X, 不会增加通信量, 以下数据为了方便，没有换算为byte
    - 每块GPU上存一份完整的参数W。将一个batch的数据分成3份，每块GPU各吃一份，做完一轮foward和backward后，各得一份梯度
    - 对梯度进行AllReduce，2 * (N-1) *X/N ， 可近似看作2X
    - W的更新由optimizer states和梯度决定，由于每块GPU上只保管部分optimizer states，因此只能将相应的W进行更新
    - 此时，每块GPU上都有部分W没有完成更新,所以我们需要对W做一次All-Gather, (N-1) *X/N, 可近似看作X
    - ZERO-1的通信数据为3X，比DPP的2X 提高了1.5倍，但是单卡存储能显著降低
  - ZERO-2: $P_{os} +P_g$ 将模型状态中的Adam状态，模型梯度分片->单个显卡 2X+14X/N,N无限大，2X，不会增加通信量
    - 每块GPU上存一份完整的参数W。将一个batch的数据分成3份，每块GPU各吃一份，做完一轮foward和backward后，算得一份(完整的?）梯度
    - 对梯度做一次Reduce-Scatter，保证每个GPU上所维持的那块梯度是聚合梯度, (N-1) *X/N, 可近似看作X
    - 每块GPU用自己对应的optimizer states和梯度去更新相应的W。更新完毕后，每块GPU维持了一块更新完毕的W。同理，对W做一次All-Gather,  (N-1) *X/N, 可近似看作X
    - ZERO-2的通信数据为2X，DPP的2X相等
  - ZERO-3: $P_{os} +P_g +P_p$将模型状态中的Adam状态，模型梯度，模型参数分片->单个显卡 16X/N,无限大，0
    - 每块GPU上只保存部分参数W。将一个batch的数据分成3份，每块GPU各吃一份
    - 做forward时，对W做一次All-Gather，取回分布在别的GPU上的W，得到一份完整的W,forward做完，立刻把不是自己维护的W抛弃。(N-1) *X/N, 可近似看作X
    - 做backward时，对W做一次All-Gather，取回完整的W， backward做完，立刻把不是自己维护的W抛弃。(N-1) *X/N, 可近似看作X
    - 做完backward，算得一份完整的梯度G，对G做一次Reduce-Scatter，从别的GPU上聚合自己维护的那部分梯度，聚合操作结束后，立刻把不是自己维护的G抛弃。(N-1) *X/N, 可近似看作X
    - 用自己维护的O和G，更新W。由于只维护部分W，因此无需再对W做任何AllReduce操作
    - ZERO-1的通信数据为3X，比DPP的2X 提高了1.5倍，显存结果优化最为明显
- ZeRO VS 模型并行： ZeRO是模型并行的形式，数据并行的实质
  - 模型并行：指在forward和backward的过程中，我只需要用自己维护的那块W来计算就行；同样的输入X，每块GPU上各算模型的一部分，最后通过某些方式聚合结果
  - ZERO: ZeRO来说，它做forward和backward的时候，是需要把各GPU上维护的W聚合起来的，即本质上还是用完整的W进行计算；它是不同的输入X，完整的参数W，最终再做聚合
- ZeRO-Offload： 显存不够，内存来凑（Note:不能让通信成为瓶颈）

### Instruct Learning VS Prompt Learning
指示学习 vs 提示学习
- Prompt是激发语言模型的补全能力, 一般指文字接龙和完形填空的能力
  - 指令数据为 json 格式，包含instruction、input、output三个字段（可以为空），每行一条样本

  ```json
  {"instruction": "在以下文本中提取所有的日期。", "input": "6月21日是夏至，这是一年中白天最长的一天。", "output": "6月21日"}
  {"instruction": "", "input": "请生成一个新闻标题，描述一场正在发生的大型自然灾害。\\n\n", "output": "\"强烈飓风肆虐，数百万人疏散！\""}
  ```
  - 预训练语料
  - 下载语料后，合并到一个 .txt 文件并按行随机打乱，或者预训练数据也可以整理成jsonl格式
  ```json
  {"text": "doc1"}
  {"text": "doc2"}
  {"text": "doc3"}
  ``` 
- Instruct是激发语言模型的理解能力， 答案很有很多选项
- 指示学习的优点是它经过多任务的微调后，也能够在其他任务上做zero-shot，而提示学习都是针对一个任务的。泛化能力不如指示学习
- 他们指的预训练是：不仅包括从头训练的，还包括基于LLama进行大规模的增量训练。他们指的微调：少量精确数据集的tune。所以区分预训练还是tune，是看数据集的大小和精度

### Instruct-tuning VS Prompt-tuning
- SFT: 有监督微调
- FT: 无监督学习
- IFT: 指令微调
- RLHF: 人类反馈强化学习
- CoT: 思维链
- PEFT: Parameter-Efficient Fine-Tuning 参数高效微调
- Lora: 大型语言模型的低秩适应，解决微调大型语言模型的问题，原始模型被冻结，我们注入新的可训练层
  
NLP中基于Prompt的fine-tune
-  Prefix-Tuning
-  Prompt-Tuning
-  P-Tuning
-  P-Tuning-v2

### 从0开始预训练
- 选取基座模型
- 词表扩充
  - WordPiece Bert采用，「常用字」和「常用词」都存到词表中，特殊字符表示不存在的词
  - Byte Pair Encoder（BPE）：LLaMA采用。 也是用词表，但未找到的词按照unicode 编码。 BPE不是按照中文字词为最小单位，而是按照 unicode 编码 作为最小粒度，中文一般是由 3 个 unicode 编码组成。Chinese-LLaMA] 在LLaMA基础上扩充了词表
- 数据源采样:多个训练数据源,通过「数据源」采样的方式，能够缓解模型在训练的时候受到「数据集规模大小」的影响

- 数据预处理：文档向量化
  - Finetune 任务中，我们通常会直接使用 truncation 将超过阈值（2048）的文本给截断
  - 预训练最好的方式（？）是将长文章按照 seq_len（2048）作分割，将切割后的向量喂给模型做训练
- 模型结构：为了加快模型的训练速度，通常会在 decoder 模型中加入一些 tricks 来缩短模型训练周期
  - 大部分加速 tricks 都集中在 Attention 计算上