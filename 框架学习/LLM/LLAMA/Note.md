

### llama-7b-hf模型结构
- llama-7b-hf
  - config.json: LlamaConfig的配置文件, 包括bos_token_id，eos_token_id等信息
  - generation_config.json
  - pytorch_model-00001-of-00002.bin ...
  - pytorch_model.bin.index.json  
  - special_tokens_map.json ： 特殊token的记录，一般为空
  - tokenizer.model： 
  - tokenizer_config.json：tokenizer的配置文件，bos_token_id，eos_token_id等信息，一般为空

### LlamaTokenizer
```py
# 加载LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
class LlamaTokenizer(PreTrainedTokenizer):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

class PreTrainedTokenizer
    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        return self.vocab_size + len(self.added_tokens_encoder) 
data = load_dataset("json", data_files=data_args.train_file, cache_dir=model_args.cache_dir)
# DatasetDict({
#     train: Dataset({
#         features: ['instruction', 'output', 'input'],
#         num_rows: 20022
#     })
# }) 通过data['instruction']获取真实数据

```
- bos_token: token开始的分割符
- eos_token: token结束的分割符
- unk_token: 遇到未知字符的token
- pad_token：填充字符的token
- LlamaTokenizer的两种使用方式：
  - tokenizer.encode，再处理bos,eos
  - result = tokenizer( prompt, truncation=True, max_length=CUTOFF_LEN + 1,
        padding="max_length",
    ), 直接返回token
  - 同时设置max_length=10, truncation=True，超出长度的将会被截断
  - 设置padding="max_length"时，必须设置tokenizer.pad_token_id，此时不够长度的会扩充
- model.resize_token_embeddings(len(tokenizer)):
  - tokenizer的__len__被重载
- max_length=10, truncation=True
-  
### [在阿里GPU实例上配置eRDMA](https://help.aliyun.com/document_detail/2248432.html?spm=a2c4g.480208.0.0.494e5fb9JFp2Cw)

- 自动安装脚本方式:您可以选择通过自动安装脚本方式来安装安装OFED驱动,eRDMA软件栈、GPU驱动、CUDA以及cuDNN等软件，自动安装脚本示例如下所示。其中，关于DRIVER_VERSION、CUDA_VERSION、CUDNN_VERSION的版本选择
- 安装OFED驱动, 安装失败确认内核版本是否为3.10.0-1160.31.1.el7.x86_64，否则需要降级kernel,3.10.0-1160.90.1.el7.x86_64 -> 3.10.0-1160.31.1.el7.x86_64
```py
#降级kernel 
grub2-set-default 'CentOS Linux (3.10.0-1160.31.1.el7.x86_64) 7 (Core)'
grub2-editenv list
#内核版本已切换为3.10.0-1160.31.1.el7.x86_64， 重启
reboot


yum install -y python-devel libmnl-devel valgrind-devel rpm-build systemd-devel libdb-devel iptables-devel lsof libselinux-devel flex cmake elfutils-devel bison libnl3-devel numactl-devel

wget https://content.mellanox.com/ofed/MLNX_OFED-5.4-3.5.8.0/MLNX_OFED_LINUX-5.4-3.5.8.0-rhel7.9-x86_64.tgz
tar -zxf MLNX_OFED_LINUX-5.4-3.5.8.0-rhel7.9-x86_64.tgz
cd MLNX_OFED_LINUX-5.4-3.5.8.0-rhel7.9-x86_64
./mlnxofedinstall --kernel-only --without-fw-update -q
```
- 如果只想安装eRDMA软件栈，cuda相关环境已安装好 （host环境）
```shell
wget http://mirrors.cloud.aliyuncs.com/erdma/env_setup.sh

bash env_setup.sh --egs

# 通过eadm工具确认eRDMA驱动是否正常安装, 返回驱动版本表示正常
eadm ver

# 激活另一个网卡端口eth1
dhclient -v  eth1

#保证两个eRDMA端口都是ACTIVE状态
ibv_devinfo 

# 带宽测试
yum install perftest -y
#服务端
ib_send_bw -q 32 -n 100 --report_gbits
#客户端
ib_send_bw -q 32 -n 100 --report_gbits server_ip  # server_ip为服务器eRDMA网卡的IP地址

```

###  基于eRDMA容器内runtime的环境
- 运行以下命令，添加PGP签名 `wget -qO - https://mirrors.aliyun.com/erdma/GPGKEY | apt-key add -` 
- Ubuntu 18.04添加apt源 `echo "deb [ arch=amd64 ] https://mirrors.aliyun.com/erdma/apt/ubuntu bionic/erdma main" | tee /etc/apt/sources.list.d/erdma.list`
- 更新apt源 `sudo apt update`
- 安装用户态驱动 `apt install libibverbs1 ibverbs-providers ibverbs-utils librdmacm1 aiacc-nccl-plugin -y`
- 前面的命令可以写入docker制作镜像，再启动容器， 也可启动容器后使用：
```shell
docker run -it \
--runtime=nvidia --shm-size=8g --ipc=host \
--device=/dev/infiniband/rdma_cm \
--device=/dev/infiniband/uverbs0 \
--device=/dev/infiniband/uverbs1 \
--ulimit memlock=-1 \
--net=host \
<docker_image_id> /bin/bash

ibv_devinfo
```
- 保证容器内runtime的环境已安装好，进行e2e测试：
```shell
git clone 
pip install -r requirements.txt
pip install torchvision
pip install SentencePiece
wget https://ali-perseus-release.oss-cn-huhehaote.aliyuncs.com/ACSpeed/example/benchmark.py
# 修改benchmark.py中 local_rank为local-rank

# 以下为train.sh脚本内容

#! /bin/bash

MASTER_ADDR=172.20.20.148
MASTER_PORT=6000
NNODES=$1
NODE_RANK=$2
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

BS=2
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port
 $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS benchmark.py \
  --world-size=$WORLD_SIZE --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT

#运行命令
bash train.sh 2 0
bash train.sh 2 1

```

### AIACC使用说明：
1. 安装：
- wget https://ali-perseus-release.oss-cn-huhehaote.aliyuncs.com/AIACC/aiacc-1.1.0.tar.gz -O aiacc-1.1.0.tar.gz
- pip install aiacc-1.1.0.tar.gz
2. 使用：只需要训练代码开头增加一行
import aiacc 
3. 配置实例免密登录
```shell
ssh-keygen -t rsa
cat ~/.ssh/id_rsa.pub
echo <公钥> >> ~/.ssh/authorized_keys
```

### centos安装hdfs-fuse

- centos安装openjdk
```
yum install java-1.8.0-openjdk  java-1.8.0-openjdk-devel

rpm -ql java-1.8.0-openjdk

$ vi /etc/profile
# 在文件结尾加入以下内容
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.****
export CLASSPATH=.:$JAVA_HOME/jre/lib/rt.jar:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
export PATH=$PATH:$JAVA_HOME/bin

source /etc/profile

java -version
```