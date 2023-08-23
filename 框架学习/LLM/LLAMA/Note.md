

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