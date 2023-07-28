# [transformer入门](https://transformers.run/intro/2021-12-11-transformers-note-2/)

### 一些常见的概念
- input_ids：词在字典中的id
- attention_mask: 在 self-attention 过程中，这一块 mask 用于标记 subword 所处句子和 padding 的区别
```py
input_ids = tokenizer(["I love China","I love my family and I enjoy the time with my family"], padding=True)
 
# print:
# {
#'input_ids': [[0, 100, 657, 436, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#              [0, 100, 657, 127, 284, 8, 38, 2254, 5, 86, 19, 127, 284, 2]], 
#'attention_mask': [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
#} 第一句话前5个词有效，后面0为padding部分
```

### transformer中的占位符
```py
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]
```
- bert中特殊标记(Special Tokens)：
  - [PAD]：在batch中对齐序列长度时，用 [PAD]进行填充以使所有序列长度相同。可以通过将其添加到较短的序列末尾来实现对齐
  - [CLS] 是 "classification" 的缩写，在文本分类任务中，它通常表示句子或文档的开头。在 BERT 中，[CLS] 对应着输入文本中第一个词的词向量，输出层中的第一个神经元通常会被用来预测文本的类别。
  - [SEP] 是 "separator" 的缩写，它通常表示句子或文档的结尾。在 BERT 中，[SEP] 对应着输入文本中最后一个词的词向量，它的作用是用来分割不同的句子。例如，在 BERT 中处理句子对时，两个句子之间通常会插入一个 [SEP] 来表示它们的分界点
  - [UNK]：此标记用于表示未知或词汇外的单词。当一个模型遇到一个它以前没有见过/无法识别的词时，它会用这个标记替换它

### transformer中的string tokens ids 三者转换
- string → tokens tokenize(text: str, **kwargs)
- tokens → string convert_tokens_to_string(tokens: List[token])
- tokens → ids convert_tokens_to_ids(tokens: List[token])
- ids → tokens convert_ids_to_tokens(ids: int or List[int], skip_special_tokens=False)
- string → ids encode(text,...)
- ids → string decode(token_ids: List[int], ...)
  
### 保存加载模型
```py
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
model.save_pretrained("./models/bert-base-cased/")
model = PeftModel.from_pretrained(model, training_args.peft_path) #lora训练

trainer.save_model() #可使用from_pretrained加载
```
- config.json：模型配置文件，存储模型结构参数，例如 Transformer 层数、特征空间维度等；
- pytorch_model.bin：又称为 state dictionary，存储模型的权重。

### 分词器
- 使用分词器 (tokenizer) 将文本按词、子词、字符切分为 tokens；
- 将所有的 token 映射到对应的 token ID。
- 切分策略
  - 按词切分 (Word-based)：如Python 的 split() 函数按空格进行分词
  - 按字符切分(Character-based)：
  - 按子词切分 (Subword)：高频词直接保留，低频词被切分为更有意义的子词
- 保存加载分词器：
  - special_tokens_map.json：映射文件，里面包含 unknown token 等特殊字符的映射关系；
  - tokenizer_config.json：分词器配置文件，存储构建分词器需要的参数；
  - vocab.txt：词表，一行一个 token，行号就是对应的 token ID（从 0 开始）
- 分词器的核心操作只有三个：tokenize, encode, decode
  - tokenizer.tokenize    得到分词
  - tokenizer.encode 实际上是tokenize和convert_tokens_to_ids两个操作的组合，类似于self.convert_tokens_to_ids(self.tokenize(text));会自动添加模型需要的特殊 token，例如 BERT 分词器会分别在序列的首尾添加 
 和 cls sep
```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.save_pretrained("./models/bert-base-cased/")
```

### 分析transformer pipeline
```py
SUPPORTED_TASKS ={    
  "text-generation": {
        "impl": TextGenerationPipeline,
        "tf": (TFAutoModelForCausalLM,) if is_tf_available() else (),
        "pt": (AutoModelForCausalLM,) if is_torch_available() else (),
        "default": {"model": {"pt": ("gpt2", "6c0e608"), "tf": ("gpt2", "6c0e608")}},
        "type": "text",
    },
    }

TASK_ALIASES = {
    "sentiment-analysis": "text-classification",
    "ner": "token-classification",
    "vqa": "visual-question-answering",
}

PIPELINE_REGISTRY = PipelineRegistry(supported_tasks=SUPPORTED_TASKS, task_aliases=TASK_ALIASES)

def check_task
    return PIPELINE_REGISTRY.check_task(task)

def pipeline():
  normalized_task, targeted_task, task_options = check_task(task)
  # ...

  pipeline_class = targeted_task["impl"]
  # ...

  return  pipeline_class
class Pipeline():
  '''
  Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following operations:
        Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output
  '''
  def __call__(self, inputs, *args):
    if inputs_is_list:
      return self.run_multi(inputs, ...) #对每个元素调用self.run_single
    else
      return self.run_single(inputs, ...)

    # preprocess， forward， postprocess一般由子类重写，Pipeline
    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

class TextGenerationPipeline(Pipeline):
   
   def __call__(self, text_inputs, **kwargs):
        return super().__call__(text_inputs, **kwargs)
```

### low_cpu_mem_usage 算法： 这是一个实验性函数，使用约 1 倍模型大小的 CPU 内存加载模型
- 保存我们拥有的 state_dict 键
- 在创建模型之前删除 state_dict，因为后者占用 1 倍模型大小的 CPU 内存
- 模型实例化后，切换到元设备，所有将从加载的参数/缓冲区中替换的参数/缓冲区
- 第二次加载 state_dict
- 替换 state_dict 中的参数/缓冲区