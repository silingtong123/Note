import os
from time import perf_counter
import numpy as np
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer,LlamaForCausalLM

def measure_pipeline_latency(generator, prompt, max_length, num_return_sequences):
    latencies = []
    # warm up
    for _ in range(2):
        output = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    # Timed run
    for _ in range(10):
        start_time = perf_counter()
        output = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)
    return f"P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f};", time_p95_ms


model_path ="/workdir/starcoderbase" #"/workdir/Finetune_LLAMA/hf/"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
#tokenizer = LlamaTokenizer.from_pretrained(model_path)

local_rank = int(os.getenv('LOCAL_RANK', '0'))
print(model.config)
generator = pipeline('text-generation', model=model,tokenizer=tokenizer, device=local_rank,do_sample=True)
set_seed(42)

import torch
import deepspeed
world_size = int(os.getenv('WORLD_SIZE', '2'))
print("-----------------world_size-----",world_size)
generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.half,
                                           replace_with_kernel_inject=True)

string = generator("DeepSpeed is", do_sample=True, max_length=50, num_return_sequences=4)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(string)

ds_results = measure_pipeline_latency(generator, "Hello, I'm a language model,", 50, 4)
print(f"DS model: {ds_results[0]}")

#run cmd: deepspeed --num_gpus 2 infer_deepspeed_pipline.py 
# deepspeed --num_gpus --include="localhost:1,3" infer_deepspeed_pipline.py