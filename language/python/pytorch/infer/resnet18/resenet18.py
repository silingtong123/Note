import torch
import torchvision
import time
# An instance of your model.
model = torchvision.models.resnet18()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("devece---------",device)
# An example input you would normally provide to your model's forward() method.
# example = torch.rand(1, 3, 224, 224).type(torch.float32).to(torch.device("cuda"))
#model.to(device)
example = torch.rand(1, 3, 224, 224)
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
for i in range(2):
    output = model(example)
T1 = time.time()
for i in range(100):
    output = model(example)
T2 = time.time()
#print(output)
print("python time cost -------------%s"%((T2-T1)*1000))

'''
GPU inference：
inputs move GPU:
example = torch.rand(1, 3, 224, 224).type(torch.float32).to(torch.device("cuda"))

model move GPU:
model.to(device) 

A100机器 单卡：python time cost -------------198.8077163696289 ms
A100机器 CPU: python time cost -------------950.3262042999268 ms
'''