import torch
import torch.nn as nn
from torch.nn import functional as F

# 获取莎士比亚的数据集
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding='utf-8') as f:
    text = f.read()
print("length of text: ", len(text))

print(text[:1000])

chars = sorted(list(set(text)))
vocav_size = len(chars)
print("length of chars: ", len(chars))
print(''.join(chars))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join(itos[i] for i in l)

print(encode("hill there"))
print(decode(encode("hill there")))

data = torch.tensor(encode(text), dtype= torch.long)
# print(data.shape, data.type)
# print(data[:1000])

n = int(0.9 * len(data))
train_data = data[:n]
eval_data = data[n:]

#块大小，类似seq_length
block_size = 8
batch_size = 4 
#设置seed是为了复现结果
torch.manual_seed(1234)

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"input is ${context}, target is {target}")


# batch_size = block_size
def get_batch(split):
    data = train_data if split == 'train' else eval_data
    idx = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1: i+1+block_size] for i in idx])
    return x, y
xa,xb = get_batch('train')
# print(xa)
# print(xb)
# print(xa.shape)
# print(xb.shape)

class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, vocab_size) 
        
    def forward(self, input, target = None):
        # input and target are both [B, T] tensor of integer
        logits = self.token_emb_table(input) #[B, T, C]
        
        if target == None: # 在生成的时候，我们不需要计算loss
            loss = None
        else:         
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
            #  print(f"loss: {loss}")
        return logits, loss
    
    def generate(self, input, max_new_token):
        #input [B*T]
        for _ in range (max_new_token):
            logits, loss = self(input)
            #只关注最后一个step
            logits= logits[:, -1, :] # B*C
            probs = F.softmax(logits, dim = -1) # B*C
            input_next = torch.multinomial(probs, num_samples= 1) #(B,1)
            #增加seq_length，也就是block_size
            input = torch.cat((input, input_next), dim = 1) #(B, T+1)
        return input
            
model = BigramLM(vocav_size)

output,_ = model(xa, xb)
print(output.shape)
# 0代表新行的起点
input = torch.zeros((1,1),dtype = torch.long)
print(decode(model.generate(input, max_new_token=100)[0].tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32
# 5000 step loss从4.5 下降到2.5
for steps in range(5000):
    #get train data
    xa, yb = get_batch('train')
    
    #evaluate the loss
    logits, loss = model(xa, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    print(loss.item())
print(decode(model.generate(input, max_new_token=100)[0].tolist()))