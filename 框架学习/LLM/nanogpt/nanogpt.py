import torch
import torch.nn as nn
from torch.nn import functional as F

with open("input.txt", "r", encoding='utf-8') as f:
    text = f.read()
print("length of text: ", len(text))

print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
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

device = 'cuda'
eval_iters = 200
eval_interval = 500
max_iter = 5000
n_emb = 384
#块大小，类似seq_length
block_size = 256
batch_size = 64
n_layer = 6
n_head = 6
dropout = 0.2
learning_rate = 3e-4
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
    x, y = x.to(device), y.to(device)
    return x, y
xa,xb = get_batch('train')
# print(xa)
# print(xb)
# print(xa.shape)
# print(xb.shape)

@torch.no_grad()
def estimate_loss():
    out = {}
    
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split]=losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ 实现单head的自注意力机制"""
    def __init__(self, head_size):
        super().__init__()
        # print('---------')
        # print(n_emb)
        # print(head_size)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn_score = q @ k.transpose(-2,-1) * C**-0.5 # [B, T, C] @[ B, C, T] ---> [B, T, T]
        attn_score = attn_score.masked_fill(self.tril[:T, : T]==0, float('-inf'))
        attn_score = F.softmax(attn_score, dim= -1)
        attn_score = self.drop(attn_score)
        out = attn_score @ v #[B, T, T] @[B, T, C] --->[B,T, C]
        return out
    
class MultiHeadAttention(nn.Module):
    """ 创建多头注意力机制"""
    def __init__(self, head_num, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(head_num)])
        self.proj = nn.Linear(head_num* head_size, n_emb) 
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        out = self.drop(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4* n_emb, n_emb),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """创建transformer块"""
    def __init__(self, n_emb, head_num):
        super().__init__()
        head_size = n_emb // head_num
        self.sa_head = MultiHeadAttention(head_num, head_size)
        self.ffn = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
        #self.lm_head = nn.Linear(n_emb, vocab_size)
        
    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
            
class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        # self.token_emb_table = nn.Embedding(vocab_size, vocab_size) 
        self.token_emb_table = nn.Embedding(vocab_size, n_emb)
        self.position_emb_table = nn.Embedding(block_size, n_emb)
        #self.sa_head = Head(n_emb)
        
        # self.sa_head = MultiHeadAttention(4, int(n_emb/4))
        # self.ffn = FeedForward(n_emb)
        
        # self.blocks = nn.Sequential(
        #     Block(n_emb, head_num=4),
        #     Block(n_emb, head_num=4),
        #     Block(n_emb, head_num=4),
        #     nn.LayerNorm(n_emb)
        # )
        
        # 使用n_layers设置block数量
        self.blocks = nn.Sequential(
            *[Block(n_emb, head_num=n_head) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)
        
    def forward(self, input, target = None):
        # input and target are both [B, T] tensor of integer
        
        B, T = input.shape
    
        token_emb = self.token_emb_table(input) #[B, T, C]
        pos_emb = self.position_emb_table(torch.arange( T, device=device)) #[T, C]
        x = token_emb + pos_emb # pos_emb会在这里广播进行匹配维度
        # x = self.sa_head(x)
        # x = self.ffn(x)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x) #[B, T, vocab_size]
        
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
            input_cond = input[:, -block_size:]
            logits, loss = self(input_cond)
            #只关注最后一个step
            logits= logits[:, -1, :] # B*C
            probs = F.softmax(logits, dim = -1) # B*C
            input_next = torch.multinomial(probs, num_samples= 1) #(B,1)
            #增加seq_length，也就是block_size
            input = torch.cat((input, input_next), dim = 1) #(B, T+1)
        return input
            
model = BigramLM()
model.to(device)
output,_ = model(xa, xb)
print(output.shape)
# 0代表新行的起点
input = torch.zeros((1,1),dtype = torch.long, device=device)
print(decode(model.generate(input, max_new_token=100)[0].tolist()))

#=============================如何使用矩阵乘法和下三角函数结合，求前x[0:t]的均值 ==========
#  ---------------------------> version 1: 使用for循环  <---------------------------
torch.manual_seed(1337)
B, T, C =4,8,32
x = torch.randn(B,T,C)
print(x.shape)
print(x[0])

xbow = torch.zeros(B, T,C)
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] #(t,C)
        xbow[b,t] = torch.mean(xprev,0)


# torch.manual_seed(42)
# a = torch.tril(torch.ones(3,3)) #下三角函数，上三角全为0
# a = a/torch.sum(a,1,keepdim=True) #归一化，保证和为1
# print("a -------",a)
# b = torch.randint(0,10,(3,2)).float()
# print("b -----", b)
# c =a @ b
# print(c.shape)
# print("c -----",c)
#  ---------------------------> version 2:use tril <---------------------------
weight = torch.tril(torch.ones(T,T))
weight = weight/weight.sum(1, keepdim=True)
xbow2 = weight @ x #(B, T ,T) @ (B, T, C) ---------> (B, T, C)
print(torch.allclose(xbow, xbow2, atol=1e-07)) #默认的1e-8会有误差，需要设置为1e-7
# o = xbow -xbow2
# print("o ====", o)
# ---------------------------> version 3: use softmax <---------------------------
tril = torch.tril(torch.ones(T,T))
w = torch.zeros(T,T)
w = w.masked_fill(tril == 0, float('-inf'))
w = F.softmax(w, dim = -1)
xbow3 = w @ x
print(torch.allclose(xbow, xbow3, atol=1e-07)) #默认的1e-8会有误差，需要设置为1e-7

# 实现单head的自注意力
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
q = query(x) #(B, T, 16)
k = key(x) #(B, T, 16)
v = value(x)

attn_score = q @ k.transpose(-2,-1) # (B, T, 16) @(B, 16, T) ---->(B, T, T)
tril = torch.tril(torch.ones(T,T))
wei = attn_score.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim = -1)
print(wei)
out = wei @ v # (B, T, T ) @ (B, T, 16) ----> (B, T, 16)
#exit(0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32
# 5000 step loss从4.5 下降到2.5
for steps in range(max_iter):
    
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss: {losses['train']:.4f}, val loss: {losses['eval']:.4f}")
    #get train data
    xa, yb = get_batch('train')
    
    #evaluate the loss
    logits, loss = model(xa, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print("loss = ", loss.item())
print(decode(model.generate(input, max_new_token=100)[0].tolist()))

