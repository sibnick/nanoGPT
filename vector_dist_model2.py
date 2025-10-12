import math
import os
import time
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

class MLP2(nn.Module):

    def __init__(self, n_embd=768, bias=False, dropout=0.0):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(n_embd, n_embd, bias=bias)
        torch.nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        loss = torch.mean(torch.abs(torch.eye(x.shape[1], device=x.device) - x.T @ x))
        loss2 = torch.mean(torch.abs(1 - torch.norm(x, p=2, dim=1)))
        return x, loss + loss2, loss2


warmup_iters = 1000
learning_rate = 1e-3
min_lr = learning_rate/100
lr_decay_iters = 10000
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
decay_lr = True
out_dir = "out_head"
eval_interval = 100
max_iters = 10_000

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
sd_hf = model_hf.state_dict()
head_weight = sd_hf["lm_head.weight"]
head_weight = torch.cat((head_weight, torch.zeros((50304 - 50257, 768))), 0)
head_weight.requires_grad_(False)
head_weight = head_weight.to("cuda")

def get_batch():
    X = head_weight + 0.05 * (torch.rand(head_weight.shape, requires_grad=False, dtype=torch.float32, device="cuda") - 0.5)
    return X, X

model = MLP2()
print("compiling the model... (takes a ~minute)")
unoptimized_model = model
model = torch.compile(model) # requires PyTorch 2.0
model.to(device)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
# training loop
X, Y = get_batch() # fetch the very first batch
t0 = time.time()
iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
checkpoint = None # free up memory
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
dt = 0
torch.save(model.state_dict(), os.path.join(out_dir, 'ckpt.pt'))

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    with ctx:
        x, loss, loss2 = model(X)
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = get_batch()
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        iloss = loss.item()
        iloss2 = loss2.item()
        print(f"iter {iter_num}: loss {iloss:.4f} {iloss2:.4f} , time {dt*1000:.2f}ms ")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
torch.save(model.state_dict(), os.path.join(out_dir, 'ckpt.pt'))