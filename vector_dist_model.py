import math
import os
import time
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel


class MLP(nn.Module):

    def __init__(self, n_embd=768, bias=False, dropout=0.0):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MLP2(nn.Module):

    def __init__(self, n_embd=768, bias=False, dropout=0.0):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4*n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(n_embd*4, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x



class VectorHead(nn.Module):
    def __init__(self, n_emb: int = 768, vocab_size : int = 50304):
        super(VectorHead, self).__init__()
        self.n_emb = n_emb
        self.head1 = MLP2()
        # self.head2 = MLP2()
        # self.head3 = MLP2()
        # self.head4 = MLP2()
        # p = 1
        # for head in (self.head1): #, self.head2, self.head3, self.head4):
        #     torch.nn.init.normal_(head.c_fc.weight, mean=0.0, std=0.02 * p)
        torch.nn.init.normal_(self.head1.c_fc.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.head1.c_proj.weight, mean=0.0, std=0.02)
        #     p *= 0.5


    def forward(self, x, targets=None):
        # x = self.head1(x)
        # x = self.head2(x)
        # x = self.head3(x)
        # x = self.head4(x)
        x = self.head1(x)#+self.head2(x)+self.head3(x)+self.head4(x)
        #loss = F.mse_loss(target=targets, input=x)
        loss = F.cosine_embedding_loss(input1=targets, input2=x, target=torch.ones(x.shape[0], device=x.device))
        return loss
        # p = 0.05
        # l = 0
        # for head in (model.head1, model.head2, model.head3, model.head4):
        #     l += torch.mean(torch.abs(head.c_fc.weight)) * p
        #     p *= 2
        #
        # return loss + l, l


warmup_iters = 10
learning_rate = 1e-2
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
max_iters = 10000

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


# init a huggingface/transformers model
model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
sd_hf = model_hf.state_dict()
head_weight = sd_hf["lm_head.weight"]
head_weight = torch.cat((head_weight, torch.zeros((50304 - 50257, 768))), 0) 
head_weight.requires_grad_(False)
head_weight = head_weight.to("cuda")

embeddings = torch.load("/home/nikolay/PycharmProjects/nanoGPT/out_head/768.pt")
vector_embeddings = embeddings
# embeddings = torch.load("/data/nikolay/flibusta-prj/static-emb/data_openweb/glove-768-8.pt")
# vector_embeddings = embeddings["_orig_mod._context_embeddings.weight"] + embeddings["_orig_mod._focal_embeddings.weight"]
vector_embeddings.requires_grad_(False)
vector_embeddings = vector_embeddings.to("cuda")


def get_batch():
    X = head_weight + 0.05 * torch.mean(torch.abs(head_weight), axis=0) * (torch.rand(head_weight.shape, requires_grad=False, dtype=torch.float32, device="cuda") - 0.5)
    Y = vector_embeddings
    return X, Y

model = VectorHead()
print("compiling the model... (takes a ~minute)")
#unoptimized_model = model
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
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    with ctx:
        loss = model(X, Y)
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = get_batch()
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        iloss = loss.item()
        print(f"iter {iter_num}: loss {iloss:.4f}, time {dt*1000:.2f}ms ")
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
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