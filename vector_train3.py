import math
import os
import pickle
import time
from contextlib import nullcontext
from itertools import accumulate

import numpy as np
import torch
from vector_model3 import GPTConfig, GPT, Emb2VectMLP
from torch.nn import functional as F

compile = True # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out_v3'
init_from = 'gpt2' # 'scratch' or 'resume' or 'gpt2*'
# data
dataset = 'openwebtext'
batch_size = 12
block_size = 1024
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
# initialize from OpenAI GPT-2 weights
model = GPT.from_pretrained(init_from, dict())
model.to(device)
model.eval()

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
# helps estimate an arbitrarily accurate loss over either split using many batches

@torch.no_grad()
def collect_data():
    X, Y = get_batch('train')
    with ctx:
        logits, _, x = model(X, Y)
        probs = F.softmax(logits.view((-1, logits.shape[2])), dim=1)
        return x.view((-1, x.shape[2])), probs

v2e_model = Emb2VectMLP(vocab_size=50257, k=1, v_size=256, bias=False)
v2e_model.to(device)
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_v2e_model = v2e_model
    v2e_model = torch.compile(v2e_model) # requires PyTorch 2.0
# training loop
warmup_iters = 100
learning_rate = 1e-4
min_lr = learning_rate/100
lr_decay_iters = 10000
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
decay_lr = True
out_dir = "out_head"
accumulate_interval = 10
max_iters = 10_000
t0 = time.time()
dt = 0
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f"Params k={v2e_model.k} v_emb={v2e_model.v_size}")

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

# optimizer
optimizer = torch.optim.AdamW(v2e_model.parameters(), lr=learning_rate, betas=(beta1, beta2))

while True:
    X, Y = collect_data()
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    with ctx:
        loss, good = v2e_model(X, Y)
    if iter_num % accumulate_interval == 0:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        iloss = loss.item()
        # iloss = loss1.item()
        # print(f"iter {iter_num}: loss {iloss:.4f} {iloss1:.4f} , time {dt * 1000:.2f}ms ")
        writer.add_scalar("Loss/iter", loss, iter_num)
        writer.add_scalar("Good/iter", good, iter_num)
        print(f"iter {iter_num}: loss {iloss:.4f}, good {good:.4f}, time {dt * 1000:.2f}ms ")
    if iter_num % accumulate_interval*10 == 0:
        writer.flush()
        torch.save(v2e_model.state_dict(), os.path.join(out_dir, 'ckpt.pt'))


    # loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    iter_num += 1
    if iter_num > max_iters:
        break
writer.close()
torch.save(v2e_model.state_dict(), os.path.join(out_dir, 'final-ckpt.pt'))