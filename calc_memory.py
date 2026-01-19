
import torch
import torch.nn as nn
from model_exp import GPT, GPTConfig

def format_size(num_params):
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return f"{num_params}"

def print_module_memory(model):
    print(f"{'Module':<50} | {'Params':<10} | {'FP32 (MB)':<10} | {'BF16 (MB)':<10}")
    print("-" * 90)
    
    total_params = 0
    
    # We want to show:
    # 1. Embeddings (wte)
    # 2. Individual layers (h.0, h.1, ...)
    # 3. Final norm (ln_f)
    # 4. LM Head
    
    def get_params(module):
        return sum(p.numel() for p in module.parameters())

    # Transformer modules
    transformer = model.transformer
    
    # WTE
    wte_params = get_params(transformer.wte)
    total_params += wte_params
    print(f"{'transformer.wte':<50} | {format_size(wte_params):<10} | {wte_params*4/(1024**2):.2f} | {wte_params*2/(1024**2):.2f}")
    
    # Layers
    for i, block in enumerate(transformer.h):
        block_params = get_params(block)
        total_params += block_params
        print(f"{f'transformer.h.{i}':<50} | {format_size(block_params):<10} | {block_params*4/(1024**2):.2f} | {block_params*2/(1024**2):.2f}")
        
        # Sub-modules of block for more detail (Optional, but let's show for first block)
        if i == 0:
            attn_params = get_params(block.attn)
            mlp_params = get_params(block.mlp)
            ln1_params = get_params(block.ln_1)
            ln2_params = get_params(block.ln_2)
            comp_params = get_params(block.compress)
            print(f"  {'-> attn':<48} | {format_size(attn_params):<10} | {attn_params*4/(1024**2):.2f} | {attn_params*2/(1024**2):.2f}")
            print(f"  {'-> mlp':<48} | {format_size(mlp_params):<10} | {mlp_params*4/(1024**2):.2f} | {mlp_params*2/(1024**2):.2f}")
            print(f"  {'-> ln_1':<48} | {format_size(ln1_params):<10} | {ln1_params*4/(1024**2):.2f} | {ln1_params*2/(1024**2):.2f}")
            print(f"  {'-> ln_2':<48} | {format_size(ln2_params):<10} | {ln2_params*4/(1024**2):.2f} | {ln2_params*2/(1024**2):.2f}")
            print(f"  {'-> compress':<48} | {format_size(comp_params):<10} | {comp_params*4/(1024**2):.2f} | {comp_params*2/(1024**2):.2f}")

    # LN_F
    ln_f_params = get_params(transformer.ln_f)
    total_params += ln_f_params
    print(f"{'transformer.ln_f':<50} | {format_size(ln_f_params):<10} | {ln_f_params*4/(1024**2):.2f} | {ln_f_params*2/(1024**2):.2f}")
    
    # LM Head
    lm_head_params = get_params(model.lm_head)
    total_params += lm_head_params
    print(f"{'lm_head':<50} | {format_size(lm_head_params):<10} | {lm_head_params*4/(1024**2):.2f} | {lm_head_params*2/(1024**2):.2f}")
    
    print("-" * 90)
    print(f"{'Total':<50} | {format_size(total_params):<10} | {total_params*4/(1024**2):.2f} | {total_params*2/(1024**2):.2f}")

# Config from train_shakespeare_byte_cmp.py
config = GPTConfig(
    block_size = 65536,
    vocab_size = 65,
    n_layer = 6,
    n_head = 8,
    n_embd = 128,
    dropout = 0.2,
    compression_factor = 16,
    bias = False,
)

model = GPT(config)
print_module_memory(model)
