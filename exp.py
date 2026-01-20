import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CompressedModule(nn.Module):
    """
    A module that compresses the sequence length by a factor of block_size (default 16) 
    using per-head learnable transformations on local blocks.
    
    Input shape: (B, T, C)
    Internal shape: (B, n_head, T/block_size, head_dim)
    Output shape: (B, T, C)
    
    Constraint: T must be divisible by block_size.
    """
    def __init__(self, n_embd, n_head, block_size=16):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size
        self.head_dim = n_embd // n_head
        
        # Per-head learnable weights for temporal compression: (n_head, block_size)
        # Each head learns its own weighted sum of the local block (like a CNN kernel).
        self.weight = nn.Parameter(torch.randn(n_head, block_size) / math.sqrt(block_size))
        
        # Per-head learnable projections for expansion.
        # Maps 1 compressed head vector to (block_size * head_dim) values.
        # Shape: (n_head, head_dim, block_size * head_dim)
        self.expand_proj       = nn.Parameter(torch.randn(n_head, self.head_dim, block_size * self.head_dim) / math.sqrt(self.head_dim))
        #self.expand_proj_final = nn.Parameter(torch.randn(n_head, self.head_dim, block_size * self.head_dim) / math.sqrt(self.head_dim))

    def compress(self, x):
        B, T, C = x.shape
        # Reshape to (B, T//BS, BS, n_head, head_dim) then transpose to (B, n_head, T//BS, BS, head_dim)
        x_blocks = x.view(B, T // self.block_size, self.block_size, self.n_head, self.head_dim)
        x_blocks = x_blocks.permute(0, 3, 1, 2, 4) # (B, n_head, T//BS, BS, head_dim)
        
        # Apply per-head temporal weighting: (n_head, BS) @ (B, n_head, T//BS, BS, head_dim) -> (B, n_head, T//BS, head_dim)
        # B: Batch, h: head, m: mid, b: block, d: dim
        mid = torch.einsum('hb, Bhmbd -> B h m d', self.weight, x_blocks)
        return mid

    def expand_inner(self, mid, proj_weight):
        # mid: (B, n_head, M, head_dim)
        # proj_weight: (n_head, head_dim, BS * head_dim)
        # Output: (B, n_head, M, BS * head_dim)
        # B: batch, h: head, m: mid, d: head_dim, v: block_size * head_dim
        expanded = torch.einsum('B h m d, h d v -> B h m v', mid, proj_weight)
        
        B, H, M, _ = expanded.shape
        # Reshape to (B, H, M, BS, D) then transpose to (B, M, BS, H, D) then flatten to (B, T, C)
        expanded = expanded.view(B, H, M, self.block_size, self.head_dim)
        expanded = expanded.permute(0, 2, 3, 1, 4).contiguous()
        expanded = expanded.view(B, M * self.block_size, self.n_embd)
        return expanded

    def forward(self, x):
        B, T, C = x.shape
        
        # Handle non-divisible lengths by padding (though usually T is a power of 2)
        padding = 0
        if T % self.block_size != 0:
            padding = self.block_size - (T % self.block_size)
            x = F.pad(x, (0, 0, 0, padding))
            
        mid_heads = self.compress(x) # (B, n_head, M, head_dim)
        
        # For external consumption in Attention/MLP, we need (B, M, C)
        # (B, n_head, M, head_dim) -> (B, M, n_head, head_dim) -> (B, M, C)
        mid = mid_heads.permute(0, 2, 1, 3).contiguous().view(B, -1, self.n_embd)
        
        restored = self.expand_inner(mid_heads, self.expand_proj)
        
        # Crop back if we padded
        if padding > 0:
            restored = restored[:, :T, :]
            
        return mid, restored

    def expand_final(self, mid):
        # mid is (B, M, C)
        # Convert back to heads: (B, n_head, M, head_dim)
        B, M, C = mid.shape
        mid_heads = mid.view(B, M, self.n_head, self.head_dim).permute(0, 2, 1, 3).contiguous()
        return self.expand_inner(mid_heads, self.expand_proj) #_final)
