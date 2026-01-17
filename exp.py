import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CompressedModule(nn.Module):
    """
    A module that compresses the sequence length by a factor of 16 using a learnable linear transformation.
    Input shape: (B, T, C)
    Middle compressed layer shape: (B, T/16, C)
    Output shape: (B, T, C)
    
    Causality:
    - The i-th compressed token at index i (representing a block of 16) is a learnable
      weighted sum of all input tokens from index 0 to (i+1)*16 - 1.
    - Output tokens in index range [16*i, 16*(i+1)) only use the i-th compressed token.
    """
    def __init__(self, block_size=16, max_T=1024):
        super().__init__()
        self.block_size = block_size
        self.max_T = max_T
        num_mid = max_T // block_size
        
        # Learnable weights for temporal compression: (num_mid, max_T)
        # Each 'mid' token is a linear combination of input tokens
        self.weight = nn.Parameter(torch.randn(num_mid, max_T) / math.sqrt(block_size))
        
        # Causal mask for cumulative property
        # mask[i, t] = 1 if input token t affects middle token i
        # Middle token i represents block i, which depends on tokens 0 to (i+1)*block_size-1
        mask = torch.zeros(num_mid, max_T)
        for i in range(num_mid):
            mask[i, : (i + 1) * block_size] = 1.0
        self.register_buffer("mask", mask)

    def compress(self, x):
        B, T, C = x.shape
        num_mid = T // self.block_size
        w = self.weight[:num_mid, :T] * self.mask[:num_mid, :T]
        mid = torch.einsum('mt, btc -> bmc', w, x)
        return mid

    def expand(self, mid):
        return mid.repeat_interleave(self.block_size, dim=1)

    def forward(self, x):
        B, T, C = x.shape
        if T > self.max_T:
            raise ValueError(f"Sequence length T ({T}) exceeds max_T ({self.max_T})")
        
        # Handle non-divisible lengths by padding
        padding = 0
        if T % self.block_size != 0:
            padding = self.block_size - (T % self.block_size)
            x = F.pad(x, (0, 0, 0, padding))
            
        mid = self.compress(x)
        restored = self.expand(mid)
        
        # Crop back if we padded
        if padding > 0:
            restored = restored[:, :T, :]
        
        return mid, restored
