import torch
import torch.nn as nn
import torch.nn.functional as F

class SinkhornFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, iterations, eps):
        # W is (n, n), small matrix
        W = F.softplus(W)
        H = W
        for _ in range(iterations):
            H = H / (H.sum(dim=-1, keepdim=True) + eps)
            H = H / (H.sum(dim=-2, keepdim=True) + eps)
        ctx.save_for_backward(H)
        return H

    @staticmethod
    def backward(ctx, grad_output):
        H, = ctx.saved_tensors
        # Implicit gradient for doubly stochastic matrices
        grad_W = H * (grad_output - (grad_output * H).sum(dim=-1, keepdim=True) 
                      - (grad_output * H).sum(dim=-2, keepdim=True))
        return grad_W, None, None

def fast_sinkhorn(W, iterations=20, eps=1e-8):
    return SinkhornFunction.apply(W, iterations, eps)

class mHCLayer(nn.Module):
    def __init__(self, n, layer_fn, residual_fn=None, iterations=20):
        super().__init__()
        self.n = n
        self.layer_fn = layer_fn
        self.residual_fn = residual_fn
        self.iterations = iterations
        
        self.H_res_raw = nn.Parameter(torch.randn(n, n) * 0.02)
        self.H_pre = nn.Parameter(torch.randn(1, n) * 0.02)
        self.H_post = nn.Parameter(torch.randn(n, 1) * 0.02)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Project mapping matrix
        H_res = fast_sinkhorn(self.H_res_raw, self.iterations)
        
        # 2. Residual path
        if self.residual_fn is not None:
            # Reshape can handle non-contiguous input from repeat()
            x_reshaped = x.reshape(batch_size * self.n, *x.shape[2:])
            x_res = self.residual_fn(x_reshaped)
            x_res = x_res.reshape(batch_size, self.n, *x_res.shape[1:])
        else:
            x_res = x
            
        # Mix residual streams
        residual = torch.einsum('ij, bj... -> bi...', H_res, x_res).contiguous()
        
        # 3. Transformed path
        H_pre = torch.sigmoid(self.H_pre)
        x_collapsed = torch.einsum('in, bn... -> bi...', H_pre, x).squeeze(1).contiguous()
        
        # NOTE: Forcing channels_last here might help torch.compile satisfy layout assertions
        # especially for convolutions in the backward pass.
        if x_collapsed.dim() == 4:
            x_collapsed = x_collapsed.to(memory_format=torch.channels_last)
            
        f_out = self.layer_fn(x_collapsed)
        
        H_post = torch.sigmoid(self.H_post)
        transformed = torch.einsum('ni, bi... -> bn...', H_post, f_out.unsqueeze(1)).contiguous()
        
        return residual + transformed

class DeepMHCLayer(mHCLayer):
    def forward(self, x):
        from torch.utils.checkpoint import checkpoint
        return checkpoint(super().forward, x, use_reentrant=False)