import torch
from exp import CompressedModule

def test_learnable_compressed_module():
    print("Starting tests for Learnable CompressedModule...")
    
    B, T, C = 2, 64, 128
    block_size = 16
    model = CompressedModule(block_size=block_size, max_T=128)
    
    # 1. Shape test
    x = torch.randn(B, T, C, requires_grad=True)
    y = model(x)
    assert y.shape == (B, T, C), f"Expected shape {(B, T, C)}, got {y.shape}"
    print("✓ Shape test passed")
    
    # 2. Causality test: Changes in input after index 15 should not affect y[0:16]
    # In the learnable version, Block 0 depends on x[0:16]
    # Block 1 depends on x[0:32]
    # Block 2 depends on x[0:48]
    x1 = torch.randn(B, T, C)
    x2 = x1.clone()
    x2[:, 16:, :] += 10.0 # Modify everything AFTER the first block
    
    y1 = model(x1)
    y2 = model(x2)
    
    # Max difference in block 0
    block0_diff = (y1[:, :16, :] - y2[:, :16, :]).abs().max().item()
    assert block0_diff < 1e-6, f"Causality broken in Block 0: diff={block0_diff}"
    print("✓ Causality (Block 0) test passed")

    # 3. Cumulative dependency test: Block 1 should depend on Block 0
    x3 = x1.clone()
    x3[:, :16, :] += 5.0
    y3 = model(x3)
    block1_diff_from_x0 = (y1[:, 16:32, :] - y3[:, 16:32, :]).abs().max().item()
    # Note: If weights happen to be 0 for some indices, this might fail, 
    # but with random init it should be non-zero.
    assert block1_diff_from_x0 > 0, "Block 1 should depend on Block 0 (cumulative property)"
    print("✓ Cumulative dependency test passed")
    
    # 4. Gradient Flow Test (Learnability)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = y.pow(2).mean()
    loss.backward()
    
    # Check if weights have gradients
    assert model.weight.grad is not None, "Weight gradient should not be None"
    assert model.weight.grad.abs().sum() > 0, "Weight gradient should be non-zero"
    
    optimizer.step()
    print("✓ Gradient flow and optimizer step passed")

    # 5. Masked Gradient Test
    # Gradients for weights that are masked out (e.g. weight[0, 16:]) should be zero
    # because they don't contribute to any output.
    masked_grad = model.weight.grad[0, 16:]
    assert masked_grad.abs().max() == 0, f"Gradients should be zero for masked weights, got max={masked_grad.abs().max()}"
    print("✓ Masked gradient correctly zeroed")

    print("All tests passed!")

if __name__ == "__main__":
    test_learnable_compressed_module()
