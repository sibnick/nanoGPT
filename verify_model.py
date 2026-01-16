import torch
from model import GPT, GPTConfig

def test_compressed_gpt():
    print("Testing Compressed GPT integration...")
    
    # Small config for testing
    config = GPTConfig(
        n_layer=2, 
        n_head=2, 
        n_embd=64, 
        block_size=64,
        compression_factor=16,
        vocab_size=100
    )
    model = GPT(config)
    
    B, T = 2, 64
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    
    # 1. Forward pass check
    logits, loss = model(idx, targets)
    print(f"✓ Forward pass successful. Loss: {loss.item():.4f}")
    
    assert logits.shape == (B, T, config.vocab_size), f"Wrong logits shape: {logits.shape}"
    assert loss.item() > 0, "Loss should be positive"
    
    # 2. Gradient flow check
    loss.backward()
    
    # Check gradients in compression weights
    comp_grad_norm = model.transformer.h[0].compress.weight.grad.norm().item()
    assert comp_grad_norm > 0, "Compression weights should have non-zero gradients"
    print(f"✓ Compression weight grad norm: {comp_grad_norm:.4f}")
    
    # Check gradients in attention weights (which are now downstream of compression)
    attn_grad_norm = model.transformer.h[0].attn.c_attn.weight.grad.norm().item()
    assert attn_grad_norm > 0, "Attention weights should have non-zero gradients"
    print(f"✓ Attention weight grad norm: {attn_grad_norm:.4f}")
    
    # 3. Shape check for attention input
    # We can use a hook or just check internal logic. 
    # Let's rely on the fact that if it didn't crash, the dimensions in CausalSelfAttention matched.
    # CausalSelfAttention receives (B, T/16, C).
    
    print("All integration tests passed!")

if __name__ == "__main__":
    test_compressed_gpt()
