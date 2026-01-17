import torch
import torch.nn.functional as F
from model_exp import GPT, GPTConfig

def test_mtp_head():
    print("Testing Multi-Token Prediction Head...")
    
    config = GPTConfig(
        n_layer=1, 
        n_head=1, 
        n_embd=32, 
        block_size=16, 
        vocab_size=10,
        compression_factor=16
    )
    model = GPT(config)
    
    B, T = 1, 16
    idx = torch.randint(0, config.vocab_size, (B, T))
    # Targets should be (B, T, 16)
    targets = torch.randint(0, config.vocab_size, (B, T, 16))
    
    # Forward pass
    logits, loss = model(idx, targets)
    print(f"✓ Forward pass successful. Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    
    # Check if gradients flow to head and inner layers
    loss.backward()
    assert model.lm_head.weight.grad is not None, "Gradients should flow to lm_head"
    assert model.transformer.h[0].attn.c_attn.weight.grad is not None, "Gradients should flow to transformer layers"
    print("✓ Gradient flow passed")
    
    # Check generation (should use similarities for the first token)
    gen_idx = model.generate(idx, max_new_tokens=2)
    assert gen_idx.shape == (B, T + 2), f"Wrong generated shape: {gen_idx.shape}"
    print("✓ Generation successful")
    
    print("All Multi-Token Prediction Head tests passed!")

if __name__ == "__main__":
    test_mtp_head()
