import torch
import torch.nn.functional as F
from model_exp import GPT, GPTConfig

def test_l2_model():
    print("Testing L2 GPT model...")
    
    config = GPTConfig(
        n_layer=1, 
        n_head=1, 
        n_embd=32, 
        block_size=16, 
        vocab_size=10,
        compression_factor=4
    )
    model = GPT(config)
    
    B, T = 1, 8
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    
    # Check weight normalization in init
    w_init_norm = model.transformer.wte.weight.norm(p=2, dim=-1)
    assert torch.allclose(w_init_norm, torch.ones_like(w_init_norm), atol=1e-5), "Embeddings should be normalized in init"
    print("✓ Init weight normalization passed")
    
    # Forward pass
    logits, loss = model(idx, targets)
    print(f"✓ Forward pass successful. Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    
    # Check if gradients flow
    loss.backward()
    assert model.transformer.wte.weight.grad is not None, "Gradients should flow to embeddings"
    print("✓ Gradient flow passed")
    
    # Check generation (should use similarities)
    gen_idx = model.generate(idx, max_new_tokens=2)
    assert gen_idx.shape == (B, T + 2), f"Wrong generated shape: {gen_idx.shape}"
    print("✓ Generation successful")
    
    # Internal normalization check during forward
    # We can use a hook or just trust the code if it runs and gradients flow to weights.
    # But let's check the inference path logits
    logits_inf, _ = model(idx)
    # Logits are similarity: x @ w.t()
    # Since x and w are unit normalized, similarity should be in range [-1, 1]
    assert logits_inf.max() <= 1.0001 and logits_inf.min() >= -1.0001, f"Logits out of range: {logits_inf.min()} to {logits_inf.max()}"
    print("✓ Inference logits range passed (unit norms confirmed)")

    print("All L2 GPT tests passed!")

if __name__ == "__main__":
    test_l2_model()
