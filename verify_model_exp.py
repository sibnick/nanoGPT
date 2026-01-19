import torch
import torch.nn.functional as F
from model_exp import GPT, GPTConfig

def test_model():
    print("Testing GPT model with Softmax Head...")
    
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
    # targets shape for multi-token prediction should be (B, T, compression_factor)
    targets = torch.randint(0, config.vocab_size, (B, T, config.compression_factor))
    
    # Check weight normalization in init (still kept for embedding dispersion)
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
    
    # Check generation
    gen_idx = model.generate(idx, max_new_tokens=2)
    assert gen_idx.shape == (B, T + 2), f"Wrong generated shape: {gen_idx.shape}"
    print("✓ Generation successful")
    
    # Inference path logits check
    logits_inf, _ = model(idx)
    # logits shape: (B, 1, 4, 10)
    assert logits_inf.shape == (B, 1, config.compression_factor, config.vocab_size)
    print("✓ Inference logits shape passed")

    print("All GPT tests passed!")

if __name__ == "__main__":
    test_model()
