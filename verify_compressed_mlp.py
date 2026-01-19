
import torch
from model_exp import GPT, GPTConfig

def verify():
    config = GPTConfig(
        block_size = 64,
        vocab_size = 256,
        n_layer = 2,
        n_head = 4,
        n_embd = 128,
        compression_factor = 4,
        bias = False,
    )
    model = GPT(config)
    
    # Test forward pass
    B, T = 2, 64
    x = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T, config.compression_factor))
    
    try:
        logits, loss, recon_loss = model(x, targets)
        print("Forward pass successful!")
        print(f"Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}")
        
        # Test generation
        print("Testing generation...")
        y = model.generate(x[:, :10], max_new_tokens=10)
        print(f"Generated sequence shape: {y.shape}")
        assert y.shape == (B, 10 + 10)
        print("Generation successful!")
        
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
