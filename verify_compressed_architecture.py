
import torch
from model_exp import GPT, GPTConfig

def test_model():
    config = GPTConfig(
        block_size = 128,
        vocab_size = 65,
        n_layer = 2,
        n_head = 4,
        n_embd = 64,
        compression_factor = 16,
        bias = False,
    )
    model = GPT(config)
    
    # Input: (B, T)
    B, T = 2, 64
    idx = torch.randint(0, config.vocab_size, (B, T))
    # Targets should be (B, T, 16) for MTP
    targets = torch.randint(0, config.vocab_size, (B, T, config.compression_factor))
    
    print(f"Input shape: {idx.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Set to eval to avoid print spam if any
    model.eval()
    with torch.no_grad():
        logits, loss, recon_loss = model(idx, targets)
        
    print(f"Logits shape: {logits if logits is None else logits.shape}")
    print(f"Loss: {loss.item() if loss is not None else 'None'}")
    print(f"Recon Loss: {recon_loss.item() if recon_loss is not None else 'None'}")
    
    if loss is not None:
        print("Forward pass successful with loss calculation.")
    
    # Check generation
    print("Testing generation...")
    generated = model.generate(idx, max_new_tokens=4, temperature=1.0)
    print(f"Generated shape: {generated.shape}")
    assert generated.shape == (B, T + 4)
    print("Verification successful!")

if __name__ == "__main__":
    test_model()
