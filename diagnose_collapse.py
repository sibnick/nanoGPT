import torch
import torch.nn.functional as F
import os
from model_exp import GPT, GPTConfig

def diagnostic():
    out_dir = 'out-shakespeare-char-cmp16'
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}")
        return

    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    # Check embeddings
    w = model.get_normalized_wte()
    # Compute similarity matrix
    sim = torch.matmul(w, w.t())
    
    print(f"Vocab size: {config.vocab_size}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Average similarity between different tokens: {(sim.sum() - config.vocab_size) / (config.vocab_size * (config.vocab_size - 1))}")
    print(f"Max similarity: {sim.max().item()}")
    print(f"Min similarity: {sim.min().item()}")
    print(f"Diversity (std of row averages): {sim.mean(dim=1).std().item()}")

    # Check one row of similarity
    print("\nSimilarity of first 5 tokens with all others (avg):")
    for i in range(5):
        print(f"Token {i}: {sim[i].mean().item():.4f}")

    if sim.mean() > 0.9:
        print("\n[WARNING] EMBEDDING COLLAPSE DETECTED: Most embeddings are nearly identical.")
    else:
        print("\nEmbeddings seem diverse.")

if __name__ == "__main__":
    diagnostic()
