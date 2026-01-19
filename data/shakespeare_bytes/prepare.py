import os
import numpy as np

# Download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    import requests
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'rb') as f:
    data = f.read()
print(f"length of dataset in bytes: {len(data):,}")

# Vocab size for bytes is always 256
vocab_size = 256
print(f"vocab size: {vocab_size}")

# Byte tokens are just the values 0-255
# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Convert to numpy arrays
train_ids = np.frombuffer(train_data, dtype=np.uint8).astype(np.uint16)
val_ids = np.frombuffer(val_data, dtype=np.uint8).astype(np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well
import pickle
meta = {
    'vocab_size': vocab_size,
    # No need for stoi/itos for bytes, can just use chr() and ord() or equivalent
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
