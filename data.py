import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import urllib.request
import os
import tiktoken

class BPETokenizer:
    """GPT-2 BPE tokenizer wrapper with added [MASK] token"""
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        # Add one token for [MASK] beyond GPT-2's vocabulary
        self.vocab_size = self.enc.n_vocab + 1  # 50258 (50257 + 1)
        self.mask_token_id = 50257  # New token after GPT-2's vocab
        self.itos = {}  # For compatibility with old code

    def encode(self, text):
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens):
        # Filter out mask tokens before decoding
        filtered_tokens = [t for t in tokens if t != self.mask_token_id]
        if not filtered_tokens:
            return ""
        return self.enc.decode(filtered_tokens)

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size]
        x = torch.tensor(chunk, dtype=torch.long)
        return x

def load_shakespeare(block_size=128):
    data_path = 'shakespeare.txt'

    if not os.path.exists(data_path):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(data_path, 'w') as f:
            f.write(text_data)

    with open(data_path, 'r') as f:
        text = f.read()

    # Use BPE tokenizer
    tokenizer = BPETokenizer()
    data = tokenizer.encode(text)

    print(f"BPE tokenization stats:")
    print(f"  Characters: {len(text):,}")
    print(f"  BPE tokens: {len(data):,}")
    print(f"  Compression: {len(text)/len(data):.2f}x")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)

    return train_dataset, val_dataset, tokenizer
