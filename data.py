import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import urllib.request
import os

class CharTokenizer:
    def __init__(self):
        self.chars = None
        self.stoi = None
        self.itos = None
        self.vocab_size = None

    def fit(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars) + 1  # +1 for mask token
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.mask_token_id = len(chars)  # Last token is mask
        self.stoi['[MASK]'] = self.mask_token_id
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.chars = chars
        return self

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens if i in self.itos])

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

    tokenizer = CharTokenizer().fit(text)
    data = tokenizer.encode(text)

    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)

    return train_dataset, val_dataset, tokenizer
