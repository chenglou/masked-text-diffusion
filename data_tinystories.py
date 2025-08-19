import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import tiktoken

class BPETokenizer:
    """GPT-2 BPE tokenizer wrapper with added [MASK] token"""
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        # Add one token for [MASK] beyond GPT-2's vocabulary
        self.vocab_size = self.enc.n_vocab + 1  # 50258 (50257 + 1)
        self.mask_token_id = 50257  # New token after GPT-2's vocab
        self.endoftext_id = self.enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]  # Should be 50256
        self.itos = {}  # For compatibility with old code

    def encode(self, text):
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens):
        # Filter out mask tokens before decoding
        filtered_tokens = [t for t in tokens if t != self.mask_token_id]
        if not filtered_tokens:
            return ""
        return self.enc.decode(filtered_tokens)

class StoryDataset(Dataset):
    """Dataset that respects story boundaries instead of using sliding windows."""
    def __init__(self, stories, block_size, tokenizer):
        """
        Args:
            stories: List of tokenized stories (each story is a list of token ids)
            block_size: Maximum sequence length
            tokenizer: Tokenizer instance for special tokens
        """
        self.stories = stories
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        # Filter stories by length and prepare them
        self.samples = []
        for story in stories:
            if len(story) <= block_size:
                # Pad shorter stories with endoftext tokens
                if len(story) < block_size:
                    # Pad with endoftext tokens at the end
                    padded = story + [tokenizer.endoftext_id] * (block_size - len(story))
                    self.samples.append(padded)
                else:
                    self.samples.append(story)
            else:
                # For longer stories, split into chunks but try to keep sentences together
                # Split at endoftext boundaries if possible
                for i in range(0, len(story), block_size):
                    chunk = story[i:i + block_size]
                    if len(chunk) < block_size:
                        # Pad the last chunk
                        chunk = chunk + [tokenizer.endoftext_id] * (block_size - len(chunk))
                    self.samples.append(chunk)
        
        print(f"  Created {len(self.samples)} samples from {len(stories)} stories")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)

def load_tinystories(block_size=128, max_train_tokens=None):
    """
    Load the TinyStories dataset with proper story boundaries.
    
    Args:
        block_size: sequence length for training
        max_train_tokens: optionally limit training data size for faster iteration
    """
    train_path = 'tinystories_train.txt'
    valid_path = 'tinystories_valid.txt'
    
    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        print("TinyStories dataset not found. Please run download_tinystories.py first.")
        raise FileNotFoundError("TinyStories dataset files not found")
    
    print("Loading TinyStories dataset...")
    
    # Use BPE tokenizer
    tokenizer = BPETokenizer()
    
    # Load and tokenize training data story by story
    print("Tokenizing training data (respecting story boundaries)...")
    
    train_stories = []
    total_train_tokens = 0
    
    with open(train_path, 'r', encoding='utf-8') as f:
        current_story = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Tokenize the line
            tokens = tokenizer.encode(line)
            current_story.extend(tokens)
            
            # Check if this line ends with endoftext
            if tokenizer.endoftext_id in tokens:
                # End of story, save it
                if current_story:
                    # Remove trailing endoftext as we'll add it back if needed for padding
                    while current_story and current_story[-1] == tokenizer.endoftext_id:
                        current_story.pop()
                    
                    if current_story:  # Only add non-empty stories
                        train_stories.append(current_story)
                        total_train_tokens += len(current_story)
                    current_story = []
                    
                    # Check if we've reached the token limit
                    if max_train_tokens and total_train_tokens >= max_train_tokens:
                        break
        
        # Don't forget the last story if it doesn't end with endoftext
        if current_story and (not max_train_tokens or total_train_tokens < max_train_tokens):
            train_stories.append(current_story)
            total_train_tokens += len(current_story)
    
    print(f"  Loaded {len(train_stories)} training stories with {total_train_tokens:,} tokens")
    
    # Load validation data
    print("Tokenizing validation data (respecting story boundaries)...")
    
    valid_stories = []
    total_valid_tokens = 0
    
    with open(valid_path, 'r', encoding='utf-8') as f:
        current_story = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            tokens = tokenizer.encode(line)
            current_story.extend(tokens)
            
            if tokenizer.endoftext_id in tokens:
                if current_story:
                    while current_story and current_story[-1] == tokenizer.endoftext_id:
                        current_story.pop()
                    if current_story:
                        valid_stories.append(current_story)
                        total_valid_tokens += len(current_story)
                    current_story = []
        
        if current_story:
            valid_stories.append(current_story)
            total_valid_tokens += len(current_story)
    
    print(f"  Loaded {len(valid_stories)} validation stories with {total_valid_tokens:,} tokens")
    
    # Create datasets
    train_dataset = StoryDataset(train_stories, block_size, tokenizer)
    val_dataset = StoryDataset(valid_stories, block_size, tokenizer)
    
    # Print statistics
    print(f"\nTinyStories dataset stats:")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print(f"  Block size: {block_size}")
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")
    print(f"  Average story length: {total_train_tokens/len(train_stories):.1f} tokens")
    
    # Show sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        decoded = tokenizer.decode(sample.numpy())
        print(f"\nSample story (truncated to 300 chars):")
        print(f"  {decoded[:300]}...")
    
    return train_dataset, val_dataset, tokenizer

# Backward compatibility
def load_shakespeare(block_size=128):
    """Backward compatibility wrapper."""
    return load_tinystories(block_size, max_train_tokens=10_000_000)

if __name__ == "__main__":
    # Test the data loader with story boundaries
    print("Testing TinyStories data loader with story boundaries...")
    train_dataset, val_dataset, tokenizer = load_tinystories(
        block_size=256,  # Larger block size for complete stories
        max_train_tokens=10_000_000  # 10M tokens for testing
    )