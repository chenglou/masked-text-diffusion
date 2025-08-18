# Masked Text Diffusion Implementation Notes

## Overview
We built a minimal masked text diffusion model from scratch, inspired by nanoGPT's simplicity. This implements discrete diffusion for text generation using progressive token masking/unmasking rather than continuous noise.

## Current Architecture (v2.0 - BPE + Modern Improvements)

### Core Model: Bidirectional Transformer with RoPE
- **6 layers, 6 heads, 384 dim** (~49M parameters with BPE vocabulary)
- **Rotary Position Embeddings (RoPE)**: No learned position embeddings, better extrapolation
- **Flash Attention**: 2-4x faster training, memory efficient
- **bfloat16**: No gradient scaler needed, cleaner code
- **BPE Tokenization**: GPT-2's tokenizer with 50,258 tokens (including [MASK])

### Tokenization Evolution

| Version | Type | Vocab Size | Sequence Meaning | Model Size |
|---------|------|------------|------------------|------------|
| v1.0 | Character | 66 | 128 characters | 11M params |
| v2.0 | BPE | 50,258 | 128 tokens (~512 chars) | 49M params |

### Diffusion Process
- **Forward process**: Progressively mask tokens according to cosine schedule
- **Reverse process**: Learn to predict original tokens from partially masked sequences
- **Key insight**: Treating text corruption as discrete diffusion where tokens transition to [MASK] states

### Noise Schedule
Cosine schedule for masking probability:
```python
def get_noise_schedule(self, t, total_steps):
    s = 0.008  # Small offset to prevent singularities
    ratio = t / total_steps
    numerator = np.cos((ratio + s) / (1 + s) * np.pi / 2) ** 2
    denominator = np.cos(s / (1 + s) * np.pi / 2) ** 2
    mask_prob = 1 - (numerator / denominator)
    return np.clip(mask_prob, 0.0, 0.999)
```

## Major Architectural Improvements

### 1. RoPE (Rotary Position Embeddings)
- **What**: Encodes position through rotation of embedding vectors
- **Why**: Better than learned embeddings - can extrapolate, no parameters, relative positions
- **Impact**: Immediate improvement in loss convergence

### 2. Flash Attention
- **What**: Fused kernel that never materializes full attention matrix
- **Why**: 2-4x faster, 50% memory savings
- **Implementation**: Single line change using `F.scaled_dot_product_attention`

### 3. bfloat16 Precision
- **What**: Brain floating point - same range as float32, less precision
- **Why**: No gradient scaler needed, prevents overflow
- **Code simplification**: Removed 5 lines of scaler management

### 4. BPE Tokenization
- **What**: Byte-Pair Encoding using GPT-2's vocabulary
- **Why**: Semantic understanding vs character spelling
- **Compression**: 3.3x fewer tokens for same text
- **Challenge**: 78% of parameters now in embeddings

## Training Insights

### Character vs BPE Loss Scales
- **Character-level**: Loss ~2.2-2.3 (choosing from 66 options)
- **BPE-level**: Loss ~3.5-4.0 (choosing from 50k options)
- Both represent similar perplexity when adjusted for vocabulary size

### Batch Size Discovery
- Larger batches don't help masked diffusion as much as expected
- Each sample gets random timestep - diversity matters more than batch size
- Optimal: 64 for character, might need 32 for BPE (memory constraints)

### Timestep Dynamics
- Model tends to unmask high-confidence tokens too early
- Results in "frozen" text that doesn't change much during generation
- Solution: Could add temperature or confidence smoothing to unmasking

### The Unmasking Schedule
```python
num_to_unmask = max(1, int(mask.float().sum() / (t + 1)))
```
This creates acceleration towards the end - sometimes unmasking 30+ tokens in final step.

## Current Performance (BPE Model)

After 16k iterations:
- **Train loss**: 4.0
- **Val loss**: 6.1
- **Generation**: Coherent Shakespeare dialogue with character names
- **Issues**: Some spelling mistakes ("KING HWARDRY VI"), overfitting gap

## Parameter Breakdown (49M Model)

```
Token embeddings:    50,258 × 384 = 19.3M
Time embeddings:     300 × 384    = 0.12M
Attention (×6):      3.5M
MLP (×6):           7.1M
Output head:         50,258 × 384 = 19.3M
-------------------------------------------
Total:              49.3M parameters
Embeddings:         78% of total parameters
```

## Key Design Decisions

### Why Bidirectional (not Autoregressive)?
- Masked diffusion needs context from both directions
- Can edit/refine anywhere in sequence
- Better for iterative generation

### Why 300 Diffusion Steps?
- Balance between generation quality and speed
- 100 steps: Too few, large jumps at end
- 1000 steps: Overkill for text, very slow
- 300 steps: Good compromise

### Why Dedicated [MASK] Token?
- Originally used <|endoftext|> (token 50256)
- Problem: Confuses semantic meaning
- Solution: Added token 50257 as dedicated [MASK]
- Vocabulary is now 50,258 tokens

## Files Overview

- `config.py` - Hyperparameters (updated for BPE)
- `model.py` - Transformer with RoPE and Flash Attention
- `diffusion.py` - Cosine noise schedule and sampling
- `data.py` - BPE tokenization using tiktoken
- `train.py` - Training loop with bfloat16
- `generate.py` - Visualization and generation

## Lessons Learned

1. **Position encoding matters**: RoPE immediately improved quality
2. **Tokenization is fundamental**: BPE completely changes model dynamics
3. **Modern optimizations compound**: Flash + bfloat16 + RoPE work together
4. **Embedding size dominates**: With large vocabulary, embeddings become the bottleneck
5. **Diffusion needs different thinking**: Timestep diversity > batch size

## Future Improvements to Consider

1. **SwiGLU activation**: Tested but minimal improvement at this scale
2. **Adaptive unmasking**: Prevent overconfident early reveals
3. **Importance sampling**: Weight harder timesteps during training
4. **Longer context**: block_size=256 for more coherent generation
5. **Scale model depth**: 8-12 layers might help with BPE tokens

## Training Recommendations

For BPE model:
- Train for 100k+ iterations (vs 50k for character model)
- Monitor validation gap - consider dropout=0.15 if overfitting
- Try batch_size=32 if memory limited
- Expect loss to plateau around 3.5-4.0 (this is good!)

## Comparison: Character vs BPE

| Metric | Character-level | BPE-level |
|--------|----------------|-----------|
| Model size | 11M | 49M |
| Vocab size | 66 | 50,258 |
| Tokens for "Hello" | 5 | 1 |
| Training speed | Fast | Slower |
| Generation quality | Letter-by-letter | Word-aware |
| Loss scale | ~2.2 | ~4.0 |
| Context window | 128 chars | ~512 chars |

The shift to BPE fundamentally changed the model from a "speller" to a "word predictor", requiring 5x more parameters but delivering much better semantic understanding.