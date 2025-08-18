# Masked Text Diffusion Implementation Notes

## Overview
We built a minimal masked text diffusion model from scratch, inspired by nanoGPT's simplicity. This implements discrete diffusion for text generation using progressive token masking/unmasking rather than continuous noise.

## Architecture Decisions

### Core Model: Bidirectional Transformer
- **6 layers, 6 heads, 384 dim** (~11M parameters)
- **Why bidirectional**: Unlike GPT (autoregressive), masked diffusion needs to see context from both directions to predict masked tokens
- **Timestep conditioning**: Added sinusoidal time embeddings so model knows corruption level

### Diffusion Process
- **Forward process**: Progressively mask tokens according to schedule (clean text → [MASK] tokens)
- **Reverse process**: Learn to predict original tokens from partially masked sequences
- **Key insight**: Treating text corruption as discrete diffusion where tokens transition to [MASK] states

### Noise Schedule
We implemented a **cosine schedule** for masking probability:
```python
def get_noise_schedule(self, t, total_steps):
    s = 0.008  # Small offset to prevent singularities
    ratio = t / total_steps
    numerator = np.cos((ratio + s) / (1 + s) * np.pi / 2) ** 2
    denominator = np.cos(s / (1 + s) * np.pi / 2) ** 2
    mask_prob = 1 - (numerator / denominator)
    return np.clip(mask_prob, 0.0, 0.999)
```
- Creates S-curve: slow start → fast middle → slow end
- Clipped at 0.999 to avoid completely masked sequences

## Training Insights

### Batch Size Discovery
Initially tried batch_size=128, but performance suffered. Key learning:
- **NOT about timestep diversity** (we were wrong initially!)
- Each sample in batch gets its own random timestep `t`
- Real issue: larger batches = smoother gradients = slower learning
- **Optimal**: batch_size=64 balances stability and learning speed

### Timesteps and Iterations
- **Diffusion is different from autoregressive**:
  - Autoregressive: Learn one task (predict next token)
  - Diffusion: Learn T tasks (denoise at each timestep)
- **Epochs don't matter**: We care about seeing diverse (batch, timestep) combinations
- Training for 100k+ iterations ensures good coverage of all timesteps

### Loss Patterns
- Loss varies dramatically by timestep:
  - t≈0: Easy (few masks), loss ~0.5-2.0
  - t≈500: Medium difficulty, loss ~2-4
  - t≈999: Hard (mostly masked), loss ~4-6
- Overall loss of 2.0-2.5 indicates good performance

## Generation Strategy

### Unmasking Schedule
The critical generation formula:
```python
num_to_unmask = max(1, int(mask.float().sum() / (t + 1)))
```

This creates an interesting pattern:
- Early steps (t=99→80): Unmask ~1 token per step
- Middle steps (t=50→20): Unmask 1-2 tokens per step  
- Final steps (t=10→0): Accelerate dramatically
- **Problem**: Big jump at t=0 (unmask all remaining)

### The Final Jump Issue
With 100 timesteps and 128 tokens:
- Step 98→1: Gradual unmasking
- Step 0: Must unmask ~30-40 tokens at once
- This causes quality degradation ("doubg", "hhas")

**Solutions explored**:
1. Cap unmasking rate → Doesn't complete generation ❌
2. More timesteps (300) → Smoother but 3x slower ✓
3. Accept the jump → Simple and works ✓

## Key Parameters Evolution

### Initial Setup (100 timesteps)
```python
diffusion_steps = 100
batch_size = 64
learning_rate = 3e-4
max_iters = 105000
```
- Fast generation but large final jump
- Good for experimentation

### Improved Setup (300 timesteps)
```python
diffusion_steps = 300  # Smoother unmasking
max_iters = 300000     # More training for better quality
```
- 3x slower generation but much smoother
- Better final quality

## Results and Quality Progression

### Training Milestones
| Iterations | Val Loss | Quality | Example |
|------------|----------|---------|---------|
| 50k | ~3.0 | Learning structure | Random characters with some patterns |
| 89k | 2.31 | Good structure, poor spelling | "GLOUCISTER:", dialogue format |
| 97k | 2.29 | Better vocabulary | "Second Keeper:", mostly real words |
| 150k+ | <2.0 | Target: Fluent Shakespeare | Correct spelling, coherent sentences |

### What the Model Learned
1. **Structure** (Excellent by 90k iters):
   - Character names with colons
   - Dialogue format
   - Line breaks and scenes

2. **Vocabulary** (Good by 100k iters):
   - Shakespeare-like words: "viole", "master", "dole"
   - Character names: "Gloucester", "Keeper"
   - Some spelling issues persist

3. **Coherence** (Needs more training):
   - Local coherence good (phrases make sense)
   - Long-range coherence needs improvement

## Sampling vs Training Decoupling

**Critical insight**: Training and sampling are independent!
- **Training**: Learn p(x₀|xₜ, t) for all t
- **Sampling**: Choose how to use those predictions

We can experiment with different unmasking strategies without retraining:
- Linear schedule
- Confidence-based adaptive rates
- Temperature scaling
- Top-k filtering

## Code Simplifications Made

1. **Removed all conditionals**: Assumed CUDA, float16
2. **Hardcoded device settings**: GPUs 0,1,2
3. **Simplified data loading**: Just Shakespeare
4. **Minimal dependencies**: Only PyTorch, numpy, tqdm

## Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Our Masked Diffusion** | Bidirectional context, parallel generation, controllable | Less explored than AR |
| **GPT-style (Autoregressive)** | Well understood, lots of research | Sequential only, no bidirectional |
| **Continuous Diffusion (D3PM)** | Closer to image diffusion | Complex for discrete data |
| **BERT-style (Masked LM)** | Similar architecture | No iterative refinement |

## Future Improvements

1. **Importance sampling**: Weight harder timesteps during training
2. **Learned noise schedule**: Let model learn optimal masking rates
3. **Conditional generation**: Add prompt conditioning
4. **Better unmasking**: Adaptive strategies based on confidence
5. **Scale up**: Bigger model, more data (enwik8)

## Key Takeaways

1. **Masked diffusion works for text** - We can adapt image diffusion ideas to discrete domains
2. **Simplicity wins** - Our minimal implementation is fully functional
3. **Batch size matters differently** - Not about timestep diversity but gradient dynamics
4. **Training/sampling decoupling** - Can improve generation without retraining
5. **The final jump is inherent** - Trade-off between completion guarantee and quality

## Files Overview

- `config.py` - All hyperparameters in one place
- `model.py` - Bidirectional transformer with time conditioning  
- `diffusion.py` - Noise schedule and sampling logic
- `data.py` - Character tokenizer and Shakespeare loader
- `train.py` - Training loop with validation
- `generate.py` - Visualization and generation scripts

Total: ~500 lines of clean, educational code demonstrating masked text diffusion from scratch!