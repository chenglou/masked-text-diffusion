# Masked Text Diffusion

A minimal implementation of masked text diffusion models, inspired by nanoGPT's simplicity. This implements discrete diffusion for text generation using a transformer architecture.

## Architecture

- **Model**: Bidirectional transformer (6 layers, 6 heads, 384 dim)
- **Diffusion**: Progressive token masking with cosine schedule
- **Training**: Predict original tokens from partially masked sequences
- **Generation**: Iterative unmasking based on model confidence

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python train.py
```

This will:
- Download Shakespeare dataset automatically
- Train for 5000 iterations
- Save best model checkpoint
- Show sample generations during training

### Generate Text
```bash
python generate.py
```

This will:
- Load the trained model
- Visualize the noise schedule
- Show the denoising process step-by-step
- Generate sample text

## Key Components

- `config.py`: All hyperparameters in one place
- `model.py`: Transformer architecture with timestep conditioning
- `diffusion.py`: Noise schedule and diffusion utilities
- `data.py`: Character-level tokenizer and data loading
- `train.py`: Training loop with validation
- `generate.py`: Generation and visualization

## How It Works

1. **Forward Process**: Gradually mask tokens according to cosine schedule
2. **Training**: Learn to predict original tokens from masked sequences + timestep
3. **Sampling**: Start with all masks, iteratively unmask based on model confidence

## Customization

Adjust hyperparameters in `config.py`:
- `n_layer`, `n_head`, `n_embd`: Model size
- `diffusion_steps`: Number of denoising steps
- `learning_rate`, `batch_size`: Training settings
- `block_size`: Maximum sequence length
