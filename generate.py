import torch
import matplotlib.pyplot as plt
import numpy as np
from config import Config
from model import MaskedDiffusionTransformer
from data import load_shakespeare
from diffusion import MaskedDiffusion

def visualize_diffusion_process(model, diffusion, tokenizer, config, prompt=None):
    """Visualize the denoising process step by step"""
    device = config.device
    
    if prompt:
        tokens = tokenizer.encode(prompt)[:config.block_size]
        if len(tokens) < config.block_size:
            tokens += [0] * (config.block_size - len(tokens))
        x = torch.tensor([tokens], device=device)
    else:
        x = torch.full((1, config.block_size), diffusion.mask_token_id, device=device, dtype=torch.long)
    
    print("Starting generation process...")
    print("=" * 50)
    
    steps_to_show = [0, 100, 250, 500, 750, 999]
    samples = []
    
    for t in reversed(range(config.diffusion_steps)):
        timesteps = torch.full((1,), t, device=device, dtype=torch.long)
        
        with torch.no_grad():
            logits = model(x, timesteps)
        
        mask = (x == diffusion.mask_token_id)
        
        if mask.any():
            num_to_unmask = max(1, int(mask.float().sum() / (t + 1)))
            
            probs = torch.softmax(logits / 0.8, dim=-1)
            confidence = probs.max(dim=-1).values
            confidence[~mask] = -float('inf')
            
            flat_confidence = confidence.view(-1)
            unmask_indices = flat_confidence.topk(min(num_to_unmask, mask.sum())).indices
            
            flat_x = x.view(-1)
            flat_probs = probs.view(-1, probs.size(-1))
            
            for idx in unmask_indices:
                if flat_x[idx] == diffusion.mask_token_id:
                    flat_x[idx] = torch.multinomial(flat_probs[idx], 1)
            
            x = flat_x.view(1, config.block_size)
        
        if config.diffusion_steps - t - 1 in steps_to_show:
            text = tokenizer.decode(x[0].cpu().numpy())
            text = text.replace(tokenizer.itos.get(diffusion.mask_token_id, ''), '[M]')
            samples.append((config.diffusion_steps - t - 1, text))
            print(f"Step {config.diffusion_steps - t - 1:4d}: {text[:100]}...")
    
    final_text = tokenizer.decode(x[0].cpu().numpy())
    print("=" * 50)
    print("Final generation:")
    print(final_text)
    
    return samples, final_text

def plot_noise_schedule(diffusion, config):
    """Plot the noise schedule"""
    steps = np.arange(config.diffusion_steps)
    mask_probs = [diffusion.get_noise_schedule(t, config.diffusion_steps) for t in steps]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, mask_probs, linewidth=2)
    plt.xlabel('Diffusion Step', fontsize=12)
    plt.ylabel('Mask Probability', fontsize=12)
    plt.title('Cosine Noise Schedule for Masked Text Diffusion', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('noise_schedule.png', dpi=150)
    plt.show()
    print("Noise schedule plot saved to noise_schedule.png")

def main():
    config = Config()
    
    print("Loading tokenizer...")
    _, _, tokenizer = load_shakespeare(config.block_size)
    config.vocab_size = tokenizer.vocab_size
    
    print("Loading model...")
    model = MaskedDiffusionTransformer(config)
    
    checkpoint_path = 'best_model.pt'
    if torch.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from iteration {checkpoint['iter_num']} with val loss {checkpoint['best_val_loss']:.4f}")
    else:
        print("No checkpoint found, using untrained model")
    
    model = model.to(config.device)
    model.eval()
    
    diffusion = MaskedDiffusion(config, tokenizer)
    
    plot_noise_schedule(diffusion, config)
    
    print("\n" + "=" * 50)
    print("Generating text with diffusion process visualization...")
    print("=" * 50 + "\n")
    
    samples, final_text = visualize_diffusion_process(model, diffusion, tokenizer, config)
    
    print("\n" + "=" * 50)
    print("Generating another sample...")
    print("=" * 50 + "\n")
    
    with torch.no_grad():
        sample = diffusion.sample(model, (1, config.block_size), config.device, temperature=1.0)
    generated_text = tokenizer.decode(sample[0].cpu().numpy())
    print(generated_text)

if __name__ == "__main__":
    main()