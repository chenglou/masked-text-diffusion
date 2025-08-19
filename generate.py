import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

from config import Config
from model import MaskedDiffusionTransformer
from data_tinystories import load_tinystories
from diffusion import MaskedDiffusion

def visualize_generation(model, diffusion, tokenizer, config):
    """Show the denoising process step by step"""

    # Start with all masks
    x = torch.full((1, config.block_size), diffusion.mask_token_id, device='cuda', dtype=torch.long)

    print("\nDenoising process:")
    print("=" * 80)

    # Show 10 evenly spaced steps throughout the denoising process
    step_interval = config.diffusion_steps // 10
    steps_to_show = [i * step_interval for i in range(10)]
    steps_to_show.append(config.diffusion_steps - 1)  # Always show the final step

    for t in reversed(range(config.diffusion_steps)):
        timesteps = torch.full((1,), t, device='cuda', dtype=torch.long)

        with torch.no_grad():
            logits = model(x, timesteps)

        mask = (x == diffusion.mask_token_id)

        if mask.any():
            # Unmask tokens progressively
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

        # Show progress at specific steps
        if config.diffusion_steps - t - 1 in steps_to_show:
            # Create text with [M] for mask tokens
            tokens_cpu = x[0].cpu().numpy()
            text_parts = []
            for token in tokens_cpu:
                if token == diffusion.mask_token_id:
                    text_parts.append('[M]')
                else:
                    # For BPE tokenizer, decode individual tokens
                    decoded = tokenizer.decode([token])
                    text_parts.append(decoded)
            text = ''.join(text_parts)

            masked_count = (x == diffusion.mask_token_id).sum().item()
            # Show a mix of beginning and random middle section to see unmasking
            if masked_count > 100:
                # When heavily masked, show just the beginning
                print(f"Step {config.diffusion_steps - t - 1:3d} ({masked_count:3d} masked): {text[:100]}...")
            else:
                # When less masked, show more text to see the actual words
                print(f"Step {config.diffusion_steps - t - 1:3d} ({masked_count:3d} masked): {text[:150]}")

    final_text = tokenizer.decode(x[0].cpu().numpy())
    print("=" * 80)
    print("\nFinal text:")
    print(final_text)

    return final_text

def main():
    config = Config()

    print("Loading data...")
    # Just load tokenizer, don't need the full dataset for generation
    _, _, tokenizer = load_tinystories(config.block_size, max_train_tokens=1000)
    config.vocab_size = tokenizer.vocab_size
    config.mask_token_id = tokenizer.vocab_size - 1

    print("Loading model...")
    model = MaskedDiffusionTransformer(config).cuda()

    checkpoint_path = 'best_model.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from iteration {checkpoint['iter_num']} with val loss {checkpoint['best_val_loss']:.4f}")
    else:
        print("No checkpoint found, using untrained model")

    model.eval()
    diffusion = MaskedDiffusion(config, tokenizer)

    # Show the denoising process
    print("\n" + "="*80)
    print("VISUALIZING DIFFUSION PROCESS")
    print("="*80)
    visualize_generation(model, diffusion, tokenizer, config)

    # Generate more samples with different temperatures
    print("\n" + "="*80)
    print("GENERATING MORE SAMPLES")
    print("="*80)

    for temp in [0.7, 0.85, 1.0]:
        print(f"\nTemperature {temp}:")
        with torch.no_grad():
            sample = diffusion.sample(model, (1, config.block_size), 'cuda', temperature=temp)
        text = tokenizer.decode(sample[0].cpu().numpy())
        print(text)

if __name__ == "__main__":
    main()
