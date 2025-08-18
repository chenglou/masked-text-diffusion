import torch
import torch.nn.functional as F
import numpy as np

class MaskedDiffusion:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else config.mask_token_id
        self.diffusion_steps = config.diffusion_steps
        
    def get_noise_schedule(self, t, total_steps):
        # Cosine schedule - smoother transition that preserves more structure
        # This schedule starts slow, accelerates in the middle, and slows down at the end
        s = 0.008  # small offset to prevent singularities
        
        # Map t to [0, 1] range
        ratio = t / total_steps
        
        # Cosine schedule formula from improved DDPM paper
        # Adapted for discrete masking instead of continuous noise
        numerator = np.cos((ratio + s) / (1 + s) * np.pi / 2) ** 2
        denominator = np.cos(s / (1 + s) * np.pi / 2) ** 2
        
        # Invert because we want masking probability to increase with t
        mask_prob = 1 - (numerator / denominator)
        
        # Ensure we're in valid probability range
        return np.clip(mask_prob, 0.0, 0.999)
    
    def add_noise(self, x, t):
        batch_size = x.shape[0]
        device = x.device
        
        mask_probs = torch.tensor([
            self.get_noise_schedule(timestep.item(), self.diffusion_steps) 
            for timestep in t
        ], device=device).unsqueeze(1)
        
        rand = torch.rand_like(x, dtype=torch.float32)
        mask = rand < mask_probs
        
        noisy_x = x.clone()
        noisy_x[mask] = self.mask_token_id
        
        return noisy_x, mask
    
    def compute_loss(self, model, x, t):
        noisy_x, mask = self.add_noise(x, t)
        
        logits = model(noisy_x, t)
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            x.view(-1),
            reduction='none'
        )
        
        loss = loss.view(x.shape)
        loss = (loss * mask.float()).sum() / mask.float().sum()
        
        return loss
    
    @torch.no_grad()
    def sample(self, model, shape, device, temperature=1.0):
        batch_size, seq_len = shape
        
        x = torch.full(shape, self.mask_token_id, device=device, dtype=torch.long)
        
        for t in reversed(range(self.diffusion_steps)):
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            logits = model(x, timesteps)
            
            mask = (x == self.mask_token_id)
            
            # Original simple schedule - guarantees completion
            num_to_unmask = max(1, int(mask.float().sum() / (t + 1)))
            
            probs = F.softmax(logits / temperature, dim=-1)
            
            confidence = probs.max(dim=-1).values
            confidence[~mask] = -float('inf')
            
            flat_confidence = confidence.view(-1)
            unmask_indices = flat_confidence.topk(num_to_unmask).indices
            
            flat_x = x.view(-1)
            flat_probs = probs.view(-1, probs.size(-1))
            
            for idx in unmask_indices:
                if flat_x[idx] == self.mask_token_id:
                    flat_x[idx] = torch.multinomial(flat_probs[idx], 1)
            
            x = flat_x.view(batch_size, seq_len)
            
            if not mask.any():
                break
        
        return x