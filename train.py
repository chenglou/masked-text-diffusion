import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Set visible CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

from config import Config
from model import MaskedDiffusionTransformer
from data import load_shakespeare
from diffusion import MaskedDiffusion

def train():
    config = Config()
    
    print("Loading data...")
    train_dataset, val_dataset, tokenizer = load_shakespeare(config.block_size)
    config.vocab_size = tokenizer.vocab_size
    config.mask_token_id = tokenizer.vocab_size - 1  # Last token is mask
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"Vocab size: {config.vocab_size}")
    print(f"Train samples: {len(train_dataset)}")
    
    print("Initializing model...")
    model = MaskedDiffusionTransformer(config).cuda()
    diffusion = MaskedDiffusion(config, tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    scaler = torch.amp.GradScaler('cuda')
    iter_num = 0
    best_val_loss = float('inf')
    
    print("Starting training...")
    model.train()
    
    for epoch in range(100):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            if iter_num >= config.max_iters:
                break
            
            batch = batch.cuda()
            t = torch.randint(0, config.diffusion_steps, (batch.shape[0],), device='cuda')
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                loss = diffusion.compute_loss(model, batch, t)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Evaluate and save
            if iter_num % config.eval_interval == 0:
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        if len(val_losses) >= config.eval_iters:
                            break
                        val_batch = val_batch.cuda()
                        t = torch.randint(0, config.diffusion_steps, (val_batch.shape[0],), device='cuda')
                        val_loss = diffusion.compute_loss(model, val_batch, t)
                        val_losses.append(val_loss.item())
                
                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f"\nIter {iter_num}: train loss {loss.item():.4f}, val loss {avg_val_loss:.4f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }, 'best_model.pt')
                    print(f"Saved best model with val loss {best_val_loss:.4f}")
                
                # Generate sample
                print("Generating sample...")
                sample = diffusion.sample(model, (1, config.block_size), 'cuda', temperature=0.8)
                text = tokenizer.decode(sample[0].cpu().numpy())
                print(f"Generated: {text[:200]}...")
                print("-" * 50)
                
                model.train()
            
            iter_num += 1
            
        if iter_num >= config.max_iters:
            break
    
    print("Training complete!")
    return model, tokenizer, diffusion

if __name__ == "__main__":
    model, tokenizer, diffusion = train()