import torch, json, os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scheduler.readability_sampler import CurriculumSampler
from model.model import GPT2Simple
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import numpy as np

# Disable distributed training to avoid initialization errors
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

# Create checkpoints directory if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)

def collate_fn(batch):
    """Custom collate function to pad sequences to the same length within a batch"""
    # Separate x and y from the batch
    x_batch, y_batch = zip(*batch)
    
    # Find the maximum length in this batch
    max_len = max(len(x) for x in x_batch)
    
    # Pad sequences to max_len
    x_padded = []
    y_padded = []
    
    for x, y in zip(x_batch, y_batch):
        # Pad x
        if len(x) < max_len:
            x_pad = torch.cat([x, torch.zeros(max_len - len(x), dtype=x.dtype)])
        else:
            x_pad = x
        x_padded.append(x_pad)
        
        # Pad y
        if len(y) < max_len:
            y_pad = torch.cat([y, torch.zeros(max_len - len(y), dtype=y.dtype)])
        else:
            y_pad = y
        y_padded.append(y_pad)
    
    # Stack the padded tensors
    x_batch = torch.stack(x_padded)
    y_batch = torch.stack(y_padded)
    
    return x_batch, y_batch

class KidDataset(Dataset):
    def __init__(self, path, tok, block_size=512):
        self.data, self.grades = [], []
        for ln in open(path):
            o=json.loads(ln)
            tokens=tok.encode(f"<s_grade={o['grade']}> "+o['text']).ids
            if len(tokens)>block_size: tokens=tokens[:block_size]
            self.data.append(tokens); self.grades.append(o['grade'])
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        x=self.data[i]; return torch.tensor(x[:-1]), torch.tensor(x[1:])

def main():
    # Check if CUDA is available and set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load tokenizer and dataset
    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    ds = KidDataset("data/data.jsonl", tok)
    
    # Print dataset statistics
    print(f"Dataset size: {len(ds)} samples")
    print(f"Vocabulary size: {tok.get_vocab_size()}")
    
    # Analyze data distribution
    grade_counts = {}
    total_tokens = 0
    for i in range(len(ds)):
        x, y = ds[i]
        grade = ds.grades[i]
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
        total_tokens += len(x)
    
    print(f"Total tokens in dataset: {total_tokens:,}")
    print(f"Average tokens per sample: {total_tokens / len(ds):.1f}")
    print("Grade distribution:")
    for grade in sorted(grade_counts.keys()):
        print(f"  Grade {grade}: {grade_counts[grade]} samples")
    
    # Create sampler and dataloader
    sampler = CurriculumSampler(ds.grades, epoch=0, total_epochs=10)
    dl = DataLoader(ds, batch_size=4, sampler=sampler, collate_fn=collate_fn)
    
    # Create model
    model = GPT2Simple(tok.get_vocab_size()).to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer and scaler
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    # Loss tracking
    all_losses = []
    epoch_losses = []

    for ep in range(10):
        sampler = CurriculumSampler(ds.grades, ep, 10)
        dl = DataLoader(ds, batch_size=4, sampler=sampler, collate_fn=collate_fn)
        
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (xb, yb) in enumerate(dl):
            xb, yb = xb.to(device), yb.to(device)
            
            # Print batch statistics occasionally
            if batch_idx % 50 == 0:
                print(f"Epoch {ep}, Batch {batch_idx}: xb shape {xb.shape}, yb shape {yb.shape}")
                print(f"  Sample tokens: {xb[0][:10].tolist()}")
                print(f"  Target tokens: {yb[0][:10].tolist()}")
            
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), yb.view(-1))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xb)
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), yb.view(-1))
                loss.backward()
                opt.step()
            
            opt.zero_grad()
            
            # Track losses
            all_losses.append(loss.item())
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_epoch_loss = epoch_loss / batch_count
        epoch_losses.append(avg_epoch_loss)
        
        print(f"Epoch {ep} - Avg loss: {avg_epoch_loss:.4f} | max_grade: {sampler.max_grade} | batches: {batch_count}")
        
        # Save model after each epoch
        torch.save(model.state_dict(), f"checkpoints/kidgpt_epoch_{ep}.pt")
        # Also save the latest model as kidgpt.pt for easy access
        torch.save(model.state_dict(), "kidgpt.pt")
        print(f"Model saved: checkpoints/kidgpt_epoch_{ep}.pt and kidgpt.pt")
        
        # Plot loss every few epochs
        if ep % 2 == 0 or ep == 9:
            plt.figure(figsize=(12, 4))
            
            # Plot all batch losses
            plt.subplot(1, 2, 1)
            plt.plot(all_losses)
            plt.title('Loss per Batch')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.yscale('log')
            
            # Plot epoch averages
            plt.subplot(1, 2, 2)
            plt.plot(epoch_losses)
            plt.title('Average Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig(f'checkpoints/loss_plot_epoch_{ep}.png')
            plt.close()
            print(f"Loss plot saved: checkpoints/loss_plot_epoch_{ep}.png")

    # Save the trained model
    print("Training completed! Saving model...")
    torch.save(model.state_dict(), "kidgpt.pt")
    print("Model saved to kidgpt.pt")

if __name__ == "__main__":
    main()
