import torch, json, os
import torch.nn as nn
from torch.utils.data import DataLoader
from scheduler.readability_sampler import CurriculumSampler
from model.model import GPT2Simple
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random

# Import from new modules
from dataset import KidDataset, collate_fn
from visualization import visualize_data_distribution, visualize_collate_effects

# Disable distributed training to avoid initialization errors
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

# Create checkpoints directory if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)

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
    
    # Visualize data distribution and padding
    data_stats = visualize_data_distribution(ds, num_samples=200, max_display_samples=25)
    
    # Analyze collate function effects
    collate_stats = visualize_collate_effects(ds, collate_fn, num_batches=15, batch_size=4)
        
    # Create model
    model = GPT2Simple(tok.get_vocab_size()).to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer and scaler
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Loss tracking
    all_losses = []
    epoch_losses = []

    for ep in range(10):
        # Use bucketed sampler for better efficiency
        sampler = BucketedSampler(ds, batch_size=4, num_buckets=10)
        dl = DataLoader(ds, batch_size=4, sampler=sampler, collate_fn=collate_fn_with_masking)
        
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, batch_data in enumerate(dl):
            # Handle new collate function output
            if len(batch_data) == 3:
                xb, yb, attention_mask = batch_data
            else:
                xb, yb = batch_data
                attention_mask = None
                
            xb, yb = xb.to(device), yb.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Print batch statistics occasionally
            if batch_idx % 50 == 0:
                print(f"Epoch {ep}, Batch {batch_idx}: xb shape {xb.shape}, yb shape {yb.shape}")
                if attention_mask is not None:
                    print(f"  Attention mask shape: {attention_mask.shape}")
                    print(f"  Active tokens per sample: {attention_mask.sum(dim=1).tolist()}")
                print(f"  Sample tokens: {xb[0][:10].tolist()}")
                print(f"  Target tokens: {yb[0][:10].tolist()}")
            
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = model(xb, attention_mask=attention_mask)
                    # Use attention mask if available
                    if attention_mask is not None:
                        # Create loss mask to ignore padding tokens
                        loss_mask = attention_mask.view(-1)
                        # Flatten logits and targets
                        logits_flat = logits.view(-1, logits.size(-1))
                        targets_flat = yb.view(-1)
                        # Apply mask to loss calculation
                        loss = nn.CrossEntropyLoss(reduction='none')(logits_flat, targets_flat)
                        loss = (loss * loss_mask).sum() / loss_mask.sum()
                    else:
                        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), yb.view(-1))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xb, attention_mask=attention_mask)
                # Use attention mask if available
                if attention_mask is not None:
                    # Create loss mask to ignore padding tokens
                    loss_mask = attention_mask.view(-1)
                    # Flatten logits and targets
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = yb.view(-1)
                    # Apply mask to loss calculation
                    loss = nn.CrossEntropyLoss(reduction='none')(logits_flat, targets_flat)
                    loss = (loss * loss_mask).sum() / loss_mask.sum()
                else:
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
        
        print(f"Epoch {ep} - Avg loss: {avg_epoch_loss:.4f} | batches: {batch_count}")
        
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
