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

def visualize_data_distribution(dataset, num_samples=100, max_display_samples=20):
    """
    Visualize the distribution of sequence lengths and padding in the dataset.
    
    Args:
        dataset: The KidDataset instance
        num_samples: Number of samples to analyze (for statistics)
        max_display_samples: Maximum number of samples to show in the heatmap
    """
    print(f"\n=== Data Distribution Analysis ===")
    
    # Collect sequence lengths
    sequence_lengths = []
    non_zero_counts = []
    
    # Analyze a subset of samples for statistics
    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in sample_indices:
        x, y = dataset[idx]
        seq_len = len(x)
        sequence_lengths.append(seq_len)
        
        # Count non-zero tokens (actual content vs padding)
        non_zero = torch.count_nonzero(x).item()
        non_zero_counts.append(non_zero)
    
    # Calculate statistics
    avg_length = np.mean(sequence_lengths)
    median_length = np.median(sequence_lengths)
    min_length = np.min(sequence_lengths)
    max_length = np.max(sequence_lengths)
    
    avg_content = np.mean(non_zero_counts)
    padding_ratio = 1 - (avg_content / avg_length) if avg_length > 0 else 0
    
    print(f"Sequence Length Statistics:")
    print(f"  Average length: {avg_length:.1f}")
    print(f"  Median length: {median_length:.1f}")
    print(f"  Min length: {min_length}")
    print(f"  Max length: {max_length}")
    print(f"  Average content tokens: {avg_content:.1f}")
    print(f"  Padding ratio: {padding_ratio:.2%}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Sequence length distribution
    axes[0, 0].hist(sequence_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(avg_length, color='red', linestyle='--', label=f'Mean: {avg_length:.1f}')
    axes[0, 0].axvline(median_length, color='orange', linestyle='--', label=f'Median: {median_length:.1f}')
    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Sequence Lengths')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Content vs padding ratio
    content_ratios = [content / length if length > 0 else 0 for content, length in zip(non_zero_counts, sequence_lengths)]
    axes[0, 1].hist(content_ratios, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(np.mean(content_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(content_ratios):.2%}')
    axes[0, 1].set_xlabel('Content Ratio (Non-zero tokens / Total tokens)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Content Ratios')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Sample heatmap (show actual token patterns)
    display_samples = min(max_display_samples, len(sample_indices))
    heatmap_data = []
    sample_lengths = []
    
    for i in range(display_samples):
        idx = sample_indices[i]
        x, y = dataset[idx]
        # Convert to numpy and pad to max length for visualization
        tokens = x.numpy()
        sample_lengths.append(len(tokens))
        heatmap_data.append(tokens)
    
    # Pad all samples to the same length for heatmap
    max_len = max(sample_lengths)
    padded_data = []
    for tokens in heatmap_data:
        if len(tokens) < max_len:
            # Pad with zeros
            padded = np.pad(tokens, (0, max_len - len(tokens)), mode='constant', constant_values=0)
        else:
            padded = tokens[:max_len]
        padded_data.append(padded)
    
    heatmap_array = np.array(padded_data)
    
    # Create heatmap where non-zero tokens are colored
    im = axes[1, 0].imshow(heatmap_array != 0, cmap='viridis', aspect='auto')
    axes[1, 0].set_xlabel('Token Position')
    axes[1, 0].set_ylabel('Sample Index')
    axes[1, 0].set_title(f'Token Presence Heatmap (First {display_samples} samples)')
    axes[1, 0].set_yticks(range(display_samples))
    axes[1, 0].set_yticklabels([f'Sample {i+1}' for i in range(display_samples)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 0])
    cbar.set_label('Token Present (1) / Padding (0)')
    
    # 4. Grade distribution vs sequence length
    grade_lengths = {}
    for idx in sample_indices:
        grade = dataset.grades[idx]
        x, y = dataset[idx]
        if grade not in grade_lengths:
            grade_lengths[grade] = []
        grade_lengths[grade].append(len(x))
    
    grades = sorted(grade_lengths.keys())
    avg_lengths_by_grade = [np.mean(grade_lengths[grade]) for grade in grades]
    
    bars = axes[1, 1].bar(grades, avg_lengths_by_grade, alpha=0.7, color='coral')
    axes[1, 1].set_xlabel('Grade')
    axes[1, 1].set_ylabel('Average Sequence Length')
    axes[1, 1].set_title('Average Sequence Length by Grade')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, avg_len in zip(bars, avg_lengths_by_grade):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{avg_len:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('checkpoints/data_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nData distribution analysis saved to: checkpoints/data_distribution_analysis.png")
    
    # Print recommendations
    print(f"\n=== Recommendations ===")
    if padding_ratio > 0.5:
        print(f"   WARNING: High padding ratio ({padding_ratio:.2%}) detected!")
        print(f"   Consider reducing block_size or implementing dynamic padding.")
    else:
        print(f"  Padding ratio ({padding_ratio:.2%}) looks reasonable.")
    
    if max_length / avg_length > 3:
        print(f"   WARNING: High variance in sequence lengths!")
        print(f"   Max length is {max_length/avg_length:.1f}x the average length.")
        print(f"   Consider using a smaller block_size or implementing bucketing.")
    
    return {
        'avg_length': avg_length,
        'padding_ratio': padding_ratio,
        'length_variance': max_length / avg_length
    }

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
