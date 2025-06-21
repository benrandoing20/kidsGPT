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

def visualize_collate_effects(dataset, collate_fn, num_batches=10, batch_size=4):
    """
    Visualize how the collate function affects the data by showing padding effects.
    
    Args:
        dataset: The KidDataset instance
        collate_fn: The collate function to analyze
        num_batches: Number of batches to analyze
        batch_size: Batch size to use
    """
    print(f"\n=== Collate Function Analysis ===")
    
    # Create a simple dataloader to get batches
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Statistics tracking
    original_lengths = []
    padded_lengths = []
    padding_ratios = []
    batch_efficiency = []
    
    # Sample some batches for detailed analysis
    batch_samples = []
    
    for batch_idx, (xb, yb) in enumerate(dl):
        if batch_idx >= num_batches:
            break
            
        # Get original lengths before padding
        original_batch_lengths = []
        for i in range(batch_size):
            if i < len(dataset):
                x_orig, y_orig = dataset[i]
                original_batch_lengths.append(len(x_orig))
        
        # Calculate padding statistics
        batch_max_len = xb.shape[1]
        batch_padding_ratios = []
        
        for orig_len in original_batch_lengths:
            padding_len = batch_max_len - orig_len
            padding_ratio = padding_len / batch_max_len if batch_max_len > 0 else 0
            batch_padding_ratios.append(padding_ratio)
        
        # Store statistics
        original_lengths.extend(original_batch_lengths)
        padded_lengths.extend([batch_max_len] * len(original_batch_lengths))
        padding_ratios.extend(batch_padding_ratios)
        
        # Calculate batch efficiency (how much of the batch is actual content)
        batch_efficiency.append(1 - np.mean(batch_padding_ratios))
        
        # Store detailed batch info for visualization
        batch_samples.append({
            'batch_idx': batch_idx,
            'original_lengths': original_batch_lengths,
            'padded_length': batch_max_len,
            'padding_ratios': batch_padding_ratios,
            'efficiency': batch_efficiency[-1],
            'xb_shape': xb.shape,
            'yb_shape': yb.shape
        })
    
    # Calculate overall statistics
    avg_original_length = np.mean(original_lengths)
    avg_padded_length = np.mean(padded_lengths)
    avg_padding_ratio = np.mean(padding_ratios)
    avg_batch_efficiency = np.mean(batch_efficiency)
    
    print(f"Collate Function Statistics:")
    print(f"  Average original length: {avg_original_length:.1f}")
    print(f"  Average padded length: {avg_padded_length:.1f}")
    print(f"  Average padding ratio: {avg_padding_ratio:.2%}")
    print(f"  Average batch efficiency: {avg_batch_efficiency:.2%}")
    print(f"  Length expansion factor: {avg_padded_length/avg_original_length:.2f}x")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Before/After length comparison
    ax1 = plt.subplot(3, 3, 1)
    x_pos = np.arange(len(original_lengths))
    ax1.bar(x_pos - 0.2, original_lengths, 0.4, label='Original Length', alpha=0.7, color='skyblue')
    ax1.bar(x_pos + 0.2, padded_lengths, 0.4, label='Padded Length', alpha=0.7, color='lightcoral')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Sequence Length')
    ax1.set_title('Original vs Padded Lengths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Padding ratio distribution
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(padding_ratios, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(avg_padding_ratio, color='red', linestyle='--', label=f'Mean: {avg_padding_ratio:.2%}')
    ax2.set_xlabel('Padding Ratio')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Padding Ratios')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Batch efficiency over batches
    ax3 = plt.subplot(3, 3, 3)
    batch_indices = [b['batch_idx'] for b in batch_samples]
    ax3.plot(batch_indices, batch_efficiency, 'o-', color='green', linewidth=2, markersize=6)
    ax3.axhline(avg_batch_efficiency, color='red', linestyle='--', label=f'Mean: {avg_batch_efficiency:.2%}')
    ax3.set_xlabel('Batch Index')
    ax3.set_ylabel('Batch Efficiency')
    ax3.set_title('Batch Efficiency Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Detailed batch analysis heatmap
    ax4 = plt.subplot(3, 3, 4)
    # Create a heatmap showing padding for each sample in each batch
    max_samples = max(len(b['original_lengths']) for b in batch_samples)
    heatmap_data = np.zeros((max_samples, len(batch_samples)))
    
    for batch_idx, batch in enumerate(batch_samples):
        for sample_idx, padding_ratio in enumerate(batch['padding_ratios']):
            if sample_idx < max_samples:
                heatmap_data[sample_idx, batch_idx] = padding_ratio
    
    im = ax4.imshow(heatmap_data, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    ax4.set_xlabel('Batch Index')
    ax4.set_ylabel('Sample Index in Batch')
    ax4.set_title('Padding Ratio Heatmap by Batch')
    ax4.set_xticks(range(len(batch_samples)))
    ax4.set_xticklabels([f'B{i}' for i in range(len(batch_samples))])
    ax4.set_yticks(range(max_samples))
    ax4.set_yticklabels([f'S{i+1}' for i in range(max_samples)])
    
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Padding Ratio')
    
    # 5. Memory usage analysis
    ax5 = plt.subplot(3, 3, 5)
    original_memory = [sum(b['original_lengths']) for b in batch_samples]
    padded_memory = [b['padded_length'] * len(b['original_lengths']) for b in batch_samples]
    memory_efficiency = [orig / pad for orig, pad in zip(original_memory, padded_memory)]
    
    x_pos = np.arange(len(batch_samples))
    ax5.bar(x_pos - 0.2, original_memory, 0.4, label='Original Memory', alpha=0.7, color='lightblue')
    ax5.bar(x_pos + 0.2, padded_memory, 0.4, label='Padded Memory', alpha=0.7, color='lightpink')
    ax5.set_xlabel('Batch Index')
    ax5.set_ylabel('Total Tokens')
    ax5.set_title('Memory Usage: Original vs Padded')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Memory efficiency
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(batch_indices, memory_efficiency, 'o-', color='purple', linewidth=2, markersize=6)
    ax6.axhline(np.mean(memory_efficiency), color='red', linestyle='--', label=f'Mean: {np.mean(memory_efficiency):.2%}')
    ax6.set_xlabel('Batch Index')
    ax6.set_ylabel('Memory Efficiency')
    ax6.set_title('Memory Efficiency Over Batches')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Length distribution comparison
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(original_lengths, bins=20, alpha=0.5, label='Original', color='blue', density=True)
    ax7.hist(padded_lengths, bins=20, alpha=0.5, label='Padded', color='red', density=True)
    ax7.set_xlabel('Sequence Length')
    ax7.set_ylabel('Density')
    ax7.set_title('Length Distribution Comparison')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Batch size vs efficiency scatter
    ax8 = plt.subplot(3, 3, 8)
    batch_sizes = [len(b['original_lengths']) for b in batch_samples]
    batch_padding_ratios = [np.mean(b['padding_ratios']) for b in batch_samples]  # Use batch-level padding ratios
    ax8.scatter(batch_sizes, batch_efficiency, c=batch_padding_ratios, cmap='viridis', s=100, alpha=0.7)
    ax8.set_xlabel('Batch Size')
    ax8.set_ylabel('Batch Efficiency')
    ax8.set_title('Batch Size vs Efficiency')
    ax8.grid(True, alpha=0.3)
    
    # Add colorbar for padding ratio
    scatter = ax8.scatter([], [], c=[], cmap='viridis')
    cbar = plt.colorbar(scatter, ax=ax8)
    cbar.set_label('Average Padding Ratio per Batch')
    
    # 9. Summary statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    stats_text = f"""
    Collate Function Analysis Summary:
    
    Original Data:
    • Avg Length: {avg_original_length:.1f}
    • Min Length: {min(original_lengths)}
    • Max Length: {max(original_lengths)}
    
    After Padding:
    • Avg Length: {avg_padded_length:.1f}
    • Padding Ratio: {avg_padding_ratio:.2%}
    • Memory Expansion: {avg_padded_length/avg_original_length:.2f}x
    
    Efficiency:
    • Batch Efficiency: {avg_batch_efficiency:.2%}
    • Memory Efficiency: {np.mean(memory_efficiency):.2%}
    
    Recommendations:
    """
    
    if avg_padding_ratio > 0.5:
        stats_text += "⚠️ High padding ratio - consider dynamic padding"
    elif avg_padding_ratio > 0.3:
        stats_text += "⚠️ Moderate padding - consider bucketing"
    else:
        stats_text += "✅ Padding looks reasonable"
    
    if avg_padded_length/avg_original_length > 2:
        stats_text += "\n⚠️ High memory expansion - consider smaller block_size"
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('checkpoints/collate_function_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCollate function analysis saved to: checkpoints/collate_function_analysis.png")
    
    # Print detailed batch information
    print(f"\n=== Detailed Batch Analysis ===")
    for batch in batch_samples[:5]:  # Show first 5 batches
        print(f"Batch {batch['batch_idx']}:")
        print(f"  Original lengths: {batch['original_lengths']}")
        print(f"  Padded length: {batch['padded_length']}")
        print(f"  Padding ratios: {[f'{r:.2%}' for r in batch['padding_ratios']]}")
        print(f"  Efficiency: {batch['efficiency']:.2%}")
        print(f"  Shapes: xb={batch['xb_shape']}, yb={batch['yb_shape']}")
        print()
    
    return {
        'avg_original_length': avg_original_length,
        'avg_padded_length': avg_padded_length,
        'avg_padding_ratio': avg_padding_ratio,
        'avg_batch_efficiency': avg_batch_efficiency,
        'memory_expansion': avg_padded_length/avg_original_length
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
    
    # Analyze collate function effects
    collate_stats = visualize_collate_effects(ds, collate_fn, num_batches=15, batch_size=4)
    
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
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
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
