import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

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
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Statistics tracking
    original_lengths = []
    padded_lengths = []
    
    # Sample some batches for detailed analysis
    for batch_idx, (xb, yb) in enumerate(dl):
        if batch_idx >= num_batches:
            break
            
        # Get original lengths before padding
        # We need to track which samples are in this batch
        # Since we're using shuffle=True, we can't easily map back to original indices
        # Instead, we'll just record the padded lengths for this batch
        batch_size_actual = xb.shape[0]
        for i in range(batch_size_actual):
            padded_lengths.append(xb.shape[1])
    
    # For original lengths, we'll sample from the dataset directly
    # since we can't easily map back from shuffled batches
    sample_indices = np.random.choice(len(dataset), min(len(padded_lengths), len(dataset)), replace=False)
    for idx in sample_indices:
        x_orig, y_orig = dataset[idx]
        original_lengths.append(len(x_orig))
    
    # Create single plot showing before/after lengths
    plt.figure(figsize=(12, 6))
    
    x_pos = np.arange(len(original_lengths))
    plt.bar(x_pos - 0.2, original_lengths, 0.4, label='Original Length', alpha=0.7, color='skyblue')
    plt.bar(x_pos + 0.2, padded_lengths, 0.4, label='Padded Length', alpha=0.7, color='lightcoral')
    plt.xlabel('Sample Index')
    plt.ylabel('Sequence Length')
    plt.title('Original vs Padded Lengths for Each Data Sample')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('checkpoints/collate_function_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCollate function analysis saved to: checkpoints/collate_function_analysis.png")
    
    # Print summary statistics
    avg_original = np.mean(original_lengths)
    avg_padded = np.mean(padded_lengths)
    padding_ratio = 1 - (avg_original / avg_padded) if avg_padded > 0 else 0
    
    print(f"\nSummary:")
    print(f"  Average original length: {avg_original:.1f}")
    print(f"  Average padded length: {avg_padded:.1f}")
    print(f"  Padding ratio: {padding_ratio:.2%}")
    
    return {
        'avg_original_length': avg_original,
        'avg_padded_length': avg_padded,
        'padding_ratio': padding_ratio
    } 