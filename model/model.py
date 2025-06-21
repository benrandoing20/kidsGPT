import torch.nn as nn
import torch

class GPT2Simple(nn.Module):
    def __init__(self, vocab_size, n_layers=12, n_heads=12, n_emb=768, block_size=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_emb))
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_emb, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, attention_mask=None):
        B, T = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        
        # Handle attention masking for padding tokens
        if attention_mask is not None:
            # Convert boolean mask to float mask for transformer
            # True values become 0 (attend), False values become -inf (ignore)
            mask = torch.where(attention_mask, 0.0, float('-inf'))
            # Create causal mask for autoregressive generation
            causal_mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1) * float('-inf')
            # Combine padding mask and causal mask
            combined_mask = mask.unsqueeze(1) + causal_mask.unsqueeze(0)
            x = self.transformer(x, src_key_padding_mask=~attention_mask)
        else:
            x = self.transformer(x)
            
        return self.head(self.ln_f(x))
