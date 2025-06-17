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

    def forward(self, idx):
        B, T = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        x = self.transformer(x)
        return self.head(self.ln_f(x))
