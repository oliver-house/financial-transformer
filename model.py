"""Transformer model for predicting next-day AAPL returns."""

import torch
import torch.nn as nn

D_MODEL = 32
N_HEADS = 4
DROPOUT = 0.1

class TinyTransformer(nn.Module):
    def __init__(self, d_features=4, seq_len=30, d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT, pool='last'):
        super().__init__()
        if pool not in ('last', 'mean'):
            raise ValueError(f"pool must be 'last' or 'mean', got '{pool}'")
        self.pool = pool
        self.input_proj = nn.Linear(d_features, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.register_buffer('positions', torch.arange(seq_len))
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        """Forward pass"""
        x = self.input_proj(x)               
        x = x + self.pos_emb(self.positions)      
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))     
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))      
        if self.pool == 'last':
            x = x[:, -1, :]           
        else:
            x = x.mean(dim=1)                 
        return self.head(x).squeeze(-1)            