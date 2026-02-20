"""Tiny transformer encoder regressor for financial return prediction."""

import torch
import torch.nn as nn

D_MODEL = 32
N_HEADS = 4
N_LAYERS = 1
DROPOUT = 0.1


class TinyTransformer(nn.Module):
    """Single-layer transformer encoder that regresses a scalar return.

    Args:
        d_features: Number of input features per timestep (default: 4).
        seq_len:    Sequence length for learned positional embeddings (default: 30).
        d_model:    Internal model dimension (default: 32).
        n_heads:    Number of attention heads (default: 4).
        dropout:    Dropout probability (default: 0.1).
        pool:       Pooling strategy — 'last' or 'mean' (default: 'last').
    """

    def __init__(
        self,
        d_features: int = 4,
        seq_len: int = 30,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        dropout: float = DROPOUT,
        pool: str = 'last',
    ):
        super().__init__()
        if pool not in ('last', 'mean'):
            raise ValueError(f"pool must be 'last' or 'mean', got '{pool}'")
        self.pool = pool

        # Input projection
        self.input_proj = nn.Linear(d_features, d_model)

        # Learned positional embedding
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.register_buffer('positions', torch.arange(seq_len))

        # Encoder block
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # Regression head
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: FloatTensor of shape (batch, seq_len, d_features).

        Returns:
            FloatTensor of shape (batch,) — predicted next-day return per sample.
        """
        # Project input features → d_model
        x = self.input_proj(x)                          # (B, T, d_model)

        # Add learned positional embeddings
        x = x + self.pos_emb(self.positions)            # (B, T, d_model)

        # Encoder block with pre-norm residuals
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))         # (B, T, d_model)
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))          # (B, T, d_model)

        # Pool
        if self.pool == 'last':
            x = x[:, -1, :]                             # (B, d_model)
        else:
            x = x.mean(dim=1)                           # (B, d_model)

        return self.head(x).squeeze(-1)                 # (B,)
