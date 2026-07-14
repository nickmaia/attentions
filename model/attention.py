import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Attention dot product
# score = Q @ K^T / sqrt(d_k)
class Attention(nn.Module):
    def __init__(self, d_model, head_dim, num_heads, mask=False, dropout=0.1):
        super().__init__()

        assert head_dim * num_heads == d_model, "head_dim * num_heads deve ser igual a d_model"
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.mask = mask
        self.divisor = head_dim ** 0.5

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key   = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.proj_out = nn.Linear(d_model, d_model)

        self.softmax      = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv=None, key_padding_mask=None):
        """
        x_q:             (B, seq_q,  d_model)
        x_kv:            (B, seq_kv, d_model)  — None → self-attention
        key_padding_mask: (B, seq_kv) bool — True nas posições de padding
        """
        if x_kv is None:
            x_kv = x_q

        B, seq_q,  _ = x_q.size()
        _, seq_kv, _ = x_kv.size()

        q = self.query(x_q).view(B, seq_q,  self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x_kv).view(B, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x_kv).view(B, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = q @ k.transpose(-2, -1) / self.divisor  # (batch, num_heads, seq_q, seq_kv)

        # Máscara causal (decoder self-attention)
        if self.mask:
            causal = torch.triu(
                torch.ones(seq_q, seq_kv, device=x_q.device), diagonal=1
            ).bool()
            attn_scores = attn_scores.masked_fill(
                causal.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Máscara de padding — ignora tokens <pad> nas keys
        if key_padding_mask is not None:
            # (B, seq_kv) → (B, 1, 1, seq_kv) para broadcast por head e query
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = self.softmax(attn_scores)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v                                             # (batch, num_heads, seq_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, seq_q, self.d_model)
        out = self.proj_dropout(self.proj_out(out))

        return out
    """
    def visualizar(self, x, batch_idx=0, save_path="attention_heads_pretty_masked.png"):
        self.eval()

        with torch.no_grad():
            _, attn_weights = self(x)

        attn = attn_weights[batch_idx].cpu().numpy()  # shape: (num_heads, seq_len, seq_len)

        sns.set_theme(style="white", context="talk")
        ncols = 4
        nrows = math.ceil(self.num_heads / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))
        axes = axes.flatten()

        for h in range(self.num_heads):
            ax = axes[h]
            sns.heatmap(
                attn[h],
                ax=ax,
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                annot=True,
                fmt=".2f",
                square=True,
                cbar=True,
                linewidths=0.5,
                linecolor="white"
            )
            ax.set_title(f"Head {h}", fontsize=14, weight="bold")
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")

        for h in range(self.num_heads, len(axes)):
            fig.delaxes(axes[h])

        fig.suptitle("Attention Weights por Head", fontsize=20, weight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()"""

if __name__ == "__main__":
    attn_masked = Attention(d_model=16, head_dim=4, num_heads=4, mask=True)
    x = torch.randn(2, 5, 16)  # (batch, seq_len, d_model)
    attn_masked.visualizar(x, batch_idx=0)
    
    attn_unmasked = Attention(d_model=16, head_dim=4, num_heads=4, mask=False)
    attn_unmasked.visualizar(x, batch_idx=0, save_path="attention_heads_pretty_unmasked.png")
    
    
    