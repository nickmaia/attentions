import torch
import torch.nn as nn

# score = q * W_a * Kᵀ
class AttentionLuongGeneral(nn.Module):
    def __init__(self, d_model, head_dim, num_heads, mask=False, dropout=0.1):
        super().__init__()

        assert head_dim * num_heads == d_model, "head_dim * num_heads deve ser igual a d_model"

        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.mask = mask

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.proj_out = nn.Linear(d_model, d_model)

        self.W_a = nn.Linear(head_dim, head_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv=None, key_padding_mask=None):
        if x_kv is None:
            x_kv = x_q

        batch_size, seq_len_q, _ = x_q.size()
        _, seq_len_kv, _ = x_kv.size()

        q = self.query(x_q).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x_kv).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x_kv).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        k = self.W_a(k)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if self.mask:
            mask = torch.triu(
                torch.ones(seq_len_q, seq_len_kv, device=x_q.device),
                diagonal=1
            ).bool()
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Máscara de padding — ignora tokens <pad> nas keys
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = self.softmax(attn_scores)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)

        output = self.proj_out(attn_output)
        output = self.proj_dropout(output)

        return output