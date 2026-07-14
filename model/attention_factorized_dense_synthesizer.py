import math
import torch
import torch.nn as nn


class AttentionFactorizedDenseSynthesizer(nn.Module):
    def __init__(self, d_model, head_dim, num_heads, max_seq_len=128, a=None, mask=False, dropout=0.1):
        super().__init__()

        assert head_dim * num_heads == d_model, "head_dim * num_heads deve ser igual a d_model"

        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.mask = mask

        if a is None:
            a = math.isqrt(max_seq_len)
            while max_seq_len % a != 0:
                a -= 1

        b = max_seq_len // a
        assert a * b == max_seq_len, "a * b deve ser igual a max_seq_len"

        self.a = a
        self.b = b

        self.score_a = nn.Linear(d_model, num_heads * a)
        self.score_b = nn.Linear(d_model, num_heads * b)

        self.value = nn.Linear(d_model, d_model, bias=False)
        self.proj_out = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def _tile(self, x, repeat):
        batch_size, num_heads, seq_len, dim = x.shape
        return x.unsqueeze(-1).expand(batch_size, num_heads, seq_len, dim, repeat).reshape(
            batch_size, num_heads, seq_len, dim * repeat
        )

    def forward(self, x_q, x_kv=None, key_padding_mask=None):
        if x_kv is None:
            x_kv = x_q

        batch_size, seq_len_q, _ = x_q.size()
        _, seq_len_kv, _ = x_kv.size()

        if seq_len_kv > self.max_seq_len:
            raise ValueError("seq_len_kv não pode ser maior que max_seq_len")

        a, b = self.a, self.b

        score_a = self.score_a(x_q).view(batch_size, seq_len_q, self.num_heads, a).transpose(1, 2)
        score_b = self.score_b(x_q).view(batch_size, seq_len_q, self.num_heads, b).transpose(1, 2)

        attn_scores = self._tile(score_a, b) * self._tile(score_b, a)
        attn_scores = attn_scores[..., :seq_len_kv]

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

        v = self.value(x_kv).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)

        output = self.proj_out(attn_output)
        output = self.proj_dropout(output)

        return output

"""
Transformer com Factorized Dense Synthesizer
=============================================
O cálculo dos scores de atenção:
    PADRÃO            :  scores = (X·Wq) · (X·Wk)ᵀ / √d
    DENSE             :  scores = G(ReLU(F(X)))               →  R^(N×N)
    FACTORIZED DENSE  :  scores = HA(FA(X)) ⊙ HB(FB(X))      →  R^(N×N)  (low-rank)

Onde:
    FA: R^d → R^a     FB: R^d → R^b     com a × b = N
    HA: tiling de a → N  (cada elemento repetido b vezes)
    HB: tiling de b → N  (cada elemento repetido a vezes)
    ⊙ : multiplicação element-wise

Redução de parâmetros vs Dense:
    Dense            :  d·H·N  por camada (projeção final)
    Factorized Dense :  d·H·(a+b) << d·H·N  quando a,b << N
"""