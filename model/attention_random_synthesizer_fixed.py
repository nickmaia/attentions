import torch
import torch.nn as nn

class AttentionRandomSynthesizerFixed(nn.Module):
    def __init__(self, d_model, head_dim, num_heads, max_seq_len=128, mask=False, dropout=0.1):
        super().__init__()

        assert head_dim * num_heads == d_model, "head_dim * num_heads deve ser igual a d_model"

        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.mask = mask

        # buffer = nao treinavel, salvo no state_dict
        R = torch.randn(num_heads, max_seq_len, max_seq_len) * 0.02
        self.register_buffer("R", R)

        self.value = nn.Linear(d_model, d_model, bias=False)
        self.proj_out = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv=None, key_padding_mask=None):
        if x_kv is None:
            x_kv = x_q

        batch_size, seq_len_q, _ = x_q.size()
        _, seq_len_kv, _ = x_kv.size()

        # Recorta para o tamanho real da sequencia
        R = self.R[:, :seq_len_q, :seq_len_kv]                 # [H, T, S]
        attn_scores = R.unsqueeze(0).expand(batch_size, -1, -1, -1)

        if self.mask:
            mask = torch.triu(
                torch.ones(seq_len_q, seq_len_kv, device=x_q.device),
                diagonal=1
            ).bool()
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

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
Transformer com Synthesizer Random Fixed
=========================================
Os scores de atenção sao gerados por uma matriz R aleatoria FIXA:

    RANDOM FIXED :  attn = softmax(R) * V   onde R ~ N(0, 0.02) e nunca atualizado

Diferencas entre as variantes:

    Fator     | Random Trainable | Random Fixed
    ----------|------------------|---------------
    R         | Parameter        | buffer (congelado)
    Treina?   | Sim              | Nao
    Scores    | R1 @ R2.T (low-rank) | R (full rank)
    Parametros| 2 * H * N * k    | 0 (apenas V e proj_out)

Hipoteses que este mecanismo testa:
    1. A atencao realmente precisa ser aprendida?
    2. Um padrao aleatorio fixo ja captura informacao posicional?
    3. Quanto do ganho do Random Trainable vem de treinar V (nao R)?
"""
