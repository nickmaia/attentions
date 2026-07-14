import torch
from torch import nn
from feed_forward import FeedForwardNetwork

class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        num_heads: int,
        ff_ratio: int,
        mask: bool = False,
        dropout: float = 0.1,
        attn_cls=None,
        attn_kwargs=None
    ):
        super(EncoderBlock, self).__init__()

        from attention import Attention as _DefaultAttention

        if attn_cls is None:
            attn_cls = _DefaultAttention
        if attn_kwargs is None:
            attn_kwargs = {}

        # SETUP
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.ff_ratio = ff_ratio
        self.mask = mask


        # LAYERS
        self.attention = attn_cls(d_model=d_model, head_dim=head_dim, num_heads=num_heads, mask=mask, dropout=dropout, **attn_kwargs)
        self.norm_1 = nn.RMSNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model=d_model, ff_ratio=ff_ratio)
        self.norm_2 = nn.RMSNorm(d_model)

        # DROPOUT
        self.dropout_attn = nn.Dropout(p=dropout)
        self.dropout_ffn = nn.Dropout(p=dropout)


    def forward(self, x, src_pad_mask=None):
        y = self.attention(x, key_padding_mask=src_pad_mask)
        x = self.norm_1(x + self.dropout_attn(y))   

        ff = self.ffn(x)
        x = self.norm_2(x + self.dropout_ffn(ff))   

        return x

# Teste
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 16
    head_dim = 4
    num_heads = 4
    ff_ratio = 4
    dropout = 0.1

    encoder_block = EncoderBlock(d_model=d_model, head_dim=head_dim, num_heads=num_heads, ff_ratio=ff_ratio, dropout=dropout)
    x = torch.rand(batch_size, seq_len, d_model)
    src_pad_mask = None  # ou crie uma máscara de padding se necessário
    output = encoder_block(x, src_pad_mask=src_pad_mask)

    print("Output shape:", output.shape)  # (batch_size, seq_len, d_model)