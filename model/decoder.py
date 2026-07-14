import torch
from torch import nn
from feed_forward import FeedForwardNetwork


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        num_heads: int,
        ff_ratio: int,
        dropout: float = 0.1,
        attn_cls=None,
        attn_kwargs=None
    ):
        super(DecoderBlock, self).__init__()

        from attention import Attention as _DefaultAttention

        if attn_cls is None:
            attn_cls = _DefaultAttention
        if attn_kwargs is None:
            attn_kwargs = {}

        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.ff_ratio = ff_ratio

        self.self_attention = attn_cls(
            d_model=d_model,
            head_dim=head_dim,
            num_heads=num_heads,
            mask=True,
            dropout=dropout,
            **attn_kwargs
        )
        self.norm_1 = nn.RMSNorm(d_model)

        self.cross_attention = attn_cls(
            d_model=d_model,
            head_dim=head_dim,
            num_heads=num_heads,
            mask=False,
            dropout=dropout,
            **attn_kwargs
        )
        self.norm_2 = nn.RMSNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model=d_model, ff_ratio=ff_ratio)
        self.norm_3 = nn.RMSNorm(d_model)

        self.dropout_attn_self = nn.Dropout(p=dropout)
        self.dropout_attn_cross = nn.Dropout(p=dropout)
        self.dropout_ffn = nn.Dropout(p=dropout)

    def forward(self, x, enc_out, src_pad_mask=None, tgt_pad_mask=None):
        # self-attention mascarada causalmente + padding do tgt
        y = self.self_attention(x, key_padding_mask=tgt_pad_mask)
        x = self.norm_1(x + self.dropout_attn_self(y))
        
        # cross-attention: queries do decoder, keys/values do encoder
        # padding mask usa src porque as keys vêm do encoder
        y = self.cross_attention(x, enc_out, key_padding_mask=src_pad_mask)
        x = self.norm_2(x + self.dropout_attn_cross(y))

        ff = self.ffn(x)
        x = self.norm_3(x + self.dropout_ffn(ff))

        return x


if __name__ == "__main__":
    batch_size = 2
    seq_len_dec = 5
    seq_len_enc = 7
    d_model = 16
    head_dim = 4
    num_heads = 4
    ff_ratio = 4

    decoder_block = DecoderBlock(
        d_model=d_model,
        head_dim=head_dim,
        num_heads=num_heads,
        ff_ratio=ff_ratio
    )

    x_dec = torch.rand(batch_size, seq_len_dec, d_model)
    x_enc = torch.rand(batch_size, seq_len_enc, d_model)

    out = decoder_block(x_dec, x_enc)

    print("Output shape:", out.shape) # (batch_size, seq_len_dec, d_model)