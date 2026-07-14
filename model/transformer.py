import math
import torch
import torch.nn as nn
from embedding import Embedding
from enconder import EncoderBlock
from decoder import DecoderBlock

PAD_IDX = 0  # índice do <pad> no vocabulário

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        head_dim: int,
        num_heads: int,
        ff_ratio: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        mask: bool = False,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
        attn_cls=None,
        attn_kwargs=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.ff_ratio = ff_ratio
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.mask = mask
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings

        self.embedding = Embedding(vocab_size, d_model, max_len)

        if attn_kwargs is None:
            attn_kwargs = {}

        self.encoder = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                head_dim=head_dim,
                num_heads=num_heads,
                ff_ratio=ff_ratio,
                mask=False,
                dropout=dropout,
                attn_cls=attn_cls,
                attn_kwargs=attn_kwargs
            )
            for _ in range(num_encoder_layers)
        ])

        self.decoder = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                head_dim=head_dim,
                num_heads=num_heads,
                ff_ratio=ff_ratio,
                dropout=dropout,
                attn_cls=attn_cls,
                attn_kwargs=attn_kwargs
            )
            for _ in range(num_decoder_layers)
        ])

        self.output_layer = nn.Linear(d_model, vocab_size)

        # Weight tying: compartilha os pesos do embedding com a output layer
        # Economiza ~vocab_size * d_model parâmetros (~122 MB para 59K vocab)
        #
        # O nn.Embedding usa init N(0,1), que é ~20× maior que o Kaiming uniform
        # do nn.Linear. Re-inicializamos com a escala correta (sqrt(1/d_model))
        # para evitar logits enormes e loss altíssima no início do treino.
        #
        # Também removemos o weight do Linear do registro de parâmetros antes
        # de substituir pelo tensor do embedding, senão model.parameters()
        # devolveria o mesmo tensor duas vezes e o optimizer aplicaria update
        # em dobro.
        if tie_embeddings:
            bound = 1.0 / math.sqrt(d_model)
            nn.init.uniform_(self.embedding.embeddings.weight, -bound, bound)
            del self.output_layer.weight
            self.output_layer.weight = self.embedding.embeddings.weight
        

    def forward(self, src, tgt):
        # (B, seq) — True onde é <pad>, atenção será bloqueada
        src_pad_mask = (src == PAD_IDX)  # (B, seq_src)
        tgt_pad_mask = (tgt == PAD_IDX)  # (B, seq_tgt)

        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)


        for encoder_layer in self.encoder:
            src_emb = encoder_layer(src_emb, src_pad_mask=src_pad_mask)


        for decoder_layer in self.decoder:
            tgt_emb = decoder_layer(tgt_emb, src_emb,
                                    src_pad_mask=src_pad_mask,
                                    tgt_pad_mask=tgt_pad_mask)

        output = self.output_layer(tgt_emb)

        return output


if __name__ == "__main__":
    vocab_size = 1000
    d_model = 512
    max_len = 100
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    head_dim = 64
    ff_ratio = 4
    batch_size = 2
    seq_len = 10

    transformer = Transformer(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=d_model,
        head_dim=head_dim,
        num_heads=num_heads,
        ff_ratio=ff_ratio,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        mask=False,
        dropout=0.1
    )

    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = transformer(src, tgt)

    print("Output shape:", output.shape) # (batch_size, seq_len, vocab_size)