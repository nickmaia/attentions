import math
import torch
import torch.nn as nn
import numpy as np


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.embeddings = nn.Embedding(vocab_size, d_model)
        pe = self._create_positional_encoding(max_len, d_model)
        self.register_buffer("pe", pe)

    def _create_positional_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1).float()  
        i = torch.arange(0, d_model // 2)
        div_term = torch.pow(10_000.0, (2 * i) / d_model)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        return pe

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.embeddings(x.long()) * math.sqrt(self.d_model) 
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x
    
# Exemplo de uso
if __name__ == "__main__":
    vocab_size = 4
    d_model = 4
    max_len = 50
    embedding_layer = Embedding(vocab_size, d_model, max_len)

    # Simulando uma entrada de tokens (batch_size=2, seq_len=10)
    input_tokens = torch.randint(0, vocab_size, (2, 10))
    output = embedding_layer(input_tokens)
    print(output.shape)  # Deve ser (2, 10, 512)