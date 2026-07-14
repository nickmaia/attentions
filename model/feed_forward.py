from torch import nn
import torch.nn.functional as F
import torch


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model:int, ff_ratio: int) -> None:
        super(FeedForwardNetwork, self).__init__()

        self.d_model = d_model
        self.d_ff = d_model * ff_ratio

        self.in_layer = nn.Linear(d_model, self.d_ff, bias=True)
        self.out_layer = nn.Linear(self.d_ff, d_model, bias=True)

    def forward(self, x):
        x = F.silu(self.in_layer(x))
        x = self.out_layer(x)
        return x


# Teste
if __name__ == "__main__":
    d_model = 512
    ff_ratio = 4
    batch_size = 2
    seq_len = 10

    ffn = FeedForwardNetwork(d_model, ff_ratio)
    input_tensor = torch.randn(batch_size, seq_len, d_model)
    output_tensor = ffn(input_tensor)

    print("Input shape:", input_tensor.shape)  # (batch_size, seq_len, d_model)
    print("Output shape:", output_tensor.shape)  # (batch_size, seq_len, d_model)