from parallel_multi_attention import MultiHeadAttention
from layernorm import LayerNorm
from gelu import GELU
from placeholder_gpt import FeedForward
import torch.nn as nn
import torch


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["embed_dim"],
            d_out = cfg["embed_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.ln1 = LayerNorm(cfg["embed_dim"])
        self.ln2 = LayerNorm(cfg["embed_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):

        residual = x
        x = self.ln1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + residual

        residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + residual
        return x


FIRST_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embed_dim" : 768, 
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(FIRST_CONFIG)
output = block(x)
print("Input shape:", x.shape)
print(f" Input : ")
print(x)
print("Output shape:", output.shape)

print(f" " * 50)
print(f" " * 50)
print(f" *** drumroll *** ")
print(f" " * 50)
print(f" " * 50)

print(f"{output}")