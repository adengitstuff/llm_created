from causalattention import CausalAttention
import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module): 
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
        


# test:
inputs = torch.tensor(
 [[0.43, 0.15, 0.89], # Your (x^1)
 [0.55, 0.87, 0.66], # journey (x^2)
 [0.57, 0.85, 0.64], # starts (x^3)
 [0.22, 0.58, 0.33], # with (x^4)
 [0.77, 0.25, 0.10], # one (x^5)
 [0.05, 0.80, 0.55]] # step (x^6)
)

# batch is shape [2, 6, 3]:
batch = torch.stack((inputs, inputs), dim=0)

torch.manual_seed(123)
context_len = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2


mha = MultiHeadAttention(
d_in, d_out, context_len, 0.0, num_heads=2)
context_vecs = mha(batch)
print (context_vecs)
print("context_vecs.shape:", context_vecs.shape)