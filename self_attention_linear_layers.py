import torch.nn as nn
import torch

# this is right before causal attention!
# the book mentions the random initializations for linear layers happen in a transposed way - i didn't know this

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) # the idea of a bias on a linear layer is totally new to me, super interesting. 
                                                             #  researched: so it's a matrix that is trained params so it's effectively (I'm trying to think in terms of high-dimensional latent spaces, lol!)
                                                             #  'sliding' space over
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec