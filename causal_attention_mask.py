
import torch
from self_attention_linear_layers import SelfAttention_v2

inputs = torch.tensor(
 [[0.43, 0.15, 0.89], # Your (x^1)
 [0.55, 0.87, 0.66], # journey (x^2)
 [0.57, 0.85, 0.64], # starts (x^3)
 [0.22, 0.58, 0.33], # with (x^4)
 [0.77, 0.25, 0.10], # one (x^5)
 [0.05, 0.80, 0.55]] # step (x^6)
)


# attention weights 

selfattn_v2 = SelfAttention_v2(d_in=3, d_out=2)

queries_all = selfattn_v2.W_query(inputs)
print(f" Queries all shape: {queries_all.shape}")
print(queries_all)

print(f" " * 50)
keys_all = selfattn_v2.W_key(inputs)
print(f" Keys all: ")
print(keys_all)
attention_scores_all = queries_all @ keys_all.T # [6, 2] @ [2, 6]

print(f" Attention scores all (queries @ keys.T) shape: {attention_scores_all.shape}")
print(f" Attention scores all:")

print(attention_scores_all)

###
# python tril to create mask:

context_length = attention_scores_all.shape[0] 
mask_simple = torch.tril(torch.ones(context_length, context_length)) 
print(mask_simple)

attention_weights = torch.softmax(attention_scores_all / (keys_all.shape[-1] ** 0.5), dim=-1)

causal_mask_simple = attention_weights*mask_simple
print(causal_mask_simple)


## Third counterintuitive step: renormalize attention weights to sum up to 1 again in each row. 
row_sums = causal_mask_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = causal_mask_simple/row_sums
print(masked_simple_norm)

# inf values trick:

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attention_scores_all.masked_fill(mask.bool(), -torch.inf)
print (masked)
attn_weights_inftrick = torch.softmax(masked / keys_all.shape [-1] **0.5, dim=1)
