

# scaled dot product attention, used in the original transformer architecture
# and the original GPT models?!

import torch


## Start with repeat of testing on one word

inputs = torch.tensor(
 [[0.43, 0.15, 0.89], # Your (x^1)
 [0.55, 0.87, 0.66], # journey (x^2)
 [0.57, 0.85, 0.64], # starts (x^3)
 [0.22, 0.58, 0.33], # with (x^4)
 [0.77, 0.25, 0.10], # one (x^5)
 [0.05, 0.80, 0.55]] # step (x^6)
)

x_2 = inputs[1] # journey
d_in = inputs.shape[1] 
d_out = 2

# The first creation of Wq, Wk, Wvalue!
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # grad false for now
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
print(" ~ ")
print(f"W_Query shape: {W_query.shape}")
print(W_query)
print(" ~ ")
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(f" ===== ")
print(f" raw input 'Journey': {x_2}")
print(f" ==== ")
print(f" ~  " * 5)
print(f"    'Journey' embedding * Weight Query [3, 2] to force 2-dim out:")
print(query_2)

# The important intuition here is to think about projecting down from 3-dimensions into 2! That could probably
#   be important?
###################################

# All at once:
keys = inputs @ W_key #[6 , 3]  @  [3, 2]
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)


######## @

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# so:

attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)


d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

# context vectors:

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)