import torch

test_inputs = torch.tensor(
 [[0.43, 0.15, 0.89], # Your (x^1)
 [0.55, 0.87, 0.66], # journey (x^2)
 [0.57, 0.85, 0.64], # starts (x^3)
 [0.22, 0.58, 0.33], # with (x^4)
 [0.77, 0.25, 0.10], # one (x^5)
 [0.05, 0.80, 0.55]] # step (x^6)
)

query = test_inputs[1]
attention_scores_2 = torch.empty(test_inputs.shape[0])
print(f" ==")
for i, x_i in enumerate(test_inputs):
    print(f" ")
    print(f" {i}:")
    print(f" x_i: {x_i}")
    print(f" query: {query}")
    attn_score = torch.dot(x_i, query)
    print(f" attention score: {attn_score}")
    attention_scores_2[i] = attn_score
    print(f" ")


print(f" ")
print(f" {attention_scores_2}")


# skipping normalization for now, since relatively familiar with softmax:

attention_weights_2 = torch.softmax(attention_scores_2, dim=0)
print(f" raw, softmax'ed weights: {attention_weights_2}")

# so up till now:
# simple tokens as multi-dim embeddings
# take one word and get a dotproduct of each other word, 'is this vector pointed the same way'-ness
# "attention scores" are logits- softmax of attention scores 
# the full context vector is: for each 1D, 3-dim word [.523, .213, .215], multiply it by its attention weight (softmaxed attention score). 
#   sum up the resulting vectors
#   that's a context vector

# this is the part correlating to: 

# "multiplying the
# embedded input tokens, x(i), with the corresponding attention weights and then summing the resulting vectors. Thus, context vector z
# (2) is the weighted sum of all input vectors, obtained by multiplying each input vector by its corresponding attention weight:"

# I guess this is the naive approach, since just embedded inputs * attention scores isn't really creating new representations that are
# needed for real, actual attention

# context vector as a sum of all input vectors, * weights:
# this is a weight * a vector space
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(test_inputs):
    context_vec_2 += attention_weights_2[i]*x_i

print(f" The first context vector!")
print(f" {context_vec_2}")

#############################################################
# now, the context vector for each input. 
atten_scores_each = torch.empty(6, 6)
for i, xi in enumerate(test_inputs):
    for j, xj in enumerate(test_inputs):
        atten_scores_each[i, j] = torch.dot(xi, xj)

print(f" Got attention scores for each!")

print(atten_scores_each)

print(f" matrix mult:")

attn_scores_matrix = test_inputs @ test_inputs.T
print(f"~ " * 50)
print(attn_scores_matrix)
print(f"~ " * 50)

attn_scores_matrix = torch.softmax(attn_scores_matrix, dim=-1)
print(attn_scores_matrix)

attn_weights = torch.softmax(attn_scores_matrix, dim=-1) # along row
print(attn_weights)

# create all context vectors for each of the 6 words
each_6_context_vec = attn_weights @ test_inputs