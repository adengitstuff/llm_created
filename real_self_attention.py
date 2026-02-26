

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

input_words_2 = inputs[1] # journey
d_in = inputs.shape[1] 
d_out = 2

# The first creation of Wq, Wk, Wvalue!
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # grad false for now
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = input_words_2 @ W_query
# print(f" x_2, the word 'journey''s embedding: ")
# print(input_words_2)

# print(f" that embedding multiplied by the query space ")
# print(query_2)

# print(" " * 50)

# print(f"{query_2}")
# print(" ~ ")
# print(f"W_Query shape: {W_query.shape}")
#print(W_query)
# print(" ~ ")
key_for_word_2 = input_words_2 @ W_key
#print(f" {key_2}")
value_for_word_2 = input_words_2 @ W_value
#print(key_for_word_2)

# print(f" ")
# print(f"{value_2}")

# print(" ### " * 25)
# print(f"W Key: {W_key}")
# print(" ### " * 25)


# print(" ### " * 25)
# print(f"W Value: {W_value}")
# print(" ### " * 25)


# print(f" ===== ")
# print(f" raw input 'Journey': {x_2}")
# print(f" ==== ")
# print(f" ~  " * 5)
# print(f"    'Journey' embedding * Weight Query [3, 2] to force 2-dim out:")
# print(query_2)

# The important intuition here is to think about projecting down from 3-dimensions into 2! That could probably
#   be important?
###################################

##### continue with more production level, just all at once:

keys = inputs @ W_key
values = inputs @ W_value

print(keys)

# compute 'Journey''s attention score with respect to itself, in Q and K spaceS:
key_for_journey = keys[1]

query_for_journey = (inputs @ W_query)[1]

print(f" Query for journey: {query_for_journey}")
print(f" Key for journey: {key_for_journey}")

attention_score_for_journey_wrt_journey = query_for_journey.dot(key_for_journey)
print(attention_score_for_journey_wrt_journey)

########
# being really explicit, taking extra steps:

print(f" 'Starts' embedding: {inputs[2]}")
print(f" 'With' embedding: {inputs[3]}")
query_for_start = (inputs @ W_query)[2]
key_for_with = (inputs @ W_key)[3]
attention_score_for_start_wrt_with = query_for_start.dot(key_for_with)
print(attention_score_for_start_wrt_with)

##### 

query_for_step = inputs[5] @ W_query
print(f" Query for step: {query_for_step}")

key_for_starts = inputs[0] @ W_key
print(f" Key for Starts: {key_for_starts}")
attention_score_for_step_wrt_starts = query_for_step.dot(key_for_starts)
print(f" Attention score for STEP with respect to Starts : {attention_score_for_step_wrt_starts}")

# The query space is evolving to ask the right questions about the innate essence of the word :o or at least
#   the dimensions' essence. the dimensions r hierarchal features, like convolutional layers; this is really powerful for the task!

### 

# repeating for all-at-once:

# Essentially:
#   'Take each row in the left thing, and dot it with each column in the right thing'
keys = inputs @ W_key 
values = inputs @ W_value


print("keys.shape:", keys.shape)
print("values.shape:", values.shape)


## All attention scores for 'Journey' at once, against Keys

attention_scores_for_journey_all = query_for_journey @ keys.T
print(f" Attention scores for journey - all: ")
print(attention_scores_for_journey_all)

                                                ## Important: the W_Value are distilling or learning 'what aspects are 
                                                #       learned to be important for other words' in a latent space. so,
                                                #       softmax'ed attention scores = attention weights, and those weights
                                                #       decide which way to pull journey's context vector. so cool. 

# I think the dot products need to handle any dimension size, 
#   and softmax will be wonky if the values are larger, actually. This I think is the
#   reason for the dk step
d_k = keys.shape[-1] # keys [6, 2] so just "2"
normalized_attention_weights_for_journey_pre_softmax = attention_scores_for_journey_all / (d_k ** 0.5)
print(f" d_k : {d_k}  |    and dk's sqrt: {d_k ** 0.5}")


print(f" attention weights for 'journey' pre-softmax: {normalized_attention_weights_for_journey_pre_softmax}")
softmaxed_attention_weights_for_journey_wrt_each_word = torch.softmax(normalized_attention_weights_for_journey_pre_softmax, dim=-1)
print(f" Softmax'ed attention weights for 'journey': {softmaxed_attention_weights_for_journey_wrt_each_word}")