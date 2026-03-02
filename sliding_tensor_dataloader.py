import urllib.request
import tiktoken
from importlib.metadata import version
from torch.utils.data import Dataset, DataLoader
import torch

# instantiate BPE tokenizer:
tokenizer = tiktoken.get_encoding("gpt2")

with open("the-verdict.txt", "r") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(f" Raw text len: {len(raw_text)}")
print(f" Encoded: encoded text len is {len(enc_text)}")

# remove first 50:
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
#print(f"enc sample first 10: {enc_sample[:50]}")
print(f"x: {x}")
y = enc_sample[1:context_size+1]
print(f"y:          {y}")

print(f" ====")
for i in range(1, context_size+1):
    context = enc_sample[:i]
    print(f"decoded from encoded tokens:")
    print(f"{tokenizer.decode(context)}")
    print(f" ===" * 5)
    print(f" context: {context}")
    print(f" desired: {enc_sample[i]}")
    print(f" ")

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt) 
        num_tokens = len(token_ids) # just to be pedantic and fully understand

        for i in range(0, num_tokens - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# dataloader:

def create_dataloader_1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader   

if __name__ == "__main__":

    with open("the-verdict.txt", "r") as f:
        rawtext = f.read()

    # dataloader = create_dataloader_1(rawtext, batch_size=2, max_length=4, stride=4, shuffle=False)
    # data_iter = iter(dataloader)
    # first_batch = next(data_iter)
    # input, target_truths = next(data_iter)
    # print(f" Inputs:\n {input}")
    # print(F" Targets: \n {target_truths}")


    # embedding test:

    input_ids = torch.tensor([2, 3, 5, 1])

    vocab_size = 6
    output_dim = 3

    # embedding layer:
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)

    # now a larger test:
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    # batch of 8 means 8 batch, * 4 max length * 256 dim

    max_length = 4
    dataloader = create_dataloader_1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets_truth = next(data_iter)
    print(f" Token ID's: \n {inputs}")
    print(f" \nInputs shape: \n {inputs.shape}")


    token_embeddings = token_embedding_layer(inputs)
    print(f" Token embedding shape:")
    print(f"{token_embeddings.shape}")

    # to create the other embedding layer with same embedding dim:

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(f" Position embedding shape:")
    print(pos_embeddings.shape)

    # ---

    input_embeddings = token_embeddings + pos_embeddings
    print(f" Input embeddings = token embeddings + position embeddings shape:")
    print(input_embeddings.shape)

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
    each_6_context_vec = attn_weights @ inputs