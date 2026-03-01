import torch
import torch.nn as nn
from transformerblock import TransformerBlock, LayerNorm
import tiktoken

TEST_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embed_dim" : 768, 
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class createdLLM(nn.Module): 
    def __init__(self, cfg):
        super().__init__()
        self.token_embeddings = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.position_embeddings = nn.Embedding(cfg["context_length"], cfg["embed_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.transformerblock = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_normalization = LayerNorm(cfg["embed_dim"])

        self.out_head = nn.Linear(
            cfg["embed_dim"], cfg["vocab_size"], bias=False # No bias, the transformer's output is put back into the vocabulary space. the logits step!
        )

    def forward(self, input_tokens):
        batch_size, token_length = input_tokens.shape
        token_embeds = self.token_embeddings(input_tokens)
        position_embeds = self.position_embeddings(torch.arange(token_length, device=input_tokens.device))
        x = token_embeds + position_embeds
        x = self.drop_emb(x)
        x = self.transformerblock(x)
        x = self.final_normalization(x)
        logits = self.out_head(x)
        return logits
    

torch.manual_seed(123)
model = createdLLM(TEST_CONFIG)

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

tokened1 = tokenizer.encode(txt1)
tokened2 = tokenizer.encode(txt2)

batch.append(torch.tensor(tokened1))
print(f" Batch length: {len(batch)}")
batch.append(torch.tensor(tokened2))
batch = torch.stack(batch, dim=0) 
print(f" Batch length: {len(batch)}")

print(" Batch:")
print(batch)

createdModel = createdLLM(TEST_CONFIG)
out = createdModel(batch)
print("\nOutput shape:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")


def generate_text_simple(model, input_tokens, new_tokens_max, context_length_max): 
    print(f" In function!")
    for _ in range(new_tokens_max): 
        print(f" In for loop")
        input_sliced = input_tokens[:, -context_length_max:] # the first index here is the batch dim! really important lol! 
        with torch.no_grad():
            logits = model(input_sliced)

        logits = logits[:, -1, :]
        logit_probabilities = torch.softmax(logits, dim=-1) # not truly necessary
        next_word = torch.argmax(logit_probabilities, dim=-1, keepdim=True) # argmax returns the index of the highest value - right!
        input_tokens = torch.cat((input_tokens, next_word), dim=-1)

    return input_tokens # lol


start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print(F" encoded sentence: {encoded}")

encoded_tokens = torch.tensor(encoded).unsqueeze(0) # add batch dim
print(f" Tensor encoded! Shape: {encoded_tokens.shape}")


# Let's go!
model.eval()

out2 = generate_text_simple(
    model=model,
    input_tokens=encoded_tokens,
    new_tokens_max=6,
    context_length_max=TEST_CONFIG["context_length"]
)

print(f" ~ ~ " * 50)
print(f" *** drumroll again ***")
print(f"  " * 50)

print(f"Output: ", out2)
print(f" Output length: ", len(out[0]))

decoded_text = tokenizer.decode(out2.squeeze(0).tolist())
print(decoded_text)
print(":) ")