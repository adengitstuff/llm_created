import torch
import torch.nn as nn
import tiktoken

#
from layernorm import LayerNorm
from gelu import GELU

FIRST_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embed_dim" : 768, 
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class gptModel(nn.Module):
    def __init__(self, cfg): 
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"]) # e.g. [50247, 768]
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["embed_dim"]) # [total or seq length?, 768]
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            * [DummyTransformerBlock(cfg)
               for _ in range (cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["embed_dim"])
        self.out_head = nn.Linear(
            cfg["embed_dim"], cfg["vocab_size"], bias=False #@ bias false!
        )
    def forward(self, input_tokens):
        batch_size, sequence_length = input_tokens.shape
        token_embeddings = self.token_embedding(input_tokens)
        position_embeddings = self.position_embedding(torch.arange(sequence_length, device=input_tokens.device))
        x = token_embeddings + position_embeddings
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
    
# feedforward here:
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),
            GELU(),
            nn.Linear(4 * cfg["embed_dim"], cfg["embed_dim"])
        )
    def forward(self, x):
        return self.layers(x)


# if __name__ == "__main__":
    # tokenizer = tiktoken.get_encoding("gpt2")
    # batch = []
    # txt1 = "Every effort moves you"
    # txt2 = "Every day holds a"

    # batch.append (torch.tensor (tokenizer.encode (txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))
    # batch = torch.stack (batch, dim=0)
    # print (batch)

    # # finally! Initialize 124M dumy model

    # torch.manual_seed(123)
    # model = gptModel(FIRST_CONFIG)  
    # logits = model(batch)
    # print(f"Output shape: {logits.shape}")
    # print("Logits: ")
    # print(logits)

    # # print(f" Layernorm recreate example: ")

    # torch.manual_seed(123)
    # batch_example = torch.randn(2, 5)
    # # layer = nn.Sequential (nn.Linear(5, 6), nn.ReLU ())
    # # out = layer(batch_example)
    # # print(out)

    # # mean = out.mean(dim=-1, keepdim=True)
    # # var = out.var(dim=-1, keepdim=True)
    # # print("Mean:\n", mean)
    # # print("Variance:\n", var)

    # # #layernorm test:

    # # out_norm = (out - mean) / torch.sqrt(var)
    # # mean = out_norm.mean(dim=-1, keepdim=True)
    # # var = out_norm.var(dim=-1, keepdim=True)
    # # print("Normalized layer outputs:\n", out_norm)
    # # print("Mean:\n", mean)
    # # print("Variance:\n", var)
    # # torch.set_printoptions(sci_mode=False)
    # # print("Mean:\n", mean)
    # # print("Variance:\n", var)

    # # end layernorm example

    # ln = LayerNorm(emb_dim=5)
    # out_ln = ln(batch_example)
    # mean = out_ln.mean(dim=-1, keepdim=True)
    # var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    # print("Mean! :\n", mean)
    # print("Variance:\n", var)


    # ffn = FeedForward(FIRST_CONFIG)
    # x = torch.rand(2, 3, 768)
    # out = ffn(x)
    # print(out.shape)