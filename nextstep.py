import tiktoken
from sliding_tensor_dataloader import create_dataloader_1
import torch
from architecture import createdLLM, text_to_token_ids, token_ids_to_text, generate_text_simple
import numpy as np


TEST_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256, # from 1024
    "embed_dim" : 768, 
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
CONFIG_TEST_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embed_dim": 1024,
    "n_heads": 16,
    "n_layers": 24, 
    "drop_rate": 0.1,
    "qkv_bias": True
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#                                       Start!                                                                  #

tokenizer = tiktoken.get_encoding("gpt2")

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
 text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

train_ratio = 0.9
# get the index
split_idx = int(train_ratio * len(text_data))
# split first n%
train_data = text_data[:split_idx]
# i guess no split_idx+1?
val_data = text_data[split_idx:]

# Dataloaders! Familiar territory :)

torch.manual_seed(123)
train_loader = create_dataloader_1(
 train_data,
 batch_size=2,
 max_length=TEST_CONFIG["context_length"],
 stride=TEST_CONFIG["context_length"], # context length is the stride
 drop_last=True, #@
 shuffle=True,
 num_workers=0
)

val_loader = create_dataloader_1(
 val_data,
 batch_size=2,
 max_length=TEST_CONFIG["context_length"],
 stride=TEST_CONFIG["context_length"], 
 drop_last=False, 
 shuffle=False,
 num_workers=0
)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)
print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

# CE Loss! Familiar territory
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1), target_batch.flatten()
    )


    return loss


# Batch loss 
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


# ###############################
# model = createdLLM(TEST_CONFIG)
# model.to(device)
# with torch.no_grad():
#     train_loss = calc_loss_loader(train_loader, model, device)
#     val_loss = calc_loss_loader(val_loader, model, device)
# print("Training loss:", train_loss)
# print("Validation loss:", val_loss)



def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel() #@
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f" Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.4f}, "
                      f"Val loss {val_loss:.2f}"
                      )
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.position_embeddings.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
        model=model, input_tokens=encoded,
        new_tokens_max=50, context_length_max=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


## Assign utility function! checks if same dimension and returns right tensor as trainable torch params. 

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f" Shape mismatch! Left: {left.shape}, and Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights(gpt, params):
    gpt.position_embeddings.weight = assign(gpt.position_embeddings.weight, params['wpe'])
    gpt.token_embeddings.weight = assign(gpt.token_embeddings.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
        (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformerblock[b].att.W_query.weight = assign(
        gpt.transformerblock[b].att.W_query.weight, q_w.T)
        gpt.transformerblock[b].att.W_key.weight = assign(
        gpt.transformerblock[b].att.W_key.weight, k_w.T)
        gpt.transformerblock[b].att.W_value.weight = assign(
        gpt.transformerblock[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformerblock[b].att.W_query.bias = assign(
            gpt.transformerblock[b].att.W_query.bias, q_b)
        gpt.transformerblock[b].att.W_key.bias = assign(
            gpt.transformerblock[b].att.W_key.bias, k_b)
        gpt.transformerblock[b].att.W_value.bias = assign(
            gpt.transformerblock[b].att.W_value.bias, v_b)
        
        gpt.transformerblock[b].att.out_proj.weight = assign(
            gpt.transformerblock[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
    


        gpt.transformerblock[b].att.out_proj.bias = assign(
            gpt.transformerblock[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.transformerblock[b].ff.layers[0].weight = assign(
            gpt.transformerblock[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformerblock[b].ff.layers[0].bias = assign(
            gpt.transformerblock[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformerblock[b].ff.layers[2].weight = assign(
            gpt.transformerblock[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformerblock[b].ff.layers[2].bias = assign(
            gpt.transformerblock[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.transformerblock[b].ln1.scale = assign(
            gpt.transformerblock[b].ln1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformerblock[b].ln1.shift = assign(
            gpt.transformerblock[b].ln1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformerblock[b].ln2.scale = assign(
            gpt.transformerblock[b].ln2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformerblock[b].ln2.shift = assign(
            gpt.transformerblock[b].ln2.shift,
            params["blocks"][b]["ln_2"]["b"])
    
    gpt.final_normalization.scale = assign(gpt.final_normalization.scale, params["g"])
    gpt.final_normalization.shift = assign(gpt.final_normalization.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

# torch.manual_seed(123)
# model = createdLLM(TEST_CONFIG)
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
# num_epochs = 10
# # Here we go!
# the_first_start_context = "Let us start training!"
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, start_context=the_first_start_context, tokenizer=tokenizer
# )

# GPT2-pretrained weights test:
# settings_gpt, params_gpt = download_and_load_gpt2(model_size="355M", models_dir="gpt-2")
# torch.save(params_gpt, "pretrained_params_355M.pt")

#print("Settings:", settings_gpt)
params_gpt = torch.load("pretrained_params_355M.pt", weights_only=False)
print("Parameter dictionary keys:", params_gpt.keys ())

# initialize pretrained:

pretrained = createdLLM(CONFIG_TEST_355M)
pretrained.eval()
load_weights(pretrained, params_gpt)
print("Token embedding weight tensor dimensions:", params_gpt["wte"].shape)


torch.manual_seed(123)
token_ids = generate_text_simple(
 model=pretrained,
 input_tokens=text_to_token_ids("Wireless signals and systems are", tokenizer).to(device),
 new_tokens_max=25,
 context_length_max=CONFIG_TEST_355M["context_length"],
 #top_k=50,
 #temperature=1.5
)
print(f" * * ")
print(f"  ")
print(f" ** the biggest drumroll yet! **")
print(f" " * 50)
print(f" ~ " * 50)
print(f" * *" * 50)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))