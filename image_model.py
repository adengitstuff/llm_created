from architecture import createdLLM
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from characterDataset import charDataset
from torch.utils.data import random_split # tryign this out
from image_data import imgs_to_tokens

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMAGE_GENERATE_CONFIG = {
    "vocab_size": 512,
    "context_length": 16,
    "embed_dim" : 64, 
    "n_heads": 4,
    "n_layers": 6,
    "drop_rate": 0.2,
    "qkv_bias": False
}

IMAGE_GENERATE_CONFIG_2A = {
    "vocab_size": 1024,
    "context_length": 16,
    "embed_dim" : 64, 
    "n_heads": 4,
    "n_layers": 8,
    "drop_rate": 0.2,
    "qkv_bias": False
}

IMAGE_GENERATE_CONFIG_3A = {
    "vocab_size": 2048,
    "context_length": 16,
    "embed_dim" : 512, 
    "n_heads": 8,
    "n_layers": 6,
    "drop_rate": 0.2,
    "qkv_bias": False
}


model = createdLLM(IMAGE_GENERATE_CONFIG_3A).to(device)
vocab_size = IMAGE_GENERATE_CONFIG_3A["vocab_size"]
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) # test 3e-4
criterion = nn.CrossEntropyLoss()
num_epochs = 45
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

tokn_sequence = imgs_to_tokens(dataset_name="jiovine/pixel-art-nouns", kmeans_path="kmeans_2048_full.pkl")
dataset = charDataset(token_sequences=tokn_sequence)


train_number = int(0.9 * len(dataset)) # cast to int fix 
val_number = len(dataset) - train_number 

train_dataset, val_dataset = random_split(dataset, [train_number, val_number])
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=0) # I'm taking the stance of "nope" to multithreading on Windows
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True, num_workers=0 )
for epoch in range(num_epochs): 
    total_loss = 0
    for i, (input, targets) in enumerate(train_dataloader):
        input, targets = input.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(input)
        # change from (-1, 512)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        #backwards:
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader):.4f}")

    if epoch % 5 == 0:
        val_loss = 0
        for input, targets in val_dataloader:
            with torch.no_grad():
                input, targets = input.to(device), targets.to(device)
                logits = model (input)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                val_loss += loss.item()

        print(f" Val Loss for Epoch {epoch}: {val_loss/len(val_dataloader)}")
        print(f" Saving checkpoint!")
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'model_tests/model_3a_2048_kmeans_epoch_{epoch}.pt')

print(f" Done training! ")
torch.save(model.state_dict(), f'model_tests/model_3a_2048_kmeans_final_train.pt')
