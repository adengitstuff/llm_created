from architecture import createdLLM
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from characterDataset import charDataset
from torch.utils.data import random_split # tryign this out
from image_data import imgs_to_tokens

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMAGE_MODE_CONFIG = {
    "vocab_size": 512,
    "context_length": 16,
    "embed_dim" : 64, 
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.3,
    "qkv_bias": False
}

model = createdLLM(IMAGE_MODE_CONFIG).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) # test 3e-4
criterion = nn.CrossEntropyLoss()
num_epochs = 38

tokn_sequence = imgs_to_tokens()
dataset = charDataset(token_sequences=tokn_sequence)


train_size = int(0.9 * len(dataset))  # 1800
val_size = len(dataset) - train_size   # 200

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=0) # nope to multithreading on Windows
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True, num_workers=0 )
for epoch in range(num_epochs): 
    total_loss = 0
    for i, (input, targets) in enumerate(train_dataloader):
        input, targets = input.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits.view(-1, 512), targets.view(-1))
        #backwards:
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader):.4f}")

    if epoch % 5 == 0:
        val_loss = 0
        for input, targets in val_dataloader:
            with torch.no_grad():
                input, targets = input.to(device), targets.to(device)
                logits = model (input)
                loss = criterion(logits.view(-1, 512), targets.view(-1))
                val_loss += loss.item()

        print(f" Val Loss for Epoch {epoch}: {val_loss/len(val_dataloader)}")
torch.save(model.state_dict(), f'character_multimodal_image_first_train_embed_128.pt')
