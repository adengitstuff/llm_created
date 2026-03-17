import torch
from torch.utils.data import Dataset

class charDataset(Dataset):
    def __init__(self, token_sequences):
        self.tokens = torch.tensor(token_sequences, dtype=torch.long)
    
    def __len__(self):
        return len(self.tokens)  # 2000
    
    def __getitem__(self, idx):
        input = self.tokens[idx, :-1]  # everything except the last 1 for input
        targets = self.tokens[idx, 1:]    # 1: for the target
        return input, targets

