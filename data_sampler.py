import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):

    def __init__(self, txt, tokeniser, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokeniser.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunks = token_ids[i:i + max_length]
            target_chunks = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunks))
            self.target_ids.append(torch.tensor(target_chunks))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]