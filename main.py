from data_sampler import GPTDataset
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


with open("the-verdict.txt", "r") as f:
    text = f.read()


def create_data_loader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokeniser = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokeniser, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

data_loader = create_data_loader(txt=text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_loader_iter = iter(data_loader)
batch = next(data_loader_iter)
print(batch)
batch = next(data_loader_iter)
print(batch)