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

input_ids = torch.tensor([2,3,5,1])
vocab_size = 50257
output_dim = 256
max_length = 4
dataloader = create_data_loader(text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
input_data, target_data = next(data_iter)

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(input_data)

context_length = max_length
pos_embedded_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedded_layer(torch.arange(context_length))
print(token_embeddings.shape)  # Should be (batch_size, max_length, output_dim)
print(pos_embeddings.shape)  # Should be (max_length, output_dim)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)  # Should be (batch_size, max_length, output_dim)