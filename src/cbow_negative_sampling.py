import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


class CBOWDataset(Dataset):
    def __init__(self, pairs, vocab_size, num_negatives=5, context_len=4):
        self.pairs = [pair for pair in pairs if len(pair[1]) == context_len]
        self.vocab_size = vocab_size
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target, context = self.pairs[idx]

        # Positive samples
        context_tensor = torch.tensor(context, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)
        label = torch.tensor(1.0)

        # Negative samples
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = random.randint(0, self.vocab_size - 1)
            if neg != target:
                negatives.append(neg)

        return context_tensor, target_tensor, torch.tensor(negatives, dtype=torch.long), label


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, context_idxs, target_idx, negative_idxs):
        # context_idxs: (batch, context_window)
        # target_idx: (batch,)
        # negative_idxs: (batch, num_negatives)

        # mean context vector
        context_embeds = self.in_embed(context_idxs)              # (batch, window, D)
        context_mean = context_embeds.mean(dim=1)                # (batch, D)

        # positive target
        target_embed = self.out_embed(target_idx)                # (batch, D)
        pos_score = torch.sum(context_mean * target_embed, dim=1)  # (batch,)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-10)

        # negative samples
        neg_embed = self.out_embed(negative_idxs)                # (batch, num_negatives, D)
        neg_score = torch.bmm(neg_embed, context_mean.unsqueeze(2)).squeeze(2)  # (batch, num_negatives)
        neg_loss = torch.sum(torch.log(1 - torch.sigmoid(neg_score) + 1e-10), dim=1)

        loss = - (pos_loss + neg_loss)                           # (batch,)
        return loss.mean()


def train_cbow(pairs, vocab_size, embedding_dim=100, epochs=5, batch_size=64, lr=0.01, num_negatives=5):
    dataset = CBOWDataset(pairs, vocab_size, num_negatives)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CBOWModel(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for context_idxs, target_idx, negative_idxs, _ in dataloader:
            optimizer.zero_grad()
            loss = model(context_idxs, target_idx, negative_idxs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")
