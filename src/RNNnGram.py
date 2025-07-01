import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader


class RNNnGramDataset(Dataset):
    def __init__(self, tokenized_sentences, vocab, seq_len, unk_idx):
        self.vocab = vocab
        self.seq_len = seq_len

        self.sentences = []

        # Filter sentences to ensure they have at least 2 tokens
        for s in tokenized_sentences:
            indices = [vocab.get(w, unk_idx) for w in s]
            if len(indices) >= 2:
                self.sentences.append(s)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # Convert words to indices, ignoring those not in the vocabulary
        indices = [self.vocab[word] for word in sentence if word in self.vocab]

        input_indices = indices[:-1]
        target_indices = indices[1:]
        assert len(input_indices) == len(target_indices)

        actual_len = len(input_indices)  # ≤ seq_len

        # pad to seq_len
        if actual_len < self.seq_len:
            pad_size = self.seq_len - actual_len
            input_indices  = input_indices  + [0] * pad_size
            target_indices = target_indices + [0] * pad_size

        return (
            torch.tensor(input_indices,  dtype=torch.long),
            torch.tensor(target_indices, dtype=torch.long),
            actual_len
        )
        return torch.tensor(indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long), len(indices)


def collate_fn(batch):
    input_seq, target_seq, lengths = zip(*batch)
    input_padded = pad_sequence(input_seq, batch_first=True, padding_value=0)  # PAD index: 0
    target_padded = pad_sequence(target_seq, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return input_padded, target_padded, lengths

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, padding_idx=0):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, max_len):
        emb = self.embedding(x)
        padded_emb = pack_padded_sequence(emb, max_len, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(padded_emb)
        # Unpack the output
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=x.size(1))
        out = self.fc(out)
        return out


def train_rnn_model(dataset, vocab_size, embedding_dim=100, hidden_size=128, num_epochs=2, batch_size=64, lr=0.01):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = RNNModel(vocab_size, embedding_dim, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(num_epochs):
        loss = 0
        for batch, target, batch_lengths in dataloader:
            optimizer.zero_grad()
            outputs = model(batch, batch_lengths)
            # Flatten the output and target for CrossEntropyLoss
            outputs = outputs.view(-1, outputs.size(-1))
            target = target.view(-1)
            # Calculate loss
            batch_loss = criterion(outputs, target)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss / len(dataloader):.4f}")
    return model

