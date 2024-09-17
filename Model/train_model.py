# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SimpleTransformer
from preprocess import TextDataset

def train_model():
    # Load dataset
    dataset = TextDataset('data/commands_responses.json')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Model parameters
    vocab_size = len(dataset.vocab)
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    model = SimpleTransformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers)

    # Training parameters
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(1):  # Change the range for more epochs
        for src, tgt in dataloader:
            # Forward pass
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def collate_fn(batch):
    # Custom collate function for padding
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

if __name__ == "__main__":
    train_model()
