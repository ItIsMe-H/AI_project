import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess import TextDataset, collate_fn
from model import SimpleTransformer

def train_model():
    # Load dataset
    dataset = TextDataset('data/commands_responses.json')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

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
        for src, tgt, src_lengths, tgt_lengths in dataloader:
            # Forward pass
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])  # Adjust tgt to match transformer expectations
            loss = criterion(output.view(-1, vocab_size), tgt[:, 1:].contiguous().view(-1))
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    train_model()
