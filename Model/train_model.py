import torch
import torch.optim as optim
from model import SimpleTransformer
from preprocess import TextDataset

def train_model():
    # Initialize dataset and dataloader
    dataset = TextDataset('data/commands_responses.json')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = SimpleTransformer(vocab_size=len(dataset.vocab), d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.vocab[dataset.PAD_TOKEN])  # Use PAD token id from dataset

    for epoch in range(10):  # Number of epochs
        for src, tgt in dataloader:
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}, Loss: {loss.item()}')

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True)
    return src_batch, tgt_batch

if __name__ == "__main__":
    train_model()
