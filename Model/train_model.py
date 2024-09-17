import torch
import torch.optim as optim
from model import SimpleTransformer
from preprocess import TextDataset, collate_fn
import os

def train_model(data_file='Data/commands_responses.json', save_path='model_checkpoint.pth'):
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset and dataloader
    dataset = TextDataset(data_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # Initialize model
    model = SimpleTransformer(vocab_size=len(dataset.vocab), d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.pad_token)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), save_path)
            print(f'Model checkpoint saved at {save_path}')
    
    print("Training completed.")
