import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SimpleTransformer
from preprocess import TextDataset, collate_fn
import os
from tqdm import tqdm

def train_model(data_file='Data/commands_responses.json', save_path='model_checkpoint.pth', 
                d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                batch_size=32, num_epochs=10, learning_rate=0.0001):
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct absolute paths
    data_file_path = os.path.join(current_dir, data_file)
    save_path = os.path.join(current_dir, save_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize dataset and dataloader
    dataset = TextDataset(data_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = SimpleTransformer(
        vocab_size=len(dataset.response_vocab),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.response_vocab.stoi[dataset.response_vocab.PAD_TOKEN])

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for src, tgt in progress_bar:
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])  # Exclude the last token of target for input
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))  # Exclude the first token of target for loss calculation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f'Model checkpoint saved at {save_path}')

    print("Training completed.")

if __name__ == "__main__":
    train_model()