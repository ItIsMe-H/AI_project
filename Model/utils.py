import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SimpleTransformer
from preprocess import TextDataset, collate_fn
import os
from tqdm import tqdm
from preprocess import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

def generate_response(model, command, dataset, device, max_length=50):
    model.eval()
    command = torch.tensor(dataset.vocab.numericalize(command)).unsqueeze(0).to(device)
    
    # Initialize the decoder input with SOS token
    decoder_input = torch.tensor([[dataset.vocab.stoi[SOS_TOKEN]]]).to(device)
    
    for _ in range(max_length):
        output = model(command, decoder_input)
        _, predicted = output.max(2)
        
        last_token = predicted[:, -1].item()
        decoder_input = torch.cat([decoder_input, torch.tensor([[last_token]]).to(device)], dim=1)
        
        if last_token == dataset.vocab.stoi[EOS_TOKEN]:
            break
    
    return ' '.join([dataset.vocab.itos[token.item()] for token in decoder_input[0][1:]])

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = output.max(2)
            correct_predictions += (predicted == tgt[:, 1:]).sum().item()
            total_predictions += (tgt[:, 1:] != dataset.vocab.stoi[PAD_TOKEN]).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy