import json
import torch
from torch.utils.data import Dataset

# Define special tokens and a default vocabulary
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
vocab = {UNK_TOKEN: 0, PAD_TOKEN: 1}
inverse_vocab = {0: UNK_TOKEN, 1: PAD_TOKEN}

def build_vocab(data_file):
    global vocab, inverse_vocab
    with open(data_file, 'r') as file:
        data = json.load(file)
    
    for item in data:
        command = item['command']
        response = item['response']
        for text in [command, response]:
            for word in text.split():
                if word not in vocab:
                    index = len(vocab)
                    vocab[word] = index
                    inverse_vocab[index] = word

def tokenize(text, vocab):
    return [vocab.get(word, vocab[UNK_TOKEN]) for word in text.split()]

def pad_sequence(sequence, max_len, pad_token_id):
    return sequence + [pad_token_id] * (max_len - len(sequence))

class TextDataset(Dataset):
    def __init__(self, data_file):
        # Build vocabulary from dataset
        build_vocab(data_file)
        
        # Load data
        with open(data_file, 'r') as file:
            self.data = json.load(file)
        
        # Determine max sequence length
        self.max_len = max(max(len(tokenize(item['command'], vocab)), len(tokenize(item['response'], vocab))) for item in self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        command = item['command']
        response = item['response']
        command_tensor = torch.tensor(pad_sequence(tokenize(command, vocab), self.max_len, vocab[PAD_TOKEN]), dtype=torch.long)
        response_tensor = torch.tensor(pad_sequence(tokenize(response, vocab), self.max_len, vocab[PAD_TOKEN]), dtype=torch.long)
        return command_tensor, response_tensor
