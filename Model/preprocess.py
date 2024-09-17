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
    try:
        with open(data_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        raise Exception(f"Data file {data_file} not found.")
    except json.JSONDecodeError:
        raise Exception(f"Error decoding JSON from {data_file}.")
    
    for item in data:
        command = item.get('command', '')
        response = item.get('response', '')
        for text in [command, response]:
            for word in text.split():
                if word not in vocab:
                    index = len(vocab)
                    vocab[word] = index
                    inverse_vocab[index] = word

def tokenize(text, vocab):
    return [vocab.get(word, vocab[UNK_TOKEN]) for word in text.split()]

def pad_sequence(sequence, max_len, pad_token_id):
    return sequence[:max_len] + [pad_token_id] * (max_len - len(sequence))

class TextDataset(Dataset):
    def __init__(self, data_file, max_len=50):
        # Build vocab and load data
        build_vocab(data_file)
        self.data = []
        with open(data_file, 'r') as file:
            self.raw_data = json.load(file)

        for item in self.raw_data:
            command = tokenize(item['command'], vocab)
            response = tokenize(item['response'], vocab)
            command = pad_sequence(command, max_len, vocab[PAD_TOKEN])
            response = pad_sequence(response, max_len, vocab[PAD_TOKEN])
            self.data.append((command, response))

        self.vocab = vocab
        self.pad_token = vocab[PAD_TOKEN]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

# Collate function for DataLoader
def collate_fn(batch):
    src, tgt = zip(*batch)
    return torch.stack(src), torch.stack(tgt)
