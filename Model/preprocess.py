# preprocess.py
import json
import torch
from torch.utils.data import Dataset
from collections import Counter
import re

class TextDataset(Dataset):
    def __init__(self, data_file, vocab_size=10000, max_length=50):
        # Load data from JSON file
        with open(data_file, 'r') as file:
            self.data = json.load(file)
        
        # Tokenization and Vocabulary creation
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab = self.build_vocab()
        self.pad_token_id = self.vocab['<pad>']
        self.tokenizer = self.get_tokenizer()

    def build_vocab(self):
        # Build vocabulary from dataset
        counter = Counter()
        for item in self.data:
            command = item['command']
            response = item['response']
            tokens = self.tokenizer(command) + self.tokenizer(response)
            counter.update(tokens)
        
        # Create vocab with special tokens
        vocab = {word: idx + 1 for idx, (word, _) in enumerate(counter.most_common(self.vocab_size - 1))}
        vocab['<pad>'] = 0  # Add padding token
        return vocab

    def get_tokenizer(self):
        # Simple whitespace tokenizer
        def tokenizer(text):
            return re.findall(r'\S+', text)
        return tokenizer

    def encode(self, text):
        # Convert text to list of token IDs
        tokens = self.tokenizer(text)
        return [self.vocab.get(token, self.vocab.get('<unk>', 0)) for token in tokens]

    def pad_sequence(self, seq):
        # Pad or truncate sequence to max_length
        if len(seq) < self.max_length:
            seq += [self.pad_token_id] * (self.max_length - len(seq))
        else:
            seq = seq[:self.max_length]
        return seq

    def __len__(self):
        # Return the number of items in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve item at index `idx`
        item = self.data[idx]
        command = item['command']
        response = item['response']
        
        command_ids = self.encode(command)
        response_ids = self.encode(response)
        
        # Pad sequences
        command_ids = self.pad_sequence(command_ids)
        response_ids = self.pad_sequence(response_ids)
        
        command_tensor = torch.tensor(command_ids, dtype=torch.long)
        response_tensor = torch.tensor(response_ids, dtype=torch.long)
        return command_tensor, response_tensor
