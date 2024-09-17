import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from typing import List

class TextDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'r') as file:
            self.data = json.load(file)
        
        # Create vocabulary and token mappings
        self.vocab = self.build_vocab()
        self.pad_token_id = self.vocab['<pad>']
        
    def build_vocab(self) -> dict:
        # Example vocabulary
        vocab = {'<pad>': 0}
        counter = Counter()
        
        for item in self.data:
            command = item['command']
            response = item['response']
            counter.update(command.split())
            counter.update(response.split())
        
        for word, _ in counter.items():
            if word not in vocab:
                vocab[word] = len(vocab)
        
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        command = item['command'].split()
        response = item['response'].split()
        
        command_tensor = torch.tensor([self.vocab.get(word, 0) for word in command], dtype=torch.long)
        response_tensor = torch.tensor([self.vocab.get(word, 0) for word in response], dtype=torch.long)
        
        return command_tensor, response_tensor

def collate_fn(batch: List[tuple]) -> tuple:
    src_batch, tgt_batch = zip(*batch)
    
    src_lengths = [len(src) for src in src_batch]
    tgt_lengths = [len(tgt) for tgt in tgt_batch]
    
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_padded, tgt_padded, src_lengths, tgt_lengths
