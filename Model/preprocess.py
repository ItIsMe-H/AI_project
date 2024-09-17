import json
import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad

class TextDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'r') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        command = item['command']
        response = item['response']
        # Convert command and response to tensor format
        command_tensor = torch.tensor(self.text_to_tensor(command), dtype=torch.long)
        response_tensor = torch.tensor(self.text_to_tensor(response), dtype=torch.long)
        return command_tensor, response_tensor

    def text_to_tensor(self, text):
        # Simple text-to-tensor conversion (this should be adjusted for real scenarios)
        return [ord(c) for c in text]  # Converting each character to its ASCII value

