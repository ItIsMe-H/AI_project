import json
import torch
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

# Define special tokens
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.stoi = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return text.split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi[UNK_TOKEN]
            for token in tokenized_text
        ]

class TextDataset(Dataset):
    # ... (other methods remain the same)

    @staticmethod
    def load_data(data_file):
        try:
            with open(data_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            raise Exception(f"Data file {data_file} not found. Current working directory: {os.getcwd()}")
        except json.JSONDecodeError:
            raise Exception(f"Error decoding JSON from {data_file}.")

    @staticmethod
    def load_data(data_file):
        try:
            with open(data_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            raise Exception(f"Data file {data_file} not found.")
        except json.JSONDecodeError:
            raise Exception(f"Error decoding JSON from {data_file}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        command = self.data[index]['command']
        response = self.data[index]['response']

        numericalized_command = [self.command_vocab.stoi[SOS_TOKEN]] + \
                                self.command_vocab.numericalize(command) + \
                                [self.command_vocab.stoi[EOS_TOKEN]]

        numericalized_response = [self.response_vocab.stoi[SOS_TOKEN]] + \
                                 self.response_vocab.numericalize(response) + \
                                 [self.response_vocab.stoi[EOS_TOKEN]]

        return torch.tensor(numericalized_command), torch.tensor(numericalized_response)

def collate_fn(batch):
    commands, responses = zip(*batch)
    commands_padded = pad_sequence(commands, batch_first=True, padding_value=0)
    responses_padded = pad_sequence(responses, batch_first=True, padding_value=0)
    return commands_padded, responses_padded