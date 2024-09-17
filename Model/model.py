# model.py
import torch
import torch.nn as nn

class ImprovedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(ImprovedTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)  # Added dropout layer

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.dropout(src)  # Apply dropout
        tgt = self.dropout(tgt)  # Apply dropout
        output = self.transformer(src, tgt)
        return self.fc_out(output)

# Example usage:
if __name__ == "__main__":
    model = SimpleTransformer(vocab_size=10000, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
    print(model)
