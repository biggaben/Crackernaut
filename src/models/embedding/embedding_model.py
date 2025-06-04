# embedding_model.py
import torch
import torch.nn as nn


class PasswordEmbedder(nn.Module):
    def __init__(self, vocab_size=128, embed_dim=64, max_length=20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.max_length = max_length
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = x.mean(dim=1)  # Average over sequence length
        return self.fc(x)