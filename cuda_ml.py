#!/usr/bin/env python3
"""
cuda_ml.py

This module implements a CUDA-accelerated machine learning component using PyTorch.
"""

import os
import re
import torch
import torch.nn as nn
import torch.optim as optim

MODEL_FILE = "ml_model.pth"

INPUT_DIM = 8
HIDDEN_DIM = 64  # Updated to match intended architecture
OUTPUT_DIM = 6

class MLP(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=6, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class RNNModel(nn.Module):
    def __init__(self, vocab_size=128, embed_dim=32, hidden_dim=64, output_dim=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        output = self.fc(hidden[-1])
        return self.relu(output)

def load_ml_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    if os.path.exists(MODEL_FILE):
        try:
            model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
            print(f"ML model loaded from {MODEL_FILE}")
        except Exception as e:
            print(f"Error loading ML model: {e}")
    return model

def save_ml_model(model):
    try:
        torch.save(model.state_dict(), MODEL_FILE)
        print("ML model saved to", MODEL_FILE)
    except Exception as e:
        print("Error saving ML model:", e)

def extract_features(variant: str, base: str, device: str = "cpu") -> torch.Tensor:
    """
    Extracts an 8D feature vector from a (base, variant) pair on the specified device.
    """
    import string
    
    def numeric_suffix_len(s: str) -> int:
        match = re.search(r"(\d+)$", s)
        return len(match.group(1)) if match else 0

    f1 = abs(numeric_suffix_len(variant) - numeric_suffix_len(base))
    f2 = sum(ch in string.punctuation for ch in variant)
    
    min_len = min(len(base), len(variant))
    cap_diff = sum(1 for i in range(min_len) if base[i].islower() != variant[i].islower())
    f3 = float(cap_diff)
    
    leet_chars = {"0", "3", "1", "$"}
    f4 = sum(ch in leet_chars for ch in variant)
    
    m_b = re.search(r"\d+$", base)
    m_v = re.search(r"^\d+", variant)
    f5 = 1 if m_b and m_v and m_b.group() == m_v.group() else 0
    
    f6 = abs(len(variant) - len(base))
    
    symbol_positions = [i for i, c in enumerate(variant) if c in string.punctuation]
    f7 = len(symbol_positions) / len(variant) if variant else 0
    
    total_leet = sum(1 for c in variant if c in leet_chars)
    f8 = total_leet / len(variant) if variant else 0
    
    features = [f1, f2, f3, f4, f5, f6, f7, f8]
    return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)

def train_model(training_data, model, epochs=10):
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    features = []
    targets = []
    for base, variant, target_dict in training_data:
        feats = extract_features(variant, base)
        features.append(feats)
        target = torch.tensor([
            target_dict["Numeric"], target_dict["Symbol"],
            target_dict["Capitalization"], target_dict["Leet"],
            target_dict["Shift"], target_dict["Repetition"]
        ])
        targets.append(target)

    X = torch.cat(features)
    y = torch.stack(targets)

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.6f}")

    model.eval()
    return model

def predict_config_adjustment(features, model):
    """
    Predict configuration adjustments using the ML model.
    
    Args:
        features (torch.Tensor): Feature tensor of shape (batch_size, input_dim).
        model: Loaded MLP model.
    
    Returns:
        torch.Tensor: Predictions tensor of shape (batch_size, output_dim), where
                      batch_size is the number of variants and output_dim is 6.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        feats = features.to(device)
        output = model(feats)
    return output

def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor):
    with torch.no_grad():
        mae = torch.abs(preds - targets).mean(dim=0)
        return {
            "Numeric_MAE": mae[0].item(),
            "Symbol_MAE": mae[1].item(),
            "Capitalization_MAE": mae[2].item(),
            "Leet_MAE": mae[3].item(),
            "Shift_MAE": mae[4].item(),
            "Repetition_MAE": mae[5].item()
        }

if __name__ == "__main__":
    test_base = "Summer2020"
    test_variant = "Summer2021!"
    model = load_ml_model()
    feats = extract_features(test_variant, test_base)
    pred = predict_config_adjustment(feats, model)
    print("Predicted single-value adjustment:", pred)
    dummy_data = [(test_base, test_variant, 0.1)]
    model = train_model(dummy_data, model, epochs=3)
    save_ml_model(model)