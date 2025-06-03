#!/usr/bin/env python3
"""
cuda_ml.py

This module implements a CUDA-accelerated machine learning component using PyTorch.
"""

# NOTE: This module implements legacy models (MLP, RNN, BiLSTM). 
# For transformer-based scoring, refer to transformer_model.py.

import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple
from variant_utils import (
    password_similarity, 
    extract_pattern_clusters, 
    mine_incremental_patterns, 
    mine_year_patterns
)
from .utils.common_utils import extract_features, MLP

INPUT_DIM = 8
HIDDEN_DIM = 64
OUTPUT_DIM = 6

class PasswordRNN(nn.Module):
    """
    Character-level RNN for password variant prediction.
    
    Processes passwords as sequences of characters and predicts
    modification preference scores for various transformation types.
    """
    def __init__(self, embed_dim, hidden_dim, num_layers, dropout, output_dim=6):
        super(PasswordRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.EmbeddingBag(embed_dim, hidden_dim, sparse=True)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(embedded, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class PasswordBiLSTM(nn.Module):
    """
    Bidirectional LSTM with attention mechanism for password variant prediction.
    
    Provides better understanding of character relationships in both directions
    and focuses on the most relevant parts of the password for predictions.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(PasswordBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim // 2,  # Size halved because bidirectional=True doubles it
                            batch_first=True,
                            bidirectional=True)
        
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (_, _) = self.lstm(embedded)
        attn_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        attn_weights = attn_weights.unsqueeze(-1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        output = self.fc(context)
        return output

def create_parallel_model(model, min_dataset_size=10000):
    """Create a DataParallel model if multiple GPUs are available."""
    if not torch.cuda.is_available():
        return model
        
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if min_dataset_size < 10000:
            print("Dataset too small for parallel training - using single GPU")
            return model.cuda()
        else:
            print(f"Using {num_gpus} GPUs for parallel training")
            return nn.DataParallel(model).cuda()
    else:
        return model.cuda()

def load_passwords(file_path: str, max_count: int = None) -> List[str]:
    """
    Load passwords from a file.
    
    Args:
        file_path (str): Path to the password file.
        max_count (int, optional): Maximum number of passwords to load.
        
    Returns:
        List[str]: List of passwords.
    """
    passwords = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                password = line.strip()
                if password:
                    passwords.append(password)
                    if max_count and len(passwords) >= max_count:
                        break
    except Exception as e:
        print(f"Error loading passwords: {e}")
    return passwords

def load_ml_model(config=None, model_type="bilstm"):
    """
    Load the appropriate ML model based on configuration and type.
    """
    model_dir = os.path.join("models", model_type)
    model_path = os.path.join(model_dir, f"{model_type}_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get parameters from config with defaults
    vocab_size = 128  # ASCII character set
    embed_dim = config.get("model_embed_dim", 32) if config else 32
    hidden_dim = config.get("model_hidden_dim", 64) if config else 64
    output_dim = config.get("model_output_dim", 6) if config else 6
    num_layers = config.get("model_num_layers", 2) if config else 2
    dropout = config.get("model_dropout", 0.2) if config else 0.2
    
    if model_type.lower() == "bilstm":
        model = PasswordBiLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
    elif model_type.lower() == "rnn":
        model = PasswordRNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
    else:  # Fallback to MLP
        input_dim = config.get("model_input_dim", 8) if config else 8
        model = MLP(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim,
            dropout=dropout
        ).to(device)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded {model_type.upper()} model from {model_path}")
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
    
    return model

def save_ml_model(model, model_type="rnn"):
    model_dir = os.path.join("models", model_type)
    model_path = os.path.join(model_dir, f"{model_type}_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def text_to_tensor(text, max_length=20, device="cpu"):
    """
    Convert text to a tensor of character indices.
    
    Args:
        text (str): Input password string
        max_length (int): Maximum length to consider
        device (str): Device to place tensor on
        
    Returns:
        torch.Tensor: Tensor of character indices
    """
    # Convert to ASCII indices, limit to 7-bit ASCII
    char_indices = [ord(c) % 128 for c in text[:max_length]]
    # Pad to max_length
    if len(char_indices) < max_length:
        char_indices += [0] * (max_length - len(char_indices))
    return torch.tensor([char_indices], dtype=torch.long, device=device)

def extract_sequence_features(variant, base, max_length=20, device="cpu"):
    """
    Extract character-level features for RNN processing.
    
    Args:
        variant (str): Password variant
        base (str): Original base password
        max_length (int): Maximum sequence length
        device (str): Device to place tensors on
        
    Returns:
        tuple: (variant_tensor, base_tensor) for model input
    """
    variant_tensor = text_to_tensor(variant, max_length, device)
    base_tensor = text_to_tensor(base, max_length, device)
    return variant_tensor, base_tensor

def extract_sequence_batch(variants, bases, max_length=20, device="cpu"):
    """
    Extract features for a batch of variant-base pairs.
    
    Args:
        variants (list): List of variant passwords
        bases (list): List of corresponding base passwords
        max_length (int): Maximum sequence length
        device (str): Device to place tensors on
        
    Returns:
        torch.Tensor: Batch tensor of shape (batch_size, seq_length)
    """
    batch_size = len(variants)
    variant_batch = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    base_batch = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    
    for i, (variant, base) in enumerate(zip(variants, bases)):
        # Convert characters to ASCII indices
        variant_indices = [ord(c) % 128 for c in variant[:max_length]]
        base_indices = [ord(c) % 128 for c in base[:max_length]]
        
        # Pad sequences
        variant_indices += [0] * (max_length - len(variant_indices))
        base_indices += [0] * (max_length - len(base_indices))
        
        # Add to batch
        variant_batch[i] = torch.tensor(variant_indices, dtype=torch.long)
        base_batch[i] = torch.tensor(base_indices, dtype=torch.long)
    
    return variant_batch, base_batch

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

def train_self_supervised(model, training_pairs, epochs=5, batch_size=64, lr=0.001):
    """
    Train the model in a self-supervised manner.
    
    Args:
        model (nn.Module): The model to train.
        training_pairs (List[Tuple[str, str, float]]): Training pairs with confidence scores.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
        
    Returns:
        nn.Module: Trained model.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    device = next(model.parameters()).device
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i in range(0, len(training_pairs), batch_size):
            batch = training_pairs[i:i + batch_size]
            batch_loss = 0.0
            for base, variant, confidence in batch:
                features = extract_features(variant, base, device=device)
                target = torch.tensor([confidence] * OUTPUT_DIM, device=device)
                optimizer.zero_grad()
                prediction = model(features)
                loss = loss_fn(prediction, target)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            total_loss += batch_loss
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(training_pairs):.6f}")
    
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

def predict_with_rnn(variant_tensor, model):
    """
    Get predictions from RNN model.
    
    Args:
        variant_tensor (torch.Tensor): Tensor of variant character indices
        model (PasswordRNN): RNN model
        
    Returns:
        torch.Tensor: Prediction tensor
    """
    with torch.no_grad():
        return model(variant_tensor)

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

def generate_realistic_training_data(passwords: List[str]) -> List[Tuple[str, str, float]]:
    """Generate realistic training pairs with confidence scores."""
    training_pairs = []
    
    # Extract year-based patterns (high confidence)
    year_pairs = mine_year_patterns(passwords)
    training_pairs.extend([(pw1, pw2, 1.0) for pw1, pw2 in year_pairs])
    
    # Extract incremental patterns (high confidence)
    incr_pairs = mine_incremental_patterns(passwords)
    training_pairs.extend([(pw1, pw2, 0.9) for pw1, pw2 in incr_pairs])
    
    # Find other likely variants with lower confidence
    pattern_clusters = extract_pattern_clusters(passwords)
    for pattern, cluster in pattern_clusters.items():
        if len(cluster) < 5 or len(cluster) > 100:  # Skip too small/large clusters
            continue
            
        # Sample pairs from the cluster (limit to avoid O(n²))
        sampled_pairs = []
        for i, pw1 in enumerate(cluster[:20]):  # Limit to first 20 per cluster
            for pw2 in cluster[i+1:i+5]:  # And just a few comparisons per password
                similarity = password_similarity(pw1, pw2)
                if similarity > 0.7:
                    sampled_pairs.append((pw1, pw2, similarity * 0.8))  # Scale confidence
        
        # Take only the top N most similar pairs from this cluster
        sampled_pairs.sort(key=lambda x: x[2], reverse=True)
        training_pairs.extend(sampled_pairs[:10])
    
    return training_pairs

if __name__ == "__main__":
    test_base = "Summer2020"
    test_variant = "Summer2021!"
    model = load_ml_model()
    feats = extract_features(test_variant, test_base)
    pred = predict_config_adjustment(feats, model)
    print("Predicted single-value adjustment:", pred)
    dummy_data = [(test_base, test_variant, {"Numeric": 0.1, "Symbol": 0.1, "Capitalization": 0.1, 
                                        "Leet": 0.1, "Shift": 0.1, "Repetition": 0.1})]
    model = train_model(dummy_data, model, epochs=3)
    save_ml_model(model)