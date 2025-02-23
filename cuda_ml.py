#!/usr/bin/env python3
"""
cuda_ml.py

This module implements a CUDA-accelerated machine learning component using PyTorch.
It is designed to help adjust configuration parameters for the Crackernaut utility 
based on human feedback during training.

Functions:
    load_ml_model() -> model: Loads a saved ML model from disk, or creates a new one.
    save_ml_model(model): Saves the given model to disk.
    extract_features(variant: str, base: str) -> torch.Tensor: Extracts a feature vector
        representing differences between a base password and a variant.
    train_model(training_data, current_config, model, epochs=10) -> model:
        Trains the model on provided training data.
    predict_config_adjustment(features, model) -> float:
        Returns the predicted adjustment based on the input feature vector.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

MODEL_FILE = "ml_model.pth"

# Define the input dimension for our feature vector.
# For this initial version, we use 6 features:
# 1. Numeric difference (if any)
# 2. Count of punctuation characters (symbols) in variant
# 3. Capitalization difference (number of characters that differ in case)
# 4. Count of leet substitutions (count of characters that are in our leet mapping)
# 5. Shift indicator (binary: 1 if numeric suffix is shifted from the end, else 0)
# 6. Repetition difference (difference in length beyond base)
INPUT_DIM = 6
HIDDEN_DIM = 16  # You can adjust this size as needed.
OUTPUT_DIM = 1   # Predict a single adjustment value.

class AdjustmentMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
        super(AdjustmentMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x is expected to be a tensor of shape (batch_size, input_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x  # Output shape: (batch_size, 1)

def load_ml_model():
    """
    Loads the ML model from disk if available; otherwise creates a new model.
    Returns:
        model: a PyTorch model with CUDA support if available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdjustmentMLP().to(device)
    if os.path.exists(MODEL_FILE):
        try:
            model.load_state_dict(torch.load(MODEL_FILE))
            print("ML model loaded from", MODEL_FILE)
        except Exception as e:
            print("Error loading ML model:", e)
            print("Using a new model.")
    else:
        print("No existing model found; initializing new model.")
    return model

def save_ml_model(model):
    """
    Saves the PyTorch model to disk.
    """
    try:
        torch.save(model.state_dict(), MODEL_FILE)
        print("ML model saved to", MODEL_FILE)
    except Exception as e:
        print("Error saving ML model:", e)

def extract_features(variant, base):
    """
    Extracts a 6-dimensional feature vector from a variant relative to a base password.
    Features (example design):
        1. Numeric difference: Absolute difference between numeric suffix lengths.
        2. Symbol count: Count of punctuation characters in the variant.
        3. Capitalization difference: Count of positions where case differs.
        4. Leet count: Count of characters that are typical leet substitutions.
        5. Shift indicator: 1 if numeric suffix is not at the very end; 0 otherwise.
        6. Repetition difference: Difference in length between variant and base.
    Returns:
        A torch.Tensor of shape (1, 6) of type float.
    """
    import string
    
    # Feature 1: Numeric difference
    def numeric_diff(s):
        m = re.search(r"(\d+)$", s)
        return len(m.group(1)) if m else 0

    num_base = numeric_diff(base)
    num_variant = numeric_diff(variant)
    numeric_diff_feature = abs(num_variant - num_base)

    # Feature 2: Symbol count (count punctuation)
    symbol_count = sum(1 for ch in variant if ch in string.punctuation)

    # Feature 3: Capitalization difference
    cap_diff = sum(1 for b, v in zip(base, variant) if b.islower() != v.islower())
    
    # Feature 4: Leet substitutions count
    leet_chars = {"0", "3", "1", "$"}
    leet_count = sum(1 for ch in variant if ch in leet_chars)

    # Feature 5: Shift indicator
    # Check if variant ends with digits while base does; if not, then assume shift occurred.
    shift_indicator = 0
    m_base = re.search(r"(\d+)$", base)
    m_variant = re.search(r"(\d+)$", variant)
    if m_base and m_variant:
        if m_variant.group() != m_base.group():
            shift_indicator = 1
    elif m_base and not m_variant:
        shift_indicator = 1

    # Feature 6: Repetition difference (difference in total length)
    repetition_diff = abs(len(variant) - len(base))

    features = [numeric_diff_feature, symbol_count, cap_diff, leet_count, shift_indicator, repetition_diff]
    # Normalize features if necessary (for now, we simply cast them to float)
    feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # shape (1, 6)
    return feature_tensor

def train_model(training_data, current_config, model, epochs=10, learning_rate=0.01):
    """
    Trains the ML model using the training data and updates it based on user feedback.
    training_data: list of tuples (base, variant, user_rating)
                   where user_rating is a float representing desired adjustment (e.g., +0.1 or -0.05)
    current_config: current configuration dictionary (not directly used here but may be used for loss calculation)
    model: the PyTorch model to be trained.
    Returns:
        The updated model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()  # Mean Squared Error loss

    # Prepare training data as feature tensors and target adjustments.
    feature_list = []
    target_list = []
    for base, variant, rating in training_data:
        features = extract_features(variant, base).to(device)
        feature_list.append(features)
        # Here, user_rating serves as the target adjustment (scalar)
        target_list.append(torch.tensor([[rating]], dtype=torch.float32).to(device))
    
    # Concatenate features and targets to form batches
    X = torch.cat(feature_list, dim=0)  # shape (num_samples, 6)
    y = torch.cat(target_list, dim=0)   # shape (num_samples, 1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X)  # shape (num_samples, 1)
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    
    model.eval()  # set model to evaluation mode
    return model

def predict_config_adjustment(features, model):
    """
    Given a feature tensor (shape (1,6)), predict the adjustment value.
    Returns a float.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        features = features.to(device)
        adjustment = model(features)
    # Return the scalar adjustment value
    return adjustment.item()

# Exported API functions:
if __name__ == "__main__":
    # Simple testing code
    test_base = "Summer2020"
    test_variant = "Summer2021!"
    feat = extract_features(test_variant, test_base)
    print("Extracted features:", feat)
    
    model = load_ml_model()
    adj = predict_config_adjustment(feat, model)
    print("Predicted adjustment:", adj)
    
    # Simulate a dummy training example: user rated the variant adjustment as +0.1
    training_data = [(test_base, test_variant, 0.1)]
    model = train_model(training_data, DEFAULT_CONFIG, model, epochs=5)
    save_ml_model(model)
