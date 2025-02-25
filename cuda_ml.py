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
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import Levenshtein  # pip install python-Levenshtein

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

class PasswordRNN(nn.Module):
    """
    Character-level RNN for password variant prediction.
    
    Processes passwords as sequences of characters and predicts
    modification preference scores for various transformation types.
    """
    def __init__(self, vocab_size=128, embed_dim=32, hidden_dim=64, output_dim=6, 
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass through the RNN.
        
        Args:
            x: Tensor of shape (batch_size, seq_len) with character indices
            
        Returns:
            Tensor of shape (batch_size, output_dim) with modification scores
        """
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embed_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Use the final hidden state - shape (num_layers, batch_size, hidden_dim)
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        dropped = self.dropout(last_hidden)
        output = self.fc(dropped)
        return output

class PasswordBiLSTM(nn.Module):
    """
    Bidirectional LSTM with attention mechanism for password variant prediction.
    
    Provides better understanding of character relationships in both directions
    and focuses on the most relevant parts of the password for predictions.
    """
    def __init__(self, vocab_size=128, embed_dim=32, hidden_dim=64, output_dim=6, 
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim // 2,  # Half size for each direction
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # Pass through BiLSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim)
        
        # Apply attention
        attn_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)  # (batch_size, seq_len)
        attn_weights = attn_weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Compute weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch_size, hidden_dim)
        
        # Final prediction
        dropped = self.dropout(context)
        output = self.fc(dropped)
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

# Move model parameters to configuration
def load_ml_model(config=None, model_type="bilstm"):
    """
    Load the appropriate ML model based on configuration and type.
    """
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
    
    # Try to load saved model
    model_path = f"{model_type}_model.pth"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded {model_type.upper()} model from {model_path}")
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
    
    return model

def save_ml_model(model, model_type="rnn"):
    """
    Save ML model with appropriate filename based on model type.
    
    Args:
        model (nn.Module): PyTorch model to save
        model_type (str): Type of model ("mlp" or "rnn")
    """
    if isinstance(model, PasswordRNN):
        model_type = "rnn"
    elif isinstance(model, MLP):
        model_type = "mlp"
    
    model_path = f"{model_type}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

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

def basic_increment(base: str) -> Set[str]:
    # Implementation
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

def find_likely_variants(passwords, similarity_threshold=0.7):
    """Find passwords that are likely variants of each other in breach data."""
    variant_pairs = []
    for i, pw1 in enumerate(passwords):
        for pw2 in passwords[i+1:]:
            # Various similarity metrics: Levenshtein distance, common patterns, etc.
            if password_similarity(pw1, pw2) > similarity_threshold:
                variant_pairs.append((pw1, pw2))
    return variant_pairs

# Implement in train_on_wordlist:
def train_on_wordlist_self_supervised(model, wordlist_path):
    passwords = load_passwords(wordlist_path)
    likely_pairs = find_likely_variants(passwords)
    
    for base, variant in likely_pairs:
        # Train model to predict the transformation between these pairs
        train_transformation_model(model, base, variant)

def generate_realistic_training_data(passwords):
    # For passwords that follow common patterns (e.g., word+year, name+number)
    # Create plausible variants by simulating common user behaviors
    training_pairs = []
    for password in passwords:
        if re.search(r'\d{4}$', password):  # Ends with 4 digits (likely a year)
            year = int(re.search(r'(\d{4})$', password).group(1))
            base = password[:-4]
            variants = [f"{base}{year+1}", f"{base}{year-1}"]
            training_pairs.extend([(password, variant) for variant in variants])
    return training_pairs

def score_variants(variants, base, model, config, device):
    # Get RNN-based scores
    rnn_scores = get_rnn_scores(variants, base, model, device)
    
    # Get traditional rule-based scores
    rule_scores = get_rule_based_scores(variants, base, config)
    
    # Combine scores with configurable weighting
    alpha = config.get("rnn_weight", 0.7)  # How much to weight the RNN vs rules
    final_scores = alpha * rnn_scores + (1-alpha) * rule_scores
    
    return list(zip(variants, final_scores))

def evaluate_realism(model, test_passwords, human_variant_pairs=None):
    # Generate variants for test passwords
    generated_variants = []
    for password in test_passwords:
        variants = generate_variants(password, model)
        generated_variants.append((password, variants))
    
    # Compare with known human variants if available
    if human_variant_pairs:
        accuracy = compare_with_human_data(generated_variants, human_variant_pairs)
        print(f"Variant prediction accuracy: {accuracy:.2f}")
    
    # Evaluate linguistic realism using n-gram analysis
    realism_score = linguistic_realism_score(generated_variants)
    print(f"Linguistic realism score: {realism_score:.2f}")

def extract_pattern_clusters(passwords, min_cluster_size=3, max_sample=100000):
    """
    Group passwords by common structural patterns.
    
    Args:
        passwords: List of passwords to analyze
        min_cluster_size: Minimum number of passwords to form a valid cluster
        max_sample: Maximum number of passwords to process for memory efficiency
    
    Returns:
        dict: Mapping from patterns to lists of matching passwords
    """
    # For large files, take a random sample to keep memory usage reasonable
    if len(passwords) > max_sample:
        import random
        random.seed(42)  # For reproducibility
        passwords = random.sample(passwords, max_sample)
    
    pattern_map = {}
    
    for password in passwords:
        # Create a structural fingerprint (e.g., "LLLDDDS" for "abc123!")
        pattern = "".join('L' if c.isalpha() else 
                          'D' if c.isdigit() else 
                          'S' for c in password)
        
        if len(pattern) < 4:  # Skip very short patterns
            continue
            
        if pattern not in pattern_map:
            pattern_map[pattern] = []
        pattern_map[pattern].append(password)
    
    # Return only patterns with sufficient examples
    return {k: v for k, v in pattern_map.items() if len(v) >= min_cluster_size}

def find_year_variants(passwords):
    """Extract year-based password variants which are common in real-world data."""
    year_base_map = {}
    year_pattern = re.compile(r'^(.+?)(\d{4})$')
    
    for password in passwords:
        match = year_pattern.match(password)
        if match:
            base, year = match.groups()
            key = base.lower()  # Case-insensitive matching
            if key not in year_base_map:
                year_base_map[key] = {}
            
            # Group by base, counting occurrences of each year
            year_base_map[key][year] = year_base_map[key].get(year, 0) + 1
    
    # Find bases with multiple years (strong indicator of variants)
    variants = []
    for base, years in year_base_map.items():
        if len(years) >= 2:  # At least two different years for the same base
            actual_passwords = [f"{base}{year}" for year in years]
            for i, pw1 in enumerate(actual_passwords):
                for pw2 in actual_passwords[i+1:]:
                    variants.append((pw1, pw2, 1.0))  # Third value is confidence
    
    return variants

def load_passwords(wordlist_path, max_lines=1000000):
    """Load password list with reasonable limits."""
    try:
        passwords = []
        with open(wordlist_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                pw = line.strip()
                if pw:
                    passwords.append(pw)
        return passwords
    except Exception as e:
        print(f"Error loading passwords: {e}")
        return []

def load_passwords(wordlist_path: str, max_count: int = 100000) -> List[str]:
    """Load passwords from a wordlist file with limit."""
    try:
        with open(wordlist_path, 'r', encoding='utf-8', errors='ignore') as f:
            return [line.strip() for line in f.readlines()[:max_count] if line.strip()]
    except Exception as e:
        print(f"Error loading wordlist: {e}")
        return []

def password_similarity(pw1: str, pw2: str) -> float:
    """Calculate password similarity using multiple metrics."""
    # Normalize lengths for comparison
    max_len = max(len(pw1), len(pw2))
    if max_len == 0:
        return 0.0
        
    # Calculate Levenshtein (edit) distance similarity
    edit_similarity = 1.0 - (Levenshtein.distance(pw1, pw2) / max_len)
    
    # Calculate character overlap similarity
    common_chars = set(pw1).intersection(set(pw2))
    char_similarity = len(common_chars) / len(set(pw1).union(set(pw2))) if pw1 and pw2 else 0
    
    # Pattern similarity (e.g. both end with numbers)
    pattern_similarity = 0.0
    if re.search(r'\d+$', pw1) and re.search(r'\d+$', pw2):
        pattern_similarity += 0.2
    if pw1.lower().startswith(pw2.lower()[:3]) or pw2.lower().startswith(pw1.lower()[:3]):
        pattern_similarity += 0.2
        
    # Combined similarity score (weighted)
    return (0.5 * edit_similarity) + (0.3 * char_similarity) + (0.2 * pattern_similarity)

def extract_pattern_clusters(passwords: List[str], min_cluster_size: int = 3) -> Dict[str, List[str]]:
    """Group passwords by common structural patterns."""
    pattern_map = defaultdict(list)
    
    for password in passwords:
        if not password:
            continue
            
        # Create a structural fingerprint (e.g., "LLLDDDS" for "abc123!")
        pattern = ''.join('L' if c.isalpha() else 
                         'D' if c.isdigit() else 
                         'S' for c in password)
        
        # Add length information to the pattern
        length_pattern = f"{pattern}_{len(password)}"
        pattern_map[length_pattern].append(password)
    
    # Return only patterns with sufficient examples
    return {k: v for k, v in pattern_map.items() if len(v) >= min_cluster_size}

def mine_year_patterns(passwords: List[str]) -> List[Tuple[str, str]]:
    """Extract high-confidence training pairs from year patterns."""
    # Find passwords ending with 4 digits (likely years)
    year_pattern = re.compile(r'^(.+?)(\d{4})$')
    
    # Group by base part
    password_map = defaultdict(list)
    for password in passwords:
        match = year_pattern.match(password)
        if match:
            base, year = match.groups()
            if len(year) == 4 and 1900 <= int(year) <= 2030:  # Validate as plausible year
                password_map[base].append(year)
    
    # Find bases with multiple years (strong indicator of variants)
    training_pairs = []
    for base, years in password_map.items():
        if len(years) >= 2:
            # Create pairs from each year variant
            base_passwords = [f"{base}{year}" for year in years]
            for i, pw1 in enumerate(base_passwords):
                for pw2 in base_passwords[i+1:]:
                    training_pairs.append((pw1, pw2))
    
    return training_pairs

def mine_incremental_patterns(passwords: List[str]) -> List[Tuple[str, str]]:
    """Find passwords with incremental number patterns."""
    training_pairs = []
    
    # Group passwords by their alphabetic prefix
    alpha_groups = defaultdict(list)
    for password in passwords:
        match = re.match(r'^([a-zA-Z]+)(\d+)$', password)
        if match:
            prefix, number = match.groups()
            alpha_groups[prefix.lower()].append((password, int(number)))
    
    # Find sequential numbers with the same prefix
    for prefix, pw_numbers in alpha_groups.items():
        if len(pw_numbers) < 2:
            continue
            
        # Sort by the numeric part
        pw_numbers.sort(key=lambda x: x[1])
        
        # Look for sequential or nearby numbers
        for i in range(len(pw_numbers) - 1):
            current_pw, current_num = pw_numbers[i]
            next_pw, next_num = pw_numbers[i + 1]
            
            # Check if numbers are close (e.g., 123 and 124)
            if 0 < next_num - current_num <= 5:
                training_pairs.append((current_pw, next_pw))
    
    return training_pairs

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

from torch.cuda.amp import autocast, GradScaler

def train_self_supervised(model, data_pairs, epochs=5, batch_size=32, lr=0.001, use_amp=True):
    """
    Train model using self-supervised learning from extracted password pairs.
    
    Args:
        model: PyTorch model
        data_pairs: List of (password1, password2, confidence) tuples
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Trained model
    """
    if not data_pairs:
        print("No training pairs provided")
        return model
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Setup for mixed precision training
    scaler = GradScaler() if use_amp and torch.cuda.is_available() else None
    
    # Convert training data to appropriate format
    all_pw1 = [pair[0] for pair in data_pairs]
    all_pw2 = [pair[1] for pair in data_pairs]
    all_confidences = torch.tensor([pair[2] for pair in data_pairs], 
                                 dtype=torch.float32, device=device)
    
    # Training loop
    num_batches = (len(data_pairs) + batch_size - 1) // batch_size
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Shuffle data
        indices = torch.randperm(len(data_pairs))
        
        # Process batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(data_pairs))
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            batch_pw1 = [all_pw1[idx] for idx in batch_indices]
            batch_pw2 = [all_pw2[idx] for idx in batch_indices]
            batch_confidence = all_confidences[batch_indices]
            
            # Convert passwords to tensors
            pw1_tensor, pw2_tensor = extract_sequence_batch(
                batch_pw1, batch_pw2, device=device
            )
            
            # Forward pass with mixed precision if enabled
            optimizer.zero_grad()
            
            if use_amp and torch.cuda.is_available():
                with autocast():
                    # Process both passwords through the model
                    output1 = model(pw1_tensor)
                    output2 = model(pw2_tensor)
                    
                    # Calculate similarity in output space (cosine similarity)
                    similarity = F.cosine_similarity(output1, output2)
                    
                    # Loss: predicted similarity should match confidence
                    loss = F.mse_loss(similarity, batch_confidence)
                
                # Backward pass with mixed precision
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                output1 = model(pw1_tensor)
                output2 = model(pw2_tensor)
                similarity = F.cosine_similarity(output1, output2)
                loss = F.mse_loss(similarity, batch_confidence)
                
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        # End of epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)
    
    # Return trained model
    model.eval()
    return model