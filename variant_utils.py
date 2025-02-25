import os
import re
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim

from typing import List, Set
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import branin
from cuda_ml import extract_features, MLP
from config_utils import PROJECT_ROOT as PROJECT_ROOT
from collections import deque
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

SYMBOLS = ["!", "@", "#", "$", "%", "^", "&", "*", "?", "-", "+"]

########################################
# Transformation Components
########################################

def _chain_basic_increment(base: str) -> Set[str]:
    variants = set()
    m = re.search(r"^(.*?)(\d+)$", base)
    if m:
        prefix, num_str = m.groups()
        num = int(num_str)
        for inc in [1, 2, 3, 5, 10]:
            variants.add(f"{prefix}{num + inc}")
    else:
        variants.add(f"{base}{random.randint(2000, 2030)}")
    return variants

def _chain_symbol_addition(base: str) -> Set[str]:
    return {f"{base}{sym}" for sym in SYMBOLS} | {f"{sym}{base}" for sym in SYMBOLS}

def _chain_capitalization_tweaks(base: str) -> Set[str]:
    variants = {
        base.capitalize(),
        base.upper(),
        base.lower()
    }
    if len(base) >= 3:
        variants.add(f"{base[:2]}{base[2].upper()}{base[3:]}")
    return variants

def _chain_leet_substitution(base: str) -> Set[str]:
    mapping = {
        "a": ["@", "4"], "e": ["3"], "i": ["1", "!"], "o": ["0"],
        "s": ["$", "5"], "t": ["7"], "z": ["2"], "g": ["9"]
    }
    variants = set()
    for i, ch in enumerate(base.lower()):
        if ch in mapping:
            variants.add(f"{base[:i]}{random.choice(mapping[ch])}{base[i+1:]}")
    return variants

def _chain_shift_variants(base: str) -> Set[str]:
    variants = set()
    if m := re.search(r"^(\D+)(\d+)$", base):
        variants.add(f"{m.group(2)}{m.group(1)}")
    return variants

def _chain_middle_insertion(base: str) -> Set[str]:
    variants = set()
    if len(base) >= 4:
        mid = len(base) // 2
        variants.update({f"{base[:mid]}{sym}{base[mid:]}" for sym in SYMBOLS})
    return variants

def _chain_repetition_variants(base: str) -> Set[str]:
    return {f"{base}{base[-1]}", f"{base}!!"} | {f"{sym*2}{base}" for sym in SYMBOLS}

########################################
# Variant Generation Core
########################################

TRANSFORMATIONS = {
    "numeric": _chain_basic_increment,
    "symbol": _chain_symbol_addition,
    "capitalization": _chain_capitalization_tweaks,
    "leet": _chain_leet_substitution,
    "shift": _chain_shift_variants,
    "repetition": _chain_repetition_variants,
    "middle_insertion": _chain_middle_insertion
}

def generate_variants(base: str, max_length: int, chain_depth: int) -> List[str]:
    """
    Generate password variants iteratively up to a specified chain depth.

    Args:
        base (str): Base password.
        max_length (int): Maximum length of variants.
        chain_depth (int): Maximum transformation depth.

    Returns:
        List[str]: List of unique variants.
    """

    variants = set()
    queue = deque([(base, 0)])  # (variant, depth)
    seen = {base}

    while queue:
        current, depth = queue.popleft()
        if depth > chain_depth:
            continue
        if len(current) <= max_length:
            variants.add(current)

        for transform in TRANSFORMATIONS.values():
            for var in transform(current):
                if var not in seen and len(var) <= max_length:
                    seen.add(var)
                    queue.append((var, depth + 1))

    return list(variants)

def generate_human_chains(base: str) -> set:
    """
    Generate human-like password variants using generalized transformations.

    Args:
        base (str): Base password.

    Returns:
        set: Set of unique variant strings.
    """
    chains = set()
    for transform in TRANSFORMATIONS.values():
        chains.update(transform(base))
    return chains

def generate_variants_parallel(base, max_length=20, chain_depth=5, num_workers=None):
    """
    Generate password variants using multiple CPU cores for improved performance.
    
    Args:
        base: Base password to generate variants from
        max_length: Maximum length of variants to consider
        chain_depth: Maximum depth of transformation chains
        num_workers: Number of worker processes (defaults to CPU count)
    
    Returns:
        List of unique password variants
    """
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)  # Leave one core free for system
    
    # Create partial functions with fixed arguments
    def process_transforms(transform_keys, base=base, max_length=max_length, chain_depth=chain_depth):
        variants = set()
        for key in transform_keys:
            transform_fn = TRANSFORMATIONS[key]
            # Apply each transformation to the base password
            for variant in transform_fn(base, max_length):
                if variant and len(variant) <= max_length:
                    variants.add(variant)
                    
                    # Apply recursive chains if depth allows
                    if chain_depth > 1:
                        for chain_variant in _generate_chain_variants(variant, max_length, chain_depth-1):
                            variants.add(chain_variant)
        
        return list(variants)
    
    # Split transformation keys across workers
    transform_keys = list(TRANSFORMATIONS.keys())
    chunk_size = max(1, len(transform_keys) // num_workers)
    transform_chunks = [transform_keys[i:i+chunk_size] for i in range(0, len(transform_keys), chunk_size)]
    
    # Process in parallel
    all_variants = set([base])  # Include the original password
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(process_transforms, transform_chunks):
            all_variants.update(result)
    
    return list(all_variants)

# Helper function to maintain compatibility with existing code
def _generate_chain_variants(base, max_length, depth):
    # Similar to original chain variant logic but simplified
    variants = set()
    for key, transform_fn in TRANSFORMATIONS.items():
        for variant in transform_fn(base, max_length):
            if variant and len(variant) <= max_length:
                variants.add(variant)
                if depth > 1:
                    for subvariant in _generate_chain_variants(variant, max_length, depth-1):
                        variants.add(subvariant)
    return variants

########################################
# Machine Learning Integration
########################################

def load_and_split_data(dataset_path: str = os.path.join(PROJECT_ROOT, "datasets", "default.csv")) -> tuple[list, list]:
    """Load base passwords from the specified file and generate training/validation data."""
    dataset_path = os.path.normpath(dataset_path)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        bases = [line.strip() for line in f if line.strip()]
    
    if not bases:
        raise ValueError(f"No valid passwords found in dataset: {dataset_path}")
    
    data = []
    for base in bases:
        variants = generate_variants(base, 20, 2)
        if not variants:
            print(f"Warning: No variants generated for base password '{base}'")
            continue
        for var in variants:
            mods = identify_modifications(var, base)
            data.append((
                base,
                var,
                {mod: 1.0 if mod in mods else 0.0 for mod in [
                    "Numeric", "Symbol", "Capitalization", 
                    "Leet", "Shift", "Repetition"
                ]}
            ))
    
    if not data:
        raise ValueError(f"No training data generated from dataset: {dataset_path}")
    
    random.shuffle(data)
    split_idx = int(0.8 * len(data))
    return data[:split_idx], data[split_idx:]

def train_model(train_data, model, epochs: int = 10):
    """Train the model on the provided training data."""
    if not train_data:
        raise ValueError("Training data is empty.")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.SmoothL1Loss()
    
    device = next(model.parameters()).device
    features = []
    targets = []
    for base, var, target in train_data:
        feat = extract_features(var, base, device=device)  # Pass device to extract_features
        if feat is not None:
            features.append(feat)  # Already on device
            target_tensor = torch.tensor(list(target.values()), dtype=torch.float32, device=device)
            targets.append(target_tensor)
        else:
            print(f"Warning: No features extracted for variant '{var}' from base '{base}'")
    
    if not features:
        raise ValueError("No valid features extracted from the training data.")
    
    X = torch.cat(features, dim=0)
    y = torch.stack(targets, dim=0)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
    
    return model

def evaluate_model(model, val_data):
    """Evaluate the model on validation data, ensuring tensors are on the correct device."""
    loss_fn = nn.SmoothL1Loss()
    device = next(model.parameters()).device
    features = []
    targets = []
    
    for base, var, target in val_data:
        feat = extract_features(var, base, device=device)  # Pass device to extract_features
        features.append(feat)  # Already on device
        target_tensor = torch.tensor(list(target.values()), dtype=torch.float32, device=device)
        targets.append(target_tensor)
    
    X_val = torch.cat(features, dim=0)
    y_val = torch.stack(targets, dim=0)
    
    model.eval()  # Ensure evaluation mode
    with torch.no_grad():
        outputs = model(X_val)
        loss = loss_fn(outputs, y_val)
    
    return loss.item()

def train_evaluate(params, dataset_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(
        hidden_dim=params["hidden_dim"], 
        dropout=params["dropout"]
    ).to(device)
    train_data, val_data = load_and_split_data(dataset_path)
    model = train_model(train_data, model)
    return evaluate_model(model, val_data)

########################################
# Optimization Setup
########################################

def optimize_hyperparameters(dataset_path: str) -> dict:
    """Perform Bayesian hyperparameter optimization using Ax."""
    # Debug logging
    print("\nStarting hyperparameter optimization...")
    print(f"Dataset path: {dataset_path}")
    print(f"Path exists: {os.path.exists(dataset_path)}")
    print(f"Path is absolute: {os.path.isabs(dataset_path)}")
    
    ax_client = AxClient(verbose_logging=True)
    print("\nInitialized Ax client")
    
    # 1. Define search space
    print("\nDefining parameter search space...")
    parameters = [
        {
            "name": "hidden_dim",
            "type": "range",
            "bounds": [32, 256],
            "value_type": "int"
        },
        {
            "name": "dropout",
            "type": "range", 
            "bounds": [0.1, 0.5],
            "value_type": "float"
        }
    ]
    print(f"Parameters to optimize: {json.dumps(parameters, indent=2)}")
    
    try:
        print("\nCreating experiment...")
        ax_client.create_experiment(
            name="mlp_optimization",
            parameters=parameters,
            objectives={"validation_loss": ObjectiveProperties(minimize=True)}
        )
        print("Experiment created successfully")
        
        # 2. Run optimization trials with debug info
        print("\nStarting optimization trials...")
        for trial in range(20):
            print(f"\nTrial {trial + 1}/20:")
            parameters, trial_index = ax_client.get_next_trial()
            print(f"Testing parameters: {parameters}")
            
            try:
                result = train_evaluate(parameters, dataset_path)
                print(f"Trial result: {result}")
                ax_client.complete_trial(trial_index=trial_index, raw_data={"validation_loss": result})
            except Exception as e:
                print(f"Trial failed: {e}")
                ax_client.complete_trial(trial_index=trial_index, raw_data={"validation_loss": float('inf')})
        
        # 3. Get and verify best parameters
        best_parameters = ax_client.get_best_parameters()
        print(f"\nOptimization complete. Best parameters found: {json.dumps(best_parameters[0], indent=2)}")
        return best_parameters[0]
        
    except Exception as e:
        print(f"\nFatal error in optimization: {e}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        return {"hidden_dim": 128, "dropout": 0.3}  # Fallback default values

########################################
# Modification Detection
########################################

def identify_modifications(variant: str, base: str) -> List[str]:
    mods = []
    if variant == base:
        return mods
    
    # Numeric check
    base_num = re.search(r'\d+$', base)
    var_num = re.search(r'\d+$', variant)
    if (bool(base_num) != bool(var_num)) or (base_num and var_num and base_num.group() != var_num.group()):
        mods.append("Numeric")
    
    # Symbol check
    if any(variant.startswith(sym) or variant.endswith(sym) for sym in SYMBOLS):
        mods.append("Symbol")
    
    # Capitalization check
    if any(c1 != c2 for c1, c2 in zip(base.lower(), variant.lower())):
        mods.append("Capitalization")
    
    # Leet check
    if any(c in variant for c in {'@', '3', '1', '$', '7', '!'}):
        mods.append("Leet")
    
    # Shift check
    if base_num and not var_num:
        mods.append("Shift")
    
    # Repetition check
    if len(variant) > len(base) and any(variant.endswith(c*2) for c in SYMBOLS + [base[-1]] if base):
        mods.append("Repetition")
    
    return list(set(mods))