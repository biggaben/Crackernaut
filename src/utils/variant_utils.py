import os
import re
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim

from typing import List, Set, Dict, Tuple, Optional
from ax.service.ax_client import AxClient, ObjectiveProperties
from .config_utils import PROJECT_ROOT
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from .common_utils import extract_features, MLP

SYMBOLS = ["!", "@", "#", "$", "%", "^", "&", "*", "?", "-", "+"]
CONFIG_FILE = os.path.join(PROJECT_ROOT, "config.json")

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
            for sub in mapping[ch]:
                variants.add(f"{base[:i]}{sub}{base[i+1:]}")
    return variants

def chain_shift_variants(base: str) -> Set[str]:
    """Shift numeric portion of password between front and back"""
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
    if not base:
        return set()  # Handle empty base case
    return {f"{base}{base[-1]}", f"{base}!!"} | {f"{sym*2}{base}" for sym in SYMBOLS}

########################################
# Variant Generation Core
########################################

TRANSFORMATIONS = {
    "numeric": _chain_basic_increment,
    "symbol": _chain_symbol_addition,
    "capitalization": _chain_capitalization_tweaks,
    "leet": _chain_leet_substitution,
    "shift": chain_shift_variants,
    "repetition": _chain_repetition_variants,
    "middle_insertion": _chain_middle_insertion
}

def _process_transform_chunk(keys, base: str, max_length: int, chain_depth: int):
    """Process a chunk of transformations for parallel processing."""
    variants = set()
    for key in keys:
        for var in TRANSFORMATIONS[key](base):
            if len(var) <= max_length:
                variants.add(var)
                if chain_depth > 1:
                    variants.update(_generate_chain_variants(var, max_length, chain_depth - 1))
    return variants

def _prepare_transform_chunks(transform_keys: list, num_workers: int):
    """Prepare transformation key chunks for parallel processing."""
    chunk_size = max(1, len(transform_keys) // num_workers)
    return [transform_keys[i:i + chunk_size] for i in range(0, len(transform_keys), chunk_size)]

def _generate_parallel_variants(base: str, max_length: int, chain_depth: int, num_workers: Optional[int]) -> List[str]:
    """Generate variants using parallel processing."""
    if num_workers is None:
        cpu_count = os.cpu_count()
        num_workers = max(1, (cpu_count or 4) - 1)
    
    transform_keys = list(TRANSFORMATIONS.keys())
    chunks = _prepare_transform_chunks(transform_keys, num_workers)
    
    all_variants = set([base])
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        process_func = partial(_process_transform_chunk, base=base, max_length=max_length, chain_depth=chain_depth)
        for result in executor.map(process_func, chunks):
            all_variants.update(result)
    return list(all_variants)

def _generate_sequential_variants(base: str, max_length: int, chain_depth: int) -> List[str]:
    """Generate variants using sequential BFS processing."""
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

def generate_variants(base: str, max_length: int, chain_depth: int, parallel: bool = False, num_workers: Optional[int] = None) -> List[str]:
    """
    Generate password variants iteratively or in parallel up to a specified chain depth.

    Args:
        base (str): Base password.
        max_length (int): Maximum length of variants.
        chain_depth (int): Maximum transformation depth.
        parallel (bool): If True, use parallel processing; otherwise, use single-threaded BFS.
        num_workers (int): Number of worker processes for parallel mode (defaults to CPU count - 1).

    Returns:
        List[str]: List of unique variants.
    """
    if parallel:
        return _generate_parallel_variants(base, max_length, chain_depth, num_workers)
    else:
        return _generate_sequential_variants(base, max_length, chain_depth)

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

# Helper function to maintain compatibility with existing code
def _generate_chain_variants(base, max_length, depth):
    """
    Generate chain variants recursively up to specified depth.
    
    Args:
        base: Base password string
        max_length: Maximum length of variants
        depth: Current recursion depth
    
    Returns:
        Set of variant strings
    """
    variants = set()
    if depth <= 0 or not base or len(base) > max_length:
        return variants
        
    for transform_key, transform_fn in TRANSFORMATIONS.items():
        new_variants = transform_fn(base)
        for variant in new_variants:
            if variant and len(variant) <= max_length:
                variants.add(variant)
                if depth > 1:
                    variants.update(_generate_chain_variants(variant, max_length, depth-1))
    return variants

########################################
# Machine Learning Integration
########################################

def load_and_split_data(dataset_path: str = os.path.join(PROJECT_ROOT, "datasets", "default.csv")) -> tuple[list, list]:
    dataset_path = os.path.normpath(dataset_path)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        bases = [line.strip() for line in f if line.strip()]
    
    if not bases:
        raise ValueError(f"No valid passwords found in dataset: {dataset_path}")    
    data = []
    for base in bases:# Generate variants without config parameter
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
    
    x_val = torch.cat(features, dim=0)
    y_val = torch.stack(targets, dim=0)
    
    model.eval()  # Ensure evaluation mode
    with torch.no_grad():
        outputs = model(x_val)
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
        if best_parameters and len(best_parameters) > 0 and best_parameters[0] is not None:
            print(f"\nOptimization complete. Best parameters found: {json.dumps(best_parameters[0], indent=2)}")
            return best_parameters[0]
        else:
            print("\nOptimization complete but no valid parameters found. Using default values.")
            return {"hidden_dim": 128, "dropout": 0.3}
        
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

def password_similarity(pw1: str, pw2: str) -> float:
    """Calculate password similarity using multiple metrics."""
    import Levenshtein
    
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
    from collections import defaultdict
    
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

def _extract_year_patterns(passwords: List[str]) -> dict:
    """Extract year patterns from passwords and group by base."""
    from collections import defaultdict
    
    year_pattern = re.compile(r'^(.+?)(\d{4})$')
    password_map = defaultdict(list)
    
    for password in passwords:
        match = year_pattern.match(password)
        if match:
            base, year = match.groups()
            if len(year) == 4 and 1900 <= int(year) <= 2030:  # Validate as plausible year
                password_map[base].append(year)
    
    return password_map

def _create_training_pairs(password_map: dict) -> List[Tuple[str, str]]:
    """Create training pairs from password variants with multiple years."""
    training_pairs = []
    
    for base, years in password_map.items():
        if len(years) >= 2:
            # Create pairs from each year variant
            base_passwords = [f"{base}{year}" for year in years]
            for i, pw1 in enumerate(base_passwords):
                for pw2 in base_passwords[i+1:]:
                    training_pairs.append((pw1, pw2))
    
    return training_pairs

def mine_year_patterns(passwords: List[str]) -> List[Tuple[str, str]]:
    """Extract high-confidence training pairs from year patterns."""
    password_map = _extract_year_patterns(passwords)
    return _create_training_pairs(password_map)

def mine_incremental_patterns(passwords: List[str]) -> List[Tuple[str, str]]:
    """Find passwords with incremental number patterns."""
    from collections import defaultdict
    
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