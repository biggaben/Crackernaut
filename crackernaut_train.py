#!/usr/bin/env python3
import asyncio
import argparse
import json
import os
import random
import time
import logging
import torch
import torch.nn.functional as F
try:
    from src.models.transformer.transformer_model import PasswordTransformer
except Exception as e:
    print(f"Import failed: {e}")
    raise
from typing import List, Tuple, Optional
from tqdm import tqdm
from src.utils.config_utils import load_configuration as load_config_from_utils
from src.utils.config_utils import save_configuration as save_config_from_utils
from src.utils.variant_utils import generate_variants, SYMBOLS
from src.cuda_ml import (
    load_ml_model as real_load_model,
    save_ml_model as real_save_model,
    extract_features,
    extract_sequence_features,
    extract_sequence_batch,
    predict_config_adjustment,
    MLP,
    PasswordRNN,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("crackernaut")

CONFIG_FILE = "config.json"
MODEL_FILE = "ml_model.pth"
DEFAULT_CONFIG = {
    "modification_weights": {
        "Numeric": 1.0,
        "Symbol": 1.0,
        "Capitalization": 1.0,
        "Leet": 1.0,
        "Shift": 1.0,
        "Repetition": 1.0
    },
    "chain_depth": 2,
    "threshold": 0.5,
    "max_length": 20,
    "model_type": "transformer",
    "model_embed_dim": 64,
    "model_num_heads": 4,
    "model_num_layers": 3,
    "model_hidden_dim": 128,
    "model_dropout": 0.2,
    "lp_chunk_size": 1000000,
    "lp_output_dir": "clusters"
}

def load_default_config() -> dict:
    """Return a fresh copy of the default config."""
    return DEFAULT_CONFIG.copy()

def load_real_world_data(path: str) -> List[List[str]]:
    """Load tab-separated base-variant pairs from a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip().split('\t') for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []

def augment_data(data: List[List[str]]) -> List[tuple]:
    """Augment data with noisy variants."""
    augmented = []
    for item in data:
        if len(item) >= 2:
            base, variant = item[0], item[1]
            noisy = variant + random.choice(SYMBOLS)  # Simple noise example
            augmented.append((base, noisy))
    # Convert original data to tuples and combine
    original_tuples = [(item[0], item[1]) for item in data if len(item) >= 2]
    return original_tuples + augmented

########################################
# Model / ML Integration
########################################

load_ml_model = real_load_model
save_ml_model = real_save_model

def predict_adjustments(variant: str, base: str, model) -> dict:
    """
    Get adjustment predictions for a variant based on the model type.
    
    Args:
        variant (str): Password variant
        base (str): Original base password
        model (nn.Module): PyTorch model
        
    Returns:
        dict: Dictionary of modification adjustments
    """
    device = next(model.parameters()).device
    
    if isinstance(model, PasswordRNN):
        # Use sequence features for RNN
        var_tensor, _ = extract_sequence_features(variant, base, device=device)
        with torch.no_grad():
            prediction = model(var_tensor)[0]
    else:
        # Use original features for MLP
        features = extract_features(variant, base, device=device)
        prediction = predict_config_adjustment(features, model)[0]
    
    rating_dict = {
        "Numeric": prediction[0].item(),
        "Symbol": prediction[1].item(),
        "Capitalization": prediction[2].item(),
        "Leet": prediction[3].item(),
        "Shift": prediction[4].item(),
        "Repetition": prediction[5].item()
    }
    return rating_dict

########################################
# Bulk Training from Wordlist
########################################

########################################
# Helper Functions for Bulk Training
########################################

def _load_wordlist(wordlist_path: str) -> Optional[List[str]]:
    """Load and validate wordlist file."""
    if not os.path.exists(wordlist_path):
        logger.error(f"Wordlist file not found: {wordlist_path}")
        return None

    try:
        with open(wordlist_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        logger.info(f"Loaded {len(lines)} passwords from wordlist")
        return lines
    except Exception as e:
        logger.error(f"Error reading wordlist: {e}")
        return None


def _setup_training_components(model, config: dict) -> Tuple[torch.device, torch.optim.Optimizer, int]:
    """Setup training device, optimizer, and batch size."""
    device = next(model.parameters()).device
    learning_rate = config.get("learning_rate", 0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    batch_size = 32 if isinstance(model, (PasswordRNN, PasswordTransformer)) else 64
    return device, optimizer, batch_size


def _generate_training_variants(batch_passwords: List[str], config: dict) -> Tuple[List[str], List[str]]:
    """Generate password variants for training batch."""
    all_variants = []
    all_bases = []
    
    for base_pw in batch_passwords:
        variants = generate_variants(base_pw, config["max_length"], config["chain_depth"])
        if variants:
            # Sample a subset of variants if there are too many
            if len(variants) > 5:
                variants = random.sample(variants, 5)
            all_variants.extend(variants)
            all_bases.extend([base_pw] * len(variants))
    
    return all_variants, all_bases


def _train_rnn_batch(model, all_variants: List[str], all_bases: List[str], device: torch.device) -> torch.Tensor:
    """Train RNN model on a batch of variants."""
    var_batch, _ = extract_sequence_batch(all_variants, all_bases, device=str(device))
    predictions = model(var_batch)
    
    # Create target tensor - simplified target based on similarity
    targets = torch.zeros_like(predictions)
    for i, (variant, base) in enumerate(zip(all_variants, all_bases)):
        similarity = len(set(variant).intersection(set(base))) / max(len(variant), len(base))
        targets[i] = torch.tensor([similarity] * 6)
    
    return F.mse_loss(predictions, targets)


def _train_mlp_batch(model, all_variants: List[str], all_bases: List[str], 
                     config: dict, device: torch.device) -> torch.Tensor:
    """Train MLP model on a batch of variants."""
    batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    for variant, base in zip(all_variants, all_bases):
        features = extract_features(variant, base, device=str(device))
        prediction = predict_config_adjustment(features, model)
        target = torch.tensor([config["modification_weights"][mod] 
                               for mod in ["Numeric", "Symbol", "Capitalization", 
                                          "Leet", "Shift", "Repetition"]], 
                             device=device)
        batch_loss = batch_loss + F.mse_loss(prediction, target)
    
    return batch_loss / max(len(all_variants), 1)


def _process_training_batch(model, batch_passwords: List[str], config: dict, 
                           optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Process a single training batch."""
    all_variants, all_bases = _generate_training_variants(batch_passwords, config)
    
    if not all_variants:
        return
        
    optimizer.zero_grad()
    
    if isinstance(model, PasswordRNN):
        loss = _train_rnn_batch(model, all_variants, all_bases, device)
    else:
        loss = _train_mlp_batch(model, all_variants, all_bases, config, device)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    # Update config weights based on the latest model
    for variant, base in zip(all_variants, all_bases):
        rating_dict = predict_adjustments(variant, base, model)
        update_config_with_rating(config, rating_dict)


def _train_single_epoch(model, lines: List[str], config: dict, optimizer: torch.optim.Optimizer, 
                       batch_size: int, device: torch.device, epoch: int, iterations: int) -> None:
    """Train model for a single epoch."""
    logger.info(f"Starting training iteration {epoch + 1}/{iterations}")
    random.shuffle(lines)  # Shuffle for better generalization
    model.train()  # Set model to training mode
    
    # Process in batches for efficiency
    total_batches = (len(lines) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(0, len(lines), batch_size), 
                         desc=f"Epoch {epoch+1}/{iterations}", total=total_batches):
        batch_passwords = [line.strip() for line in lines[batch_idx:batch_idx+batch_size] if line.strip()]
        if not batch_passwords:
            continue
            
        _process_training_batch(model, batch_passwords, config, optimizer, device)
    
    logger.info(f"Finished epoch {epoch + 1} with {len(lines)} passwords")


########################################
# Bulk Training from Wordlist
########################################

def bulk_train_on_wordlist(wordlist_path: str, config: dict, model, iterations: int) -> None:
    """
    Process wordlist multiple times for deeper learning.
    
    Args:
        wordlist_path (str): Path to wordlist file
        config (dict): Configuration dictionary
        model (nn.Module): PyTorch model
        iterations (int): Number of training iterations
    """
    # Load and validate wordlist
    lines = _load_wordlist(wordlist_path)
    if lines is None:
        return

    # Setup training components
    device, optimizer, batch_size = _setup_training_components(model, config)
    
    # Training loop
    for epoch in range(iterations):
        _train_single_epoch(model, lines, config, optimizer, batch_size, device, epoch, iterations)
    
    model.eval()
# Interactive Training
########################################

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def parse_csv_integers(user_input: str) -> List[int]:
    """
    Parse comma-separated integers, ignoring invalid entries.
    """
    parts = [p.strip() for p in user_input.split(',')]
    valid_indices = []
    for p in parts:
        if p.isdigit():
            valid_indices.append(int(p))
        else:
            logger.warning(f"Invalid input: '{p}' is not a number. Ignoring.")
    return valid_indices

def handle_empty_variants(config: dict, model) -> Tuple[bool, Optional[str]]:
    """
    Handle the case when no variants are generated.
    """
    logger.warning("No variants generated! Possibly max_length too small.")
    choice = input("Enter [k] to change base password, [c] for config, [reset], [save], or [exit]: ").strip().lower()
    
    if choice == 'k':
        new_base = input("Enter new base password: ").strip()
        if new_base:
            config["current_base"] = new_base
            logger.info(f"Base password updated to '{new_base}'.")
            return False, new_base
        else:
            logger.warning("Invalid input, not updating base.")
            time.sleep(1)
            return False, None
    elif choice == 'c':
        print("\nCurrent configuration:")
        print(json.dumps(config, indent=4))
        input("Press Enter to continue...")
        return False, None
    elif choice == 'reset':
        config.update(load_default_config())
        logger.info("Configuration reset to defaults.")
        time.sleep(1)
        return False, config["current_base"]
    elif choice == 'save':
        save_config_from_utils(config)
        save_ml_model(model)
        logger.info("Config & model saved. Exiting training.")
        return True, None
    elif choice == 'exit':
        logger.info("Exiting without saving.")
        return True, None
    else:
        logger.warning("Invalid option. Try again.")
        time.sleep(1)
        return False, None

def handle_user_selection(indices: List[int], shown_variants: List[str], 
                         base_password: str, model, config: dict) -> None:
    """
    Process user-selected variants for training.
    """
    selected_variants = []
    for i in indices:
        if 1 <= i <= len(shown_variants):
            chosen_var = shown_variants[i-1]
            selected_variants.append(chosen_var)
            logger.info(f"Selected variant: {chosen_var}")
    if selected_variants:
        logger.info(f"Training on {len(selected_variants)} selected variants")
        train_on_selected_variants(selected_variants, base_password, model, config)
        for variant in selected_variants:
            rating_dict = predict_adjustments(variant, base_password, model)
            logger.info(f"New ratings for '{variant}': {rating_dict}")

def train_on_selected_variants(variants: List[str], base_password: str, model, config: dict) -> None:
    """Train the model on user-selected password variants."""
    if not variants:
        return

    model.train()
    device = next(model.parameters()).device
    learning_rate = config.get("learning_rate", 0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Simple training loop for selected variants
    for variant in variants:
        try:
            if isinstance(model, PasswordRNN) or isinstance(model, PasswordTransformer):
                # For sequence models, the target might be more complex or a general "high score"
                var_tensor, _ = extract_sequence_features(variant, base_password, device=device)
                prediction = model(var_tensor.unsqueeze(0)) # Add batch dimension
                # Target for selected variants: aim for high activation or specific pattern
                # This is a simplified target; a more sophisticated one might be needed
                target = torch.ones_like(prediction, device=device)
            elif isinstance(model, MLP):
                features = extract_features(variant, base_password, device=device)
                prediction = model(features.unsqueeze(0)) # Add batch dimension if model expects it
                # Target for selected "good" variants: e.g., all modification types are positively contributing
                target = torch.ones_like(prediction, device=device)
            else:
                logger.warning(f"Unsupported model type {type(model)} for interactive training step.")
                continue

            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        except Exception as e:
            logger.error(f"Error training on variant '{variant}': {e}")
            continue

def handle_config_commands(choice: str, config: dict, model) -> Tuple[bool, Optional[str]]:
    """
    Handle configuration and control commands.
    """
    if choice == 'k':
        new_base = input("Enter new base password: ").strip()
        if new_base:
            config["current_base"] = new_base
            logger.info(f"Base password updated to '{new_base}'.")
            return False, new_base
        else:
            logger.warning("Invalid input, not updating base.")
            time.sleep(1)
            return False, None
    elif choice == 'c':
        print("\nCurrent configuration:")
        print(json.dumps(config, indent=4))
        input("Press Enter to continue...")
        return False, None
    elif choice == 'reset':
        config.update(load_default_config())
        logger.info("Configuration reset to defaults.")
        time.sleep(1)
        return False, config["current_base"]
    elif choice == 'save':
        save_config_from_utils(config)
        save_ml_model(model)
        logger.info("Config & model saved. Exiting training.")
        return True, None
    elif choice == 'exit':
        logger.info("Exiting without saving.")
        return True, None
    else:
        logger.warning("Invalid option. Try again.")
        time.sleep(1)
        return False, None

########################################
# Helper Functions for Interactive Training
########################################

def _display_variants_and_options(shown_variants: List[str], total_variants: int, num_to_show: int) -> None:
    """Display password variants and available options to the user."""
    print(f"Showing {num_to_show} out of {total_variants} variants:")
    for idx, var in enumerate(shown_variants, start=1):
        print(f"  [{idx}] {var}")
    
    print("\nOptions:")
    print("  Enter comma-separated indices (e.g. '1,3') to accept variants")
    print("  [r] reject all variants / show another sample")
    print("  [k] change base password")
    print("  [c] show configuration")
    print("  [reset] reset config to defaults")
    print("  [save] save config & model, then exit")
    print("  [exit] exit without saving")


def _process_user_input(choice: str, shown_variants: List[str], base_password: str, 
                       model, config: dict) -> Tuple[bool, bool, Optional[str]]:
    """
    Process user input and return control flags.
    
    Returns:
        Tuple[bool, bool, Optional[str]]: (should_exit, should_regenerate, new_base_password)
    """
    if choice and all(ch.isdigit() or ch == ',' for ch in choice):
        indices = parse_csv_integers(choice)
        handle_user_selection(indices, shown_variants, base_password, model, config)
        return False, False, None
    
    if choice == 'r':
        logger.info("Rejected sample variants. Showing another sample.")
        time.sleep(1)
        return False, False, None
    
    # Handle configuration commands
    should_exit, new_base = handle_config_commands(choice, config, model)
    should_regenerate = new_base is not None
    return should_exit, should_regenerate, new_base


def _handle_training_iteration(base_password: str, all_variants: List[str], config: dict, 
                              model, num_alternatives: int) -> Tuple[bool, bool, str]:
    """
    Handle a single iteration of the interactive training loop.
    
    Returns:
        Tuple[bool, bool, str]: (should_exit, should_continue_loop, updated_base_password)
    """
    clear_screen()
    logger.info(f"Base password: {base_password}")
    logger.info(f"Total variants available: {len(all_variants)}")
    
    # Handle empty variants case
    if not all_variants:
        should_exit, new_base = handle_empty_variants(config, model)
        if should_exit:
            return True, False, base_password
        if new_base:
            return False, True, new_base
        return False, True, base_password
    
    # Show variants and get user input
    num_to_show = min(num_alternatives, len(all_variants))
    shown_variants = random.sample(all_variants, num_to_show)
    
    _display_variants_and_options(shown_variants, len(all_variants), num_to_show)
    choice = input("\nEnter your choice: ").strip().lower()
    
    # Process user input
    should_exit, should_regenerate, new_base = _process_user_input(
        choice, shown_variants, base_password, model, config
    )
    
    if should_exit:
        return True, False, base_password
    
    if should_regenerate:
        return False, True, new_base or base_password
    
    return False, False, base_password


def interactive_training(config: dict, model, num_alternatives: int = 5) -> dict:
    """
    Provides an interactive training loop:
      - Displays a sample of variants from config['current_base']
      - Accepts user input for variant selection or configuration changes    """
    base_password = config.get("current_base", "Password123")
    all_variants = generate_variants(base_password, config["max_length"], config["chain_depth"])
    
    while True:
        should_exit, should_continue, updated_base = _handle_training_iteration(
            base_password, all_variants, config, model, num_alternatives
        )
        
        if should_exit:
            break
            
        if should_continue:
            base_password = updated_base
            all_variants = generate_variants(base_password, config["max_length"], config["chain_depth"])
    
    return config

########################################
# Updating config with rating
########################################

def update_config_with_rating(config: dict, rating_dict: dict) -> None:
    """Update config weights based on ML predictions."""
    learning_rate = config.get("learning_rate", 0.01)
    for mod, pred in rating_dict.items():
        config["modification_weights"][mod] += pred * learning_rate
        config["modification_weights"][mod] = max(config["modification_weights"][mod], 0.0)

########################################
# Model Evaluation
########################################

def evaluate_model(model, test_data):
    """
    Evaluate model performance on test data.
    """
    model.eval()
    
    total_samples = len(test_data)
    correct_predictions = 0
    mse_loss = 0.0
    
    for base, variant in test_data:
        rating_dict = predict_adjustments(variant, base, model)
        avg_rating = sum(rating_dict.values()) / len(rating_dict)
        
        if avg_rating > 0:
            correct_predictions += 1
        
        target = 1.0
        mse_loss += (avg_rating - target) ** 2
    
    if total_samples > 0:
        accuracy = correct_predictions / total_samples
        mse = mse_loss / total_samples
    else:
        accuracy = 0.0
        mse = 0.0
    
    return {
        "accuracy": accuracy,
        "mse": mse,
        "samples": total_samples
    }

########################################
# Main
########################################

def main():
    parser = argparse.ArgumentParser(description="Crackernaut Training Script")
    parser.add_argument("-b", "--base", type=str, help="Base password for interactive training")
    parser.add_argument("--wordlist", type=str, help="Path to a wordlist for bulk training")
    parser.add_argument("-t", "--times", type=int, default=1, help="Number of iterations for wordlist training")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive training session")
    parser.add_argument("-a", "--alternatives", type=int, default=5, help="Number of variants to show in interactive mode")
    parser.add_argument("--learning-rate", type=float, help="Override the default learning rate")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--model", type=str, choices=["mlp", "rnn", "transformer"], default="transformer",
                        help="Model architecture to use (mlp, rnn, or transformer)")
    parser.add_argument("--supervised", action="store_true", help="Use supervised learning on wordlist")
    parser.add_argument("--prepare", action="store_true", help="Trigger list preparation from a massive dataset")
    parser.add_argument("--lp-dataset", type=str, help="Path to the massive password dataset for list preparation")
    parser.add_argument("--lp-output", type=str, default="clusters", help="Output directory for list preparation clusters")
    parser.add_argument("--lp-chunk-size", type=int, default=1000000, help="Chunk size for list preparation")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    config = load_config_from_utils(CONFIG_FILE)
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate

    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model == "transformer":
        if config["model_embed_dim"] % config["model_num_heads"] != 0:
            logger.warning("model_embed_dim must be divisible by model_num_heads for transformer model. Adjusting model_embed_dim.")
            config["model_embed_dim"] = (config["model_embed_dim"] // config["model_num_heads"]) * config["model_num_heads"]

        model = PasswordTransformer(
            vocab_size=128,
            embed_dim=config.get("model_embed_dim", 64),
            num_heads=config.get("model_num_heads", 4),
            num_layers=config.get("model_num_layers", 3),
            hidden_dim=config.get("model_hidden_dim", 128),
            dropout=config.get("model_dropout", 0.2),
            output_dim=6
        ).to(device)
    elif args.model == "rnn":
        model = PasswordRNN(config["model_embed_dim"], config["model_hidden_dim"], config["model_num_layers"], config["model_dropout"]).to(device)
    elif args.model == "mlp":
        model = MLP(config["model_embed_dim"], config["model_hidden_dim"], config["model_num_layers"], config["model_dropout"]).to(device)

    if args.prepare:
        if not args.lp_dataset:
            logger.error("Please specify the dataset path with --lp-dataset")
            exit(1)
        from src.list_preparer import prepare_list
        asyncio.run(prepare_list(args.lp_dataset, output_dir=args.lp_output))
        exit(0)

    if args.wordlist:
        bulk_train_on_wordlist(args.wordlist, config, model, args.times)
    elif args.interactive:
        if args.base:
            config["current_base"] = args.base
        interactive_training(config, model, args.alternatives)
    else:
        logger.error("No training mode specified. Use --wordlist or --interactive.")
        exit(1)

    save_config_from_utils(config)
    save_ml_model(model, model_type=args.model)

if __name__ == "__main__":
    main()