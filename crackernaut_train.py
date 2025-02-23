#!/usr/bin/env python3
"""
crackernaut_train.py

This script trains the Crackernaut configuration model. It supports:
  1) Bulk training from a wordlist file (if --wordlist is given).
  2) Interactive training (if --interactive is set) to refine the model and
     configuration based on user feedback.

Optional arguments:
  -b, --base        Base password to use in interactive training (if desired).
  --wordlist FILE   Text file with one password per line for bulk training.
  --interactive     Launch the interactive session.

Usage:
  python crackernaut_train.py --wordlist common_1k.txt
  python crackernaut_train.py --interactive -b "Summer2023"
  python crackernaut_train.py --wordlist big_list.txt --interactive

No placeholders are used; all code is workable.
"""

import argparse
import json
import os
import random
import time

from typing import List
from config_utils import load_configuration as load_config_from_utils
from config_utils import save_configuration as save_config_from_utils
from variant_utils import generate_variants, optimize_hyperparameters, SYMBOLS
from cuda_ml import (
    load_ml_model as real_load_model,
    save_ml_model as real_save_model,
    extract_features,
    predict_config_adjustment,
    MLP
)

CONFIG_FILE = "config.json"
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
    "current_base": "Password123!"
}

def load_default_config() -> dict:
    """Return a fresh copy of the default config."""
    return DEFAULT_CONFIG.copy()

def load_real_world_data(path: str) -> List[tuple]:
    """Load tab-separated base-variant pairs from a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip().split('\t') for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def augment_data(data: List[tuple]) -> List[tuple]:
    """Augment data with noisy variants."""
    augmented = []
    for base, variant in data:
        noisy = variant + random.choice(SYMBOLS)  # Simple noise example
        augmented.append((base, noisy))
    return data + augmented

########################################
# Model / ML Integration
########################################

load_ml_model = real_load_model
save_ml_model = real_save_model

def predict_adjustments(variant: str, base: str, model) -> dict:
    device = next(model.parameters()).device
    features = extract_features(variant, base, device=device)
    prediction = predict_config_adjustment(features, model)[0]  # Shape (6,)
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

def bulk_train_on_wordlist(wordlist_path: str, config: dict, model, iterations: int) -> None:
    """Process wordlist multiple times for deeper learning."""
    if not os.path.exists(wordlist_path):
        print(f"Wordlist file not found: {wordlist_path}")
        return

    try:
        with open(wordlist_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading wordlist: {e}")
        return

    for epoch in range(iterations):
        print(f"\n--- Training Iteration {epoch + 1}/{iterations} ---")
        random.shuffle(lines)  # Shuffle for better generalization
        count = 0
        for line in lines:
            pw = line.strip()
            if not pw:
                continue
            variants = generate_variants(pw, config["max_length"], config["chain_depth"])
            for var in variants:
                rating_dict = predict_adjustments(var, pw, model)
                update_config_with_rating(config, rating_dict)
            count += 1
        print(f"Finished iteration {epoch + 1} with {count} samples")

########################################
# Interactive Training
########################################

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def parse_csv_integers(user_input: str) -> List[int]:
    """
    Parse comma-separated integers, ignoring invalid entries.

    Args:
        user_input (str): User input string.

    Returns:
        List[int]: List of valid integer indices.
    """
    parts = [p.strip() for p in user_input.split(',')]
    valid_indices = []
    for p in parts:
        if p.isdigit():
            valid_indices.append(int(p))
        else:
            print(f"Invalid input: '{p}' is not a number. Ignoring.")
    return valid_indices

def interactive_training(config: dict, model, num_alternatives: int = 5) -> dict:
    """
    Provides an interactive loop:
      - Show a sample of variants from config['current_base']
      - Accept multiple, reject, or do config changes
      - No advanced RL, just direct updates
    """
    base_password = config.get("current_base", "Password123")
    all_variants = generate_variants(base_password, config["max_length"], config["chain_depth"])
    
    while True:
        clear_screen()
        print(f"Base password: {base_password}")
        print(f"Total variants available: {len(all_variants)}")
        
        # Handle empty variant list
        if not all_variants:
            print("No variants generated! Possibly set max_length too small.")
            choice = input("Enter [k] to change base password, [c] for config, [reset], [save], or [exit]: ").strip().lower()
            # Handle accordingly (implement logic for options)
            if choice == 'k':
                new_base = input("Enter new base password: ").strip()
                if new_base:
                    config["current_base"] = new_base
                    base_password = new_base
                    all_variants = generate_variants(base_password, config["max_length"], config["chain_depth"])
                    print(f"Base password updated to '{new_base}'.")
                else:
                    print("Invalid input, not updating base.")
                time.sleep(1)
            elif choice == 'c':
                print("\nCurrent configuration:")
                print(json.dumps(config, indent=4))
                input("Press Enter to continue...")
            elif choice == 'reset':
                config = load_default_config()
                print("Configuration reset to defaults.")
                time.sleep(1)
            elif choice == 'save':
                save_config_from_utils(config)
                save_ml_model(model)
                print("Config & model saved. Exiting training.")
                break
            elif choice == 'exit':
                print("Exiting without saving.")
                break
            else:
                print("Invalid option. Try again.")
                time.sleep(1)
            continue
        
        # Sample variants to show
        num_to_show = min(num_alternatives, len(all_variants))
        shown_variants = random.sample(all_variants, num_to_show)
        
        print(f"Showing {num_to_show} out of {len(all_variants)} variants:")
        for idx, var in enumerate(shown_variants, start=1):
            print(f"  [{idx}] {var}")
        
        print("\nOptions:")
        print("  Enter comma-separated indices (e.g. '1,3') to accept multiple variants")
        print("  [r] reject all variants / show another sample")
        print("  [k] change base password (keyword)")
        print("  [c] show configuration")
        print("  [reset] reset config to defaults")
        print("  [save] save config & model, then exit")
        print("  [exit] exit without saving")
        
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice and all(ch.isdigit() or ch == ',' for ch in choice):
            indices = parse_csv_integers(choice)
            for i in indices:
                if 1 <= i <= len(shown_variants):
                    chosen_var = shown_variants[i-1]
                    rating_dict = predict_adjustments(chosen_var, base_password, model)
                    update_config_with_rating(config, rating_dict)
                    print(f"Accepted variant: {chosen_var}")
                    print(f"Rating: {rating_dict}")
                    time.sleep(1)
        elif choice == 'r':
            print("Rejected all sample variants. Showing another sample.")
            time.sleep(1)
            # Continue to next iteration to show another sample
        elif choice == 'k':
            new_base = input("Enter new base password: ").strip()
            if new_base:
                config["current_base"] = new_base
                base_password = new_base
                all_variants = generate_variants(base_password, config["max_length"], config["chain_depth"])
                print(f"Base password updated to '{new_base}'.")
            else:
                print("Invalid input, not updating base.")
            time.sleep(1)
        elif choice == 'c':
            print("\nCurrent configuration:")
            print(json.dumps(config, indent=4))
            input("Press Enter to continue...")
        elif choice == 'reset':
            config = load_default_config()
            print("Configuration reset to defaults.")
            time.sleep(1)
        elif choice == 'save':
            save_config_from_utils(config)
            save_ml_model(model)
            print("Config & model saved. Exiting training.")
            break
        elif choice == 'exit':
            print("Exiting without saving.")
            break
        else:
            print("Invalid option. Try again.")
            time.sleep(1)
    return config

########################################
# Updating config with rating
########################################

def update_config_with_rating(config: dict, rating_dict: dict) -> None:
    """Update config weights based on ML predictions."""
    learning_rate = 0.01
    for mod, pred in rating_dict.items():
        config["modification_weights"][mod] += pred * learning_rate
        config["modification_weights"][mod] = max(config["modification_weights"][mod], 0.0)

########################################
# Main
########################################

def main():
    parser = argparse.ArgumentParser(description="Crackernaut Training Script")
    parser.add_argument("-b", "--base", type=str, help="Base password for interactive training")
    parser.add_argument("--wordlist", type=str, help="Path to a wordlist for bulk training")
    parser.add_argument("-t", "--times", type=int, default=1, help="Number of iterations for wordlist training")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive training session")
    parser.add_argument("-a", "--alternatives", type=int, default=5, help="Number of variants to show each round in interactive mode")
    args = parser.parse_args()

    config = load_config_from_utils()
    model = load_ml_model()

    if args.wordlist:
        wordlist_path = os.path.normpath(args.wordlist)
        if not os.path.exists(wordlist_path):
            print(f"Error: Wordlist not found at {wordlist_path}")
        else:
            try:
                print(f"\nStarting bulk training ({args.times} iteration{'s' if args.times > 1 else ''})")
                bulk_train_on_wordlist(wordlist_path, config, model, args.times)
                
                print("\n--- Starting hyperparameter optimization ---")
                best_params = optimize_hyperparameters(wordlist_path)
                print(f"\nOptimization complete. Best parameters: {json.dumps(best_params, indent=2)}")

                optimized_model = MLP(
                    hidden_dim=best_params["hidden_dim"],
                    dropout=best_params["dropout"]
                )
                save_ml_model(optimized_model)
                config["last_optimization"] = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "hidden_dim": best_params["hidden_dim"],
                    "dropout": best_params["dropout"]
                }
                save_config_from_utils(config)
            except Exception as e:
                print(f"\nCritical error during bulk processing: {str(e)}")
                save_ml_model(model)
                save_config_from_utils(config)

    if args.interactive:
        print("\n--- Starting interactive session ---")
        try:
            current_model = load_ml_model()
            if args.base:
                config["current_base"] = args.base
                print(f"Using provided base password: {args.base}")
            interactive_training(config, current_model, num_alternatives=args.alternatives)
        except Exception as e:
            print(f"Failed to start interactive session: {str(e)}")

    try:
        save_ml_model(model)
        save_config_from_utils(config)
        print("\nSession ended successfully. Model and config states preserved.")
    except Exception as e:
        print(f"\nWarning: Failed to save final state - {str(e)}")

if __name__ == "__main__":
    main()