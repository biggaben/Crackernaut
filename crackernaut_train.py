#!/usr/bin/env python3
"""
crackernaut_train.py

This script is dedicated to training the Crackernaut configuration model.
It interactively presents sample password variants (generated from a given base password)
and adjusts modification weights based on user feedback via a CUDA-accelerated ML model.

Usage:
    python crackernaut_train.py [-b BASE]

Options:
    -b, --base    (For training) Use this as the base password. If not provided, a random
                  base is chosen from a predefined training word list.

Example:
    python crackernaut_train.py --base football2000!
"""

import argparse
import json
import os
import random
import re
import sys
import time

from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init(autoreset=True)

# Import ML module functions from cuda_ml.py
from cuda_ml import load_ml_model, save_ml_model, predict_config_adjustment

# Default configuration parameters
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
    "max_length": 20  # maximum length for variants
}

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_configuration():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            print(Fore.GREEN + "Configuration loaded from", CONFIG_FILE)
            return config
        except Exception as e:
            print(Fore.RED + "Error loading configuration:", e)
    print(Fore.YELLOW + "Using default configuration.")
    return DEFAULT_CONFIG.copy()

def save_configuration(config):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        print(Fore.GREEN + "Configuration saved to", CONFIG_FILE)
    except Exception as e:
        print(Fore.RED + "Error saving configuration:", e)

def load_training_word_list():
    """Return a predefined list of training base passwords."""
    return ["Summer2020", "football2000!", "Passw0rd!", "HelloWorld123", "MySecret!"]

def determine_training_base(cli_base):
    if cli_base:
        return cli_base
    else:
        training_list = load_training_word_list()
        chosen = random.choice(training_list)
        print(Fore.CYAN + "No training base provided; randomly selected:", chosen)
        return chosen

# ---------------------- Variant Generation (Simplified) ---------------------- #

def leetspeak(password):
    mapping = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$"}
    return "".join(mapping.get(c.lower(), c) for c in password)

def generate_variants(base, max_length, chain_depth):
    variants = set()
    # Base variant
    variants.add(base)
    
    # Type A: Numeric Modification
    m = re.search(r"^(.*?)(\d+)$", base)
    if m:
        prefix, num_str = m.group(1), m.group(2)
        num = int(num_str)
        for inc in [1, 2, 3]:
            variants.add(prefix + str(num + inc))
    else:
        variants.add(base + "2021")
    
    # Type B: Symbol Addition
    symbols = ["!", "@", "#"]
    for sym in symbols:
        variants.add(base + sym)
        variants.add(sym + base)
    
    # Type C: Capitalization Tweaks
    variants.add(base.capitalize())
    variants.add(base.upper())
    if len(base) >= 3:
        variant = base[:2] + base[2].upper() + base[3:]
        variants.add(variant)
    
    # Type D: Leet Speak (one substitution)
    for i, ch in enumerate(base):
        if ch.lower() in {"o", "e", "i", "s"}:
            mapping = {"o": "0", "e": "3", "i": "1", "s": "$"}
            variant = base[:i] + mapping[ch.lower()] + base[i+1:]
            variants.add(variant)
            break
    
    # Type E: Shifting Components / Middle Insertion
    if m and len(m.group(1)) >= 2:
        prefix, num_str = m.group(1), m.group(2)
        variants.add(base[0] + num_str + base[1:len(prefix)])
    if len(base) >= 4:
        mid = len(base) // 2
        for sym in symbols:
            variants.add(base[:mid] + sym + base[mid:])
    
    # Type F: Repetition / Padding
    if base:
        variants.add(base + base[-1])
        for sym in symbols:
            variants.add(sym*2 + base)
            variants.add(base + sym*2)
    
    if max_length is not None:
        variants = {v for v in variants if len(v) <= max_length}
    return list(variants)

def select_random_sample(variants, count=3):
    if len(variants) <= count:
        return list(variants)
    return random.sample(variants, count)

# ---------------------- Modification Identification ---------------------- #

def identify_modifications(variant, base):
    mods = []
    if variant != base:
        if re.search(r"\d", variant) and not re.search(r"\d", base):
            mods.append("Numeric")
        elif re.search(r"\d+$", variant) and re.search(r"\d+$", base):
            if re.search(r"\d+$", variant).group() != re.search(r"\d+$", base).group():
                mods.append("Numeric")
        if variant[0] in "!@#$%?&" or variant[-1] in "!@#$%?&":
            mods.append("Symbol")
        if variant != base and variant.lower() == base.lower():
            mods.append("Capitalization")
        if any(ch in variant for ch in ["0", "3", "1", "$"]) and leetspeak(base) != variant:
            mods.append("Leet")
        if base in variant and not variant.startswith(base) and not variant.endswith(base):
            mods.append("Shift")
        if len(variant) > len(base) and variant[-1] == variant[-2]:
            mods.append("Repetition")
    return mods

# ---------------------- ML Integration for Training ---------------------- #
# ML functions are imported from cuda_ml.py

# ---------------------- Interactive Training ---------------------- #

def interactive_training(config, training_base, ml_model):
    while True:
        clear_screen()
        print(Fore.CYAN + Style.BRIGHT + "Base password:")
        print(Fore.CYAN + f"   {training_base}\n")
        
        variants = generate_variants(training_base, config["max_length"], config["chain_depth"])
        sample = select_random_sample(variants, 3)
        print(Fore.YELLOW + "Sample Variants:")
        for idx, variant in enumerate(sample, start=1):
            print(Fore.YELLOW + f"  [{idx}] {variant}")
        print(Style.RESET_ALL)
        print("Options:")
        print("  [1-3] Accept variant(s) (enter comma-separated numbers, e.g., 1,3)")
        print("  [r]   Reject all options / Request new sample")
        print("  [k]   Enter new base password (new keyword)")
        print("  [c]   Show current configuration")
        print("  [reset] Reset configuration to defaults")
        print("  [save] Save configuration and exit training")
        print("  [exit] Exit without saving")
        choice = input("Enter your choice: ").strip().lower()
        
        if choice and all(c.isdigit() or c == ',' for c in choice):
            indices = [int(x.strip()) for x in choice.split(",") if x.strip().isdigit()]
            accepted_variants = [sample[i-1] for i in indices if 1 <= i <= len(sample)]
            for selected in accepted_variants:
                mods = identify_modifications(selected, training_base)
                if mods:
                    print(Fore.GREEN + f"Accepted variant: {selected}")
                    print(Fore.GREEN + f"Identified modifications: {mods}")
                    for mod in mods:
                        features = extract_features(selected, training_base)
                        predicted_adjustment = predict_config_adjustment(features, ml_model)
                        config["modification_weights"][mod] = config["modification_weights"].get(mod, 1.0) + predicted_adjustment
                    print(Fore.GREEN + f"Updated weights: {config['modification_weights']}")
                    time.sleep(2)
                else:
                    print(Fore.RED + "No modifications detected; no adjustments made.")
                    time.sleep(2)
        elif choice == "r":
            mods_in_sample = set()
            for variant in sample:
                mods_in_sample.update(identify_modifications(variant, training_base))
            if mods_in_sample:
                for mod in mods_in_sample:
                    config["modification_weights"][mod] = max(0.1, config["modification_weights"].get(mod, 1.0) - 0.05)
                print(Fore.RED + f"Rejected sample. Updated weights: {config['modification_weights']}")
                time.sleep(2)
            else:
                print("No modifications detected in sample to adjust.")
                time.sleep(2)
        elif choice == "k":
            new_base = input("Enter new base password: ").strip()
            if new_base and new_base:
                training_base = new_base
                print(Fore.CYAN + f"Training base updated to: {training_base}")
                time.sleep(2)
            else:
                print(Fore.RED + "Invalid base password entered. Keeping the current base.")
                time.sleep(2)
        elif choice == "c":
            print("Current configuration:")
            print(json.dumps(config, indent=4))
            input("Press Enter to continue...")
        elif choice == "reset":
            config = DEFAULT_CONFIG.copy()
            print("Configuration reset to defaults.")
            time.sleep(2)
        elif choice == "save":
            save_configuration(config)
            save_ml_model(ml_model)
            print("Configuration saved. Exiting training.")
            break
        elif choice == "exit":
            print("Exiting training without saving.")
            break
        else:
            print("Invalid option. Please try again.")
            time.sleep(2)
    return config

def main_training_mode():
    config = load_configuration()
    ml_model = load_ml_model()  # Load CUDA-accelerated ML model
    parser = argparse.ArgumentParser(description="Crackernaut Training Mode")
    parser.add_argument("-b", "--base", type=str, help="(For training) Use this as the base password")
    args = parser.parse_args()
    training_base = args.base if args.base else random.choice(load_training_word_list())
    print("Using training base:", training_base)
    updated_config = interactive_training(config, training_base, ml_model)
    print("Training complete.")
    sys.exit(0)

def load_training_word_list():
    return ["Summer2020", "football2000!", "Passw0rd!", "HelloWorld123", "MySecret!"]

# ---------------------- Feature Extraction for ML ---------------------- #

def extract_features(variant, base):
    import string
    def numeric_diff(s):
        m = re.search(r"(\d+)$", s)
        return len(m.group(1)) if m else 0
    num_base = numeric_diff(base)
    num_variant = numeric_diff(variant)
    f1 = abs(num_variant - num_base)
    
    f2 = sum(1 for ch in variant if ch in string.punctuation)
    
    f3 = sum(1 for b, v in zip(base, variant) if b.islower() != v.islower())
    
    leet_chars = {"0", "3", "1", "$"}
    f4 = sum(1 for ch in variant if ch in leet_chars)
    
    f5 = 0
    m_base = re.search(r"(\d+)$", base)
    m_variant = re.search(r"(\d+)$", variant)
    if m_base and m_variant:
        if m_variant.group() != m_base.group():
            f5 = 1
    elif m_base and not m_variant:
        f5 = 1
    
    f6 = abs(len(variant) - len(base))
    
    try:
        import torch
        feature_tensor = torch.tensor([f1, f2, f3, f4, f5, f6], dtype=torch.float32).unsqueeze(0)
        return feature_tensor
    except ImportError:
        print("PyTorch is required for ML features.")
        sys.exit(1)

if __name__ == "__main__":
    main_training_mode()
