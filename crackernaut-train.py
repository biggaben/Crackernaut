#!/usr/bin/env python3
import argparse
import json
import os
import random
import re

# For interactive input and simple variant generation
import sys

# ---------------------- Configuration Handling ---------------------- #

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
    "max_length": 20  # default maximum length
}

def load_configuration():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            print("Configuration loaded from", CONFIG_FILE)
            return config
        except Exception as e:
            print("Error loading configuration:", e)
    print("Using default configuration.")
    return DEFAULT_CONFIG.copy()

def save_configuration(config):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        print("Configuration saved to", CONFIG_FILE)
    except Exception as e:
        print("Error saving configuration:", e)

def load_training_word_list():
    # Hard-coded training word list (could also be loaded from file)
    return ["Summer2020", "football2000!", "Passw0rd!", "HelloWorld123", "MySecret!"]

def determine_training_base(cli_base):
    # In training mode, if user supplies -n, use that as base.
    if cli_base:
        return cli_base
    else:
        training_list = load_training_word_list()
        chosen = random.choice(training_list)
        print("No training base provided; randomly selected:", chosen)
        return chosen

# ---------------------- Variant Generation (Simplified) ---------------------- #

def generate_variants(base, max_length, chain_depth):
    """
    For training purposes, generate a set of variants using simple modifications.
    In a full implementation this function would incorporate various chains.
    """
    variants = set()
    # Base variant:
    variants.add(base)
    # Type A: Numeric increment (if base ends with numbers)
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
    # Type D: Leet Speak Substitution (only one substitution)
    for i, ch in enumerate(base):
        if ch.lower() in {"o", "e", "i", "s"}:
            # Replace first occurrence only
            mapping = {"o": "0", "e": "3", "i": "1", "s": "$"}
            variant = base[:i] + mapping[ch.lower()] + base[i+1:]
            variants.add(variant)
            break
    # Type E: Shifting Components (if numeric suffix exists)
    if m and len(m.group(1)) >= 2:
        prefix, num_str = m.group(1), m.group(2)
        variant = base[0] + num_str + base[1:len(prefix)]
        variants.add(variant)
    # Type F: Repetition / Padding
    if base:
        variants.add(base + base[-1])
    # Limit variants by max_length if provided
    if max_length is not None:
        variants = {v for v in variants if len(v) <= max_length}
    return list(variants)

# ---------------------- Simple Modification Identification ---------------------- #

def identify_modifications(variant, base):
    """
    Identify which modifications are present in variant compared to base.
    This is a very simple heuristic implementation.
    """
    mods = []
    if variant != base:
        # Check for numeric change
        if re.search(r"\d", variant) and not re.search(r"\d", base):
            mods.append("Numeric")
        elif re.search(r"\d+", variant) and re.search(r"\d+$", base):
            base_num = re.search(r"\d+$", base).group()
            variant_num = re.search(r"\d+$", variant).group()
            if variant_num != base_num:
                mods.append("Numeric")
        # Check for symbol addition (if variant starts or ends with a non-alphanumeric)
        if variant[0] in string.punctuation or variant[-1] in string.punctuation:
            mods.append("Symbol")
        # Check for capitalization change
        if variant != base and variant.lower() == base.lower():
            mods.append("Capitalization")
        # Check for leet substitution (if variant contains 0,3,1,$)
        if any(x in variant for x in ["0", "3", "1", "$"]) and variant.lower() != base.lower():
            mods.append("Leet")
        # Check for repetition (if variant is longer than base and ends with a repeated char)
        if len(variant) > len(base) and variant[-1] == variant[-2]:
            mods.append("Repetition")
        # Check for shifting (if the numeric part is moved)
        if variant != base and base in variant and not variant.startswith(base) and not variant.endswith(base):
            mods.append("Shift")
    return mods

# ---------------------- Interactive Training ---------------------- #

def select_random_sample(variants, count=3):
    """Select a random sample of 'count' variants from the list."""
    if len(variants) <= count:
        return list(variants)
    return random.sample(variants, count)

def interactive_training(config, training_base):
    print("--------------------------------------------------")
    print("Training mode initiated with base password:", training_base)
    print("Current configuration:", config)
    print("--------------------------------------------------")
    adjustment_value = 0.1  # how much to adjust weight per feedback

    while True:
        variants = generate_variants(training_base, config["max_length"], config["chain_depth"])
        sample = select_random_sample(variants, 3)
        print("Sample Variants:")
        for idx, variant in enumerate(sample, start=1):
            print(f"  [{idx}] {variant}")
        print("Options:")
        print("  [1-3] Accept variant (enter option number)")
        print("  [r]  Reject all options / Request new sample")
        print("  [c]  Show current configuration")
        print("  [reset] Reset configuration to defaults")
        print("  [save] Save configuration and exit training")
        print("  [exit] Exit without saving")
        choice = input("Enter your choice: ").strip().lower()
        
        if choice in {"1", "2", "3"}:
            selected = sample[int(choice)-1]
            mods = identify_modifications(selected, training_base)
            if mods:
                print("Accepted variant:", selected)
                print("Identified modifications:", mods)
                for mod in mods:
                    config["modification_weights"][mod] = config["modification_weights"].get(mod, 1.0) + adjustment_value
                print("Updated weights:", config["modification_weights"])
            else:
                print("No modifications detected; no weight changes applied.")
        elif choice == "r":
            # For rejected sample, decrease weights for all modifications present in the sample.
            mods_in_sample = set()
            for variant in sample:
                mods_in_sample.update(identify_modifications(variant, training_base))
            if mods_in_sample:
                for mod in mods_in_sample:
                    config["modification_weights"][mod] = max(0.1, config["modification_weights"].get(mod, 1.0) - (adjustment_value/2))
                print("Rejected sample. Updated weights:", config["modification_weights"])
            else:
                print("No modifications detected in sample to adjust.")
        elif choice == "c":
            print("Current configuration:")
            print(json.dumps(config, indent=4))
        elif choice == "reset":
            config = DEFAULT_CONFIG.copy()
            print("Configuration reset to defaults.")
        elif choice == "save":
            save_configuration(config)
            print("Exiting training mode. Configuration saved.")
            break
        elif choice == "exit":
            print("Exiting training mode without saving.")
            break
        else:
            print("Invalid option. Please try again.")
        print("--------------------------------------------------")
    return config

def main_training_mode():
    config = load_configuration()
    # In training mode, if user provided -n as the training base, use that;
    # otherwise, choose a random one.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-n", "--number", type=str, help="(In training mode) Use this as the training base password")
    args, unknown = parser.parse_known_args()
    training_base = determine_training_base(args.number)
    updated_config = interactive_training(config, training_base)
    # updated_config is already saved if the user chose to save.
    print("Training complete.")
    sys.exit(0)

if __name__ == "__main__":
    # This script is for training mode only.
    parser = argparse.ArgumentParser(description="Crackernaut Training Mode")
    parser.add_argument("--train", action="store_true", help="Launch training mode")
    parser.add_argument("-n", "--number", type=str, help="(For training) Use this as the base password")
    args = parser.parse_args()
    
    if args.train:
        main_training_mode()
    else:
        print("This script is for training mode only. Use --train to start training.")
