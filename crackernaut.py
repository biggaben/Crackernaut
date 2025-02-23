#!/usr/bin/env python3
"""
crackernaut.py

Main production script for Crackernaut – a utility that generates human-like
password variants from a given base password. The script loads configuration
parameters (from a JSON file or defaults), generates variants via basic and
advanced transformation chains, applies smart filtering based on a heuristic
score, and outputs the top variants to the console or a specified file.

Usage:
    python crackernaut.py -p <base_password> [-l <max_length>] [-n <output_count>] [-o <output_file>]
    
    If -p is not provided, the script prompts for the base password.
    The -l flag sets the maximum length of variants.
    The -n flag limits the number of variants output.
    The -o flag saves the output variants to a file.
"""

import argparse
import json
import os
import random
import re
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
    "max_length": 20  # Default maximum length
}

def load_configuration():
    """Load configuration parameters from a file if available, otherwise use defaults."""
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

# ---------------------- Variant Generation Functions ---------------------- #

def leetspeak(password):
    """Return a leetspeak version of the password."""
    mapping = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$"}
    return "".join(mapping.get(c.lower(), c) for c in password)

def generate_basic_variants(base):
    """Generate basic variants: reverse, swapcase, leetspeak, symbol additions."""
    variants = set()
    variants.add(base)
    variants.add(base[::-1])
    variants.add(base.swapcase())
    variants.add(leetspeak(base))
    
    expansions = ['!', '@', '#', '$', '123', '2023', '01', '001', '12', '1234']
    for exp in expansions:
        variants.add(base + exp)
        variants.add(exp + base)
    
    # Add a random symbol addition from a common set.
    common_punctuations = ["!", "@", "#", "$", "%", "&", "*", "?"]
    variants.add(base + random.choice(common_punctuations))
    variants.add(random.choice(common_punctuations) + base)
    return variants

def chain_basic_increment(base):
    """If the base ends with a number, generate numeric increments; otherwise, append a default."""
    variants = set()
    m = re.search(r"^(.*?)(\d+)$", base)
    if m:
        prefix, num_str = m.group(1), m.group(2)
        num = int(num_str)
        for inc in [1, 2, 3]:
            variants.add(prefix + str(num + inc))
    else:
        variants.add(base + "2021")
    return variants

def chain_symbol_addition(base):
    """Generate variants by adding symbols to the beginning or end."""
    variants = set()
    symbols = ["!", "@", "#", "$", "%", "?", "&"]
    for sym in symbols:
        variants.add(base + sym)
        variants.add(sym + base)
    return variants

def chain_capitalization_tweaks(base):
    """Generate capitalization variants."""
    variants = set()
    variants.add(base.capitalize())
    variants.add(base.upper())
    if len(base) >= 3:
        variant = base[:2] + base[2].upper() + base[3:]
        variants.add(variant)
    return variants

def chain_leet_substitution(base):
    """Generate a leet substitution variant by replacing one character."""
    variants = set()
    mapping = {"o": "0", "e": "3", "i": "1", "s": "$"}
    for i, ch in enumerate(base):
        if ch.lower() in mapping:
            variant = base[:i] + mapping[ch.lower()] + base[i+1:]
            variants.add(variant)
            break  # Only one substitution per variant
    return variants

def chain_shift_variants(base):
    """Generate a variant by shifting the numeric suffix or inserting a symbol in the middle."""
    variants = set()
    m = re.search(r"^(.*?)(\d+)$", base)
    if m and len(m.group(1)) >= 2:
        prefix, num_str = m.group(1), m.group(2)
        variants.add(base[0] + num_str + base[1:len(prefix)])
    if len(base) >= 4:
        mid = len(base) // 2
        symbols = ["!", "@", "#", "$", "%", "?", "&"]
        for sym in symbols:
            variants.add(base[:mid] + sym + base[mid:])
    return variants

def chain_repetition_variants(base):
    """Generate a variant by minimal repetition or padding."""
    variants = set()
    if base:
        variants.add(base + base[-1])
    symbols = ["!", "@", "#", "$", "%", "?", "&"]
    for sym in symbols:
        variants.add(sym*2 + base)
        variants.add(base + sym*2)
    return variants

def chain_middle_insertion(base):
    """Insert a symbol into the middle of the base."""
    variants = set()
    if len(base) >= 4:
        mid = len(base) // 2
        symbols = ["!", "@", "#", "$", "%", "?", "&"]
        for sym in symbols:
            variants.add(base[:mid] + sym + base[mid:])
    return variants

def generate_human_chains(base):
    """Generate variants using advanced human-like chains."""
    chains = set()
    chains.update(chain_basic_increment(base))
    chains.update(chain_symbol_addition(base))
    chains.update(chain_capitalization_tweaks(base))
    chains.update(chain_leet_substitution(base))
    chains.update(chain_shift_variants(base))
    chains.update(chain_repetition_variants(base))
    chains.update(chain_middle_insertion(base))
    return chains

def combine_variants(base, max_length, chain_depth):
    """Combine basic and advanced variants; optionally combine two different modifications if allowed."""
    basic_variants = generate_basic_variants(base)
    advanced_variants = generate_human_chains(base)
    combined = basic_variants.union(advanced_variants)
    # For this first version, we do not further combine variants beyond single modifications.
    if max_length is not None:
        combined = {v for v in combined if len(v) <= max_length}
    return list(combined)

# ---------------------- Scoring and Filtering ---------------------- #

def count_modifications(variant, base):
    """Count a simple number of differences between variant and base."""
    if variant == base:
        return 0
    count = 0
    # Check if numeric modification exists.
    if re.search(r"\d", variant) and not re.search(r"\d", base):
        count += 1
    elif re.search(r"\d+$", variant) and re.search(r"\d+$", base):
        if re.search(r"\d+$", variant).group() != re.search(r"\d+$", base).group():
            count += 1
    # Check for symbol addition.
    if variant[0] in "!@#$%?&" or variant[-1] in "!@#$%?&":
        count += 1
    # Check for capitalization change.
    if variant != base and variant.lower() == base.lower():
        count += 1
    # Check for leet substitution.
    if any(ch in variant for ch in ["0", "3", "1", "$"]) and leetspeak(base) != variant:
        count += 1
    # Check for shift.
    if base in variant and not variant.startswith(base) and not variant.endswith(base):
        count += 1
    # Check for repetition.
    if len(variant) > len(base) and variant[-1] == variant[-2]:
        count += 1
    return count

def heuristic_score(variant, base, config):
    """
    Compute a heuristic score based on which modification types are present.
    For each identified modification type, add the corresponding weight.
    """
    score = 0.0
    mods = []
    # Numeric modification
    if re.search(r"\d", variant) and not re.search(r"\d", base):
        mods.append("Numeric")
    elif re.search(r"\d+$", variant) and re.search(r"\d+$", base):
        if re.search(r"\d+$", variant).group() != re.search(r"\d+$", base).group():
            mods.append("Numeric")
    # Symbol addition
    if variant[0] in "!@#$%?&" or variant[-1] in "!@#$%?&":
        mods.append("Symbol")
    # Capitalization tweak
    if variant != base and variant.lower() == base.lower():
        mods.append("Capitalization")
    # Leet substitution
    if any(ch in variant for ch in ["0", "3", "1", "$"]) and leetspeak(base) != variant:
        mods.append("Leet")
    # Shifting
    if base in variant and not variant.startswith(base) and not variant.endswith(base):
        mods.append("Shift")
    # Repetition
    if len(variant) > len(base) and variant[-1] == variant[-2]:
        mods.append("Repetition")
    
    for mod in set(mods):
        weight = config["modification_weights"].get(mod, 1.0)
        score += weight
    return score

def penalty(mod_count):
    """Apply a penalty based on the number of modifications."""
    return 0.2 * mod_count

def smart_filter_candidates(variants, base, config):
    """Score and filter candidates, returning those that meet a threshold."""
    scored_list = []
    for variant in variants:
        mod_count = count_modifications(variant, base)
        score = heuristic_score(variant, base, config) - penalty(mod_count)
        if score >= config["threshold"]:
            scored_list.append((variant, score))
    scored_list.sort(key=lambda x: x[1], reverse=True)
    return [v for v, s in scored_list]

# ---------------------- Main Function ---------------------- #

def main():
    parser = argparse.ArgumentParser(description="Crackernaut Variant Generator")
    parser.add_argument("-p", "--password", type=str, help="Base password (if not provided, prompt for input)")
    parser.add_argument("-l", "--length", type=int, help="Maximum length of generated variants")
    parser.add_argument("-n", "--number", type=int, help="Limit number of variants to output")
    parser.add_argument("-o", "--output", type=str, help="File to save the variants")
    args = parser.parse_args()
    
    if args.password:
        base = args.password
    else:
        base = input("Enter base password: ").strip()
    
    if not base or len(base) < 3:
        print("Invalid base password.")
        sys.exit(1)
    
    config = load_configuration()
    if args.length:
        config["max_length"] = args.length
    
    # Generate combined variants from basic and advanced human-like chains.
    variants = combine_variants(base, config["max_length"], config["chain_depth"])
    # Apply smart filtering based on the heuristic score.
    smart_variants = smart_filter_candidates(variants, base, config)
    
    if args.number:
        output_variants = smart_variants[:args.number]
    else:
        output_variants = smart_variants
    
    if args.output:
        try:
            with open(args.output, "w") as f:
                for variant in output_variants:
                    f.write(variant + "\n")
            print(f"Variants saved to {args.output}")
        except Exception as e:
            print("Error saving variants:", e)
    else:
        print("Generated Variants:")
        for variant in output_variants:
            print(variant)

if __name__ == "__main__":
    main()
