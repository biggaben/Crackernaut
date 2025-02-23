#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys

# ---------------------- Configuration ---------------------- #
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
    "max_length": 20  # Default maximum length of variants
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

# ---------------------- Variant Generation ---------------------- #

def generate_basic_variants(base):
    """Generate basic modifications: reverse, swapcase, leetspeak, and symbol additions."""
    variants = set()
    variants.add(base)
    variants.add(base[::-1])
    variants.add(base.swapcase())
    variants.add(leetspeak(base))
    
    expansions = ['!', '@', '#', '$', '123', '2023', '01', '001', '12', '1234']
    for exp in expansions:
        variants.add(base + exp)
        variants.add(exp + base)
    
    # Add a random symbol addition
    common_punctuations = ["!", "@", "#", "$", "%", "&", "*", "?"]
    variants.add(base + random.choice(common_punctuations))
    variants.add(random.choice(common_punctuations) + base)
    
    return variants

def generate_human_chains(base):
    """Generate variants from advanced human-like chains."""
    chains = set()
    chains.update(chain_basic_increment(base))
    chains.update(chain_symbol_addition(base))
    chains.update(chain_capitalization_tweaks(base))
    chains.update(chain_leet_substitution(base))
    chains.update(chain_shift_variants(base))
    chains.update(chain_repetition_variants(base))
    chains.update(chain_middle_insertion(base))
    return chains

def chain_basic_increment(base):
    """If base ends with a number, increment it by 1-3."""
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
    """Append or prepend a common symbol."""
    variants = set()
    symbols = ["!", "@", "#", "$", "%", "?", "&"]
    for sym in symbols:
        variants.add(base + sym)
        variants.add(sym + base)
    return variants

def chain_capitalization_tweaks(base):
    """Generate variants with slight capitalization changes."""
    variants = set()
    variants.add(base.capitalize())
    variants.add(base.upper())
    if len(base) >= 3:
        variant = base[:2] + base[2].upper() + base[3:]
        variants.add(variant)
    return variants

def chain_leet_substitution(base):
    """Apply a single leet substitution."""
    variants = set()
    mapping = {"o": "0", "e": "3", "i": "1", "s": "$"}
    for i, ch in enumerate(base):
        if ch.lower() in mapping:
            variant = base[:i] + mapping[ch.lower()] + base[i+1:]
            variants.add(variant)
            break  # Only one substitution per variant
    return variants

def chain_shift_variants(base):
    """If base ends with a number, shift the numeric suffix into a new position;
       also, insert a symbol in the middle if possible."""
    variants = set()
    m = re.search(r"^(.*?)(\d+)$", base)
    if m and len(m.group(1)) >= 2:
        prefix, num_str = m.group(1), m.group(2)
        # Move number after the first character
        variants.add(base[0] + num_str + base[1:len(prefix)])
    if len(base) >= 4:
        mid = len(base) // 2
        symbols = ["!", "@", "#", "$", "%", "?", "&"]
        for sym in symbols:
            variants.add(base[:mid] + sym + base[mid:])
    return variants

def chain_repetition_variants(base):
    """Generate minimal repetition variants."""
    variants = set()
    if base:
        variants.add(base + base[-1])
    symbols = ["!", "@", "#", "$", "%", "?", "&"]
    for sym in symbols:
        variants.add(sym * 2 + base)
        variants.add(base + sym * 2)
    return variants

def chain_middle_insertion(base):
    """Insert a symbol in the middle of the base password."""
    variants = set()
    if len(base) >= 4:
        mid = len(base) // 2
        symbols = ["!", "@", "#", "$", "%", "?", "&"]
        for sym in symbols:
            variants.add(base[:mid] + sym + base[mid:])
    return variants

def generate_variants(base, max_length, chain_depth):
    """
    Generate variants from the base using both basic transformations and human-like chains.
    Optionally filter out those exceeding max_length.
    """
    basic = generate_basic_variants(base)
    advanced = generate_human_chains(base)
    combined = basic.union(advanced)
    
    # For simplicity, here we assume chain_depth=1 means no combination; chain_depth>=2
    # could allow pairwise combinations. (This sample implementation does not deeply combine.)
    # You can extend this part in the future.
    variants = combined
    if max_length is not None:
        variants = {v for v in variants if len(v) <= max_length}
    return list(variants)

# ---------------------- Scoring and Filtering ---------------------- #

def count_modifications(variant, base):
    """A very simple heuristic: count differences in characters between variant and base."""
    # In a full implementation, you would analyze the types of modifications.
    if variant == base:
        return 0
    # For simplicity, count if variant != base, add 1 per known modification keyword.
    count = 0
    if variant.lower() != base.lower():
        count += 1
    if variant != variant.capitalize():
        count += 1
    if any(x in variant for x in ["0", "3", "1", "$"]) and leetspeak(base) != variant:
        count += 1
    if variant != base and (variant[0] in "!@#$%?&" or variant[-1] in "!@#$%?&"):
        count += 1
    return count

def heuristic_score(variant, base, config):
    """Compute a heuristic score using the configuration weights."""
    score = 0.0
    mods = []
    # Check for numeric change
    if re.search(r"\d", variant) and not re.search(r"\d", base):
        mods.append("Numeric")
    elif re.search(r"\d+$", variant) and re.search(r"\d+$", base):
        if re.search(r"\d+$", variant).group() != re.search(r"\d+$", base).group():
            mods.append("Numeric")
    # Check for symbol addition
    if variant[0] in string.punctuation or variant[-1] in string.punctuation:
        mods.append("Symbol")
    # Check for capitalization change
    if variant != base and variant.lower() == base.lower():
        mods.append("Capitalization")
    # Check for leet substitution
    if any(ch in variant for ch in ["0", "3", "1", "$"]) and leetspeak(base) != variant:
        mods.append("Leet")
    # Check for shifting (a naive check: if base is in variant but not at start or end)
    if base in variant and not variant.startswith(base) and not variant.endswith(base):
        mods.append("Shift")
    # Check for repetition
    if len(variant) > len(base) and variant[-1] == variant[-2]:
        mods.append("Repetition")
    
    for mod in set(mods):
        weight = config["modification_weights"].get(mod, 1.0)
        score += weight
    return score

def penalty(mod_count):
    """Apply a penalty proportional to the number of modifications."""
    return 0.2 * mod_count

def smart_filter_candidates(variants, base, config):
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
    
    variants = generate_variants(base, config["max_length"], config["chain_depth"])
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
