#!/usr/bin/env python3
"""
crackernaut.py

Main production script for Crackernaut – a utility that generates human-like
password variants from a given base password. The script loads configuration
parameters (from a JSON file or defaults), generates variants via transformation
chains, applies smart filtering based on a machine learning model, and outputs
the top variants to the console or a specified file.

Usage:
    python crackernaut.py -p <base_password> [-l <max_length>] [-n <output_count>] [-o <output_file>]
    
    If -p is not provided, the script prompts for the base password.
    The -l flag sets the maximum length of variants.
    The -n flag limits the number of variants output.
    The -o flag saves the output variants to a file.
"""

import argparse
import os
import re
import sys
import torch
from config_utils import load_configuration
from variant_utils import generate_variants, SYMBOLS
from cuda_ml import load_ml_model, extract_features, predict_config_adjustment

def score_variants(variants, base, model, config, device):
    if not variants:
        return []
    
    features_list = [extract_features(var, base, device) for var in variants]
    features_tensor = torch.cat(features_list, dim=0).to(device)
    
    predictions = predict_config_adjustment(features_tensor, model)
    
    weights = torch.tensor([config["modification_weights"][mod] for mod in ["Numeric", "Symbol", "Capitalization", "Leet", "Shift", "Repetition"]], device=device)
    
    scores = (predictions * weights).sum(dim=1).cpu().numpy()
    
    scored_variants = list(zip(variants, scores))
    scored_variants.sort(key=lambda x: x[1], reverse=True)
    
    return scored_variants

def main():
    """Main function to generate and output password variants."""
    parser = argparse.ArgumentParser(description="Crackernaut Variant Generator")
    parser.add_argument("-p", "--password", type=str, help="Base password (if not provided, prompt for input)")
    parser.add_argument("-l", "--length", type=int, help="Maximum length of generated variants")
    parser.add_argument("-n", "--number", type=int, help="Limit number of variants to output")
    parser.add_argument("-o", "--output", type=str, help="File to save the variants")
    args = parser.parse_args()
    
    # Load configuration settings from config_utils
    config = load_configuration()
    
    # Load the ML model from cuda_ml and determine the device (CPU or CUDA)
    model = load_ml_model()
    device = next(model.parameters()).device
    
    if args.password:
        base = args.password.strip()
    else:
        base = input("Enter base password: ").strip()
    
    # Validate the base password: must contain letters and be at least 3 characters
    while not base or len(base) < 3 or not re.search(r"[a-zA-Z]", base):
        print("Invalid base password. Must contain letters and be at least 3 characters.")
        base = input("Enter base password: ").strip()
    
    # Override the config's max_length if provided via args
    if args.length:
        config["max_length"] = args.length
    
    # Generate variants using variant_utils
    variants = generate_variants(base, config["max_length"], config["chain_depth"])
    
    if not variants:
        print("No variants generated. Try adjusting the configuration.")
        return
    
    # Score the generated variants using the ML model
    scored_variants = score_variants(variants, base, model, config, device)
    
    if not scored_variants:
        print("No variants meet the scoring criteria.")
        return
    
    # Select the top N variants if specified, otherwise select all
    if args.number:
        output_variants = [var for var, score in scored_variants[:args.number]]
    else:
        output_variants = [var for var, score in scored_variants]
    
    # Handle output: save to file if specified, otherwise print to console
    if args.output:
        try:
            # Check if the output file already exists
            if os.path.exists(args.output):
                choice = input(f"File {args.output} exists. Overwrite (o), append (a), or cancel (c)? ").strip().lower()
                if choice == 'o':
                    mode = 'w'  # Overwrite
                elif choice == 'a':
                    mode = 'a'  # Append
                else:
                    print("Operation canceled.")
                    return
            else:
                mode = 'w'  # Write new file
            # Write variants to the file
            with open(args.output, mode) as f:
                for variant in output_variants:
                    f.write(variant + "\n")
            print(f"Variants saved to {args.output}")
        except IOError as e:
            print(f"Error saving variants: {e}")
    else:
        # Output variants to the console
        print("Generated Variants:")
        for variant in output_variants:
            print(variant)

if __name__ == "__main__":
    main()