#!/usr/bin/env python3
import argparse
import logging
import torch
import os
import sys
from config_utils import load_configuration, save_configuration
from variant_utils import generate_variants
from performance_utils import measure_processing_time

# Add transformer model import
from transformer_model import PasswordTransformer
# Keep import for legacy models
from cuda_ml import load_ml_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('crackernaut')

def setup_device():
    """Set up and return the computation device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device

def load_model(config, device):
    """
    Load a model based on configuration
    
    Args:
        config: Configuration dictionary
        device: Computation device
        
    Returns:
        Loaded model
    """
    model_type = config.get("model_type", "transformer").lower()
    model_path = config.get("model_path", "ml_model.pth")
    
    if model_type == "transformer":
        logger.info("Loading transformer model")
        model = PasswordTransformer(
            vocab_size=128,
            embed_dim=config.get("model_embed_dim", 64),
            num_heads=config.get("model_num_heads", 4),
            num_layers=config.get("model_num_layers", 3),
            hidden_dim=config.get("model_hidden_dim", 128),
            dropout=config.get("model_dropout", 0.2),
            output_dim=6  # For the six variant types
        ).to(device)
        
        # Try to load saved model weights
        if not model.load(model_path, device):
            logger.warning(f"Could not load transformer model weights from {model_path}")
    else:
        # Load legacy model (MLP, RNN, BiLSTM)
        logger.info(f"Loading legacy {model_type} model")
        model = load_ml_model(config, model_type=model_type, device=device)
    
    return model

def score_password_variants(model, base_password, variants, device):
    """
    Score variants for a password
    
    Args:
        model: Model to use for scoring
        base_password: Original password
        variants: List of variant passwords
        device: Computation device
        
    Returns:
        Dictionary with scored variants
    """
    # If it's the transformer model, use its batch_score method
    if isinstance(model, PasswordTransformer):
        # Include the base password at the beginning
        all_passwords = [base_password] + variants
        scores_list = model.batch_score(all_passwords, device=device)
        
        # Extract base score and variant scores
        base_score = scores_list[0]
        variant_scores = scores_list[1:]
    else:
        # Legacy model scoring
        # (Assuming the legacy model has methods to score passwords)
        # This would need to be implemented based on the actual legacy model interface
        logger.warning("Legacy model scoring not fully implemented")
        base_score = {}
        variant_scores = [{} for _ in variants]
    
    # Return results in the expected format
    results = {
        "base_password": base_password,
        "base_score": base_score,
        "variants": []
    }
    
    # Add variant info
    for i, variant in enumerate(variants):
        results["variants"].append({
            "password": variant,
            "score": variant_scores[i]
        })
    
    return results

def main():
    """Main function to run Crackernaut"""
    parser = argparse.ArgumentParser(description="Crackernaut - Password Variant Generator & Scorer")
    parser.add_argument("--password", "-p", type=str, help="Base password to analyze")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--depth", "-d", type=int, help="Chain depth for variant generation")
    parser.add_argument("--model", "-m", type=str, help="Model type: transformer, rnn, bilstm, mlp")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_configuration(args.config)
    
    # Override configuration with command-line arguments
    if args.depth:
        config["chain_depth"] = args.depth
    if args.model:
        config["model_type"] = args.model
    
    # Set up device
    device = setup_device()
    
    # Load model
    model = load_model(config, device)
    
    # Get base password
    base_password = args.password
    if not base_password:
        base_password = input("Enter a base password: ").strip()
    
    # Generate variants
    with measure_processing_time("Variant generation"):
        variants = generate_variants(base_password, config)
    
    logger.info(f"Generated {len(variants)} variants for '{base_password}'")
    
    # Score variants
    with measure_processing_time("Variant scoring"):
        results = score_password_variants(model, base_password, variants, device)
    
    # Print results
    print(f"\nBase password: {base_password}")
    print("Top variants by score:")
    
    # Sort variants by highest score
    def get_max_score(var_data):
        return max(var_data["score"].values())
    
    sorted_variants = sorted(results["variants"], key=get_max_score, reverse=True)
    
    for i, var_data in enumerate(sorted_variants[:10]):  # Show top 10
        variant = var_data["password"]
        max_score = get_max_score(var_data)
        print(f"{i+1}. {variant} (Score: {max_score:.4f})")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())