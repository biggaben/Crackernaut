#!/usr/bin/env python3
import argparse
import logging
import torch
import os
import sys

from src.utils.config_utils import load_configuration, save_configuration
from src.utils.variant_utils import generate_variants
# from src.utils.performance_utils import measure_processing_time
from src.list_preparer import prepare_list
from src.models.transformer.transformer_model import PasswordTransformer
from src.cuda_ml import load_ml_model


CONFIG_FILE = "config.json"
MODEL_FILE = "ml_model.pth"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('crackernaut')

def setup_device():
    """Set up and return the computation device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device

def load_model(config, device):
    model_type = config.get("model_type", "rnn").lower()
    if model_type == "transformer":
        model = PasswordTransformer(
            vocab_size=128,
            embed_dim=config.get("model_embed_dim", 64),
            num_heads=config.get("model_num_heads", 4),
            num_layers=config.get("model_num_layers", 3),
            hidden_dim=config.get("model_hidden_dim", 128),
            dropout=config.get("model_dropout", 0.2),
            output_dim=6
        ).to(device)
    else:
        model = load_ml_model(config, model_type=model_type).to(device)
    
    model_dir = os.path.join(os.path.dirname(__file__), "models", model_type)
    model_file_path = os.path.join(model_dir, f"{model_type}_model.pth")
    if os.path.exists(model_file_path):
        model.load_state_dict(torch.load(model_file_path, map_location=device))
        logger.info(f"Loaded model from {model_file_path}")
    else:
        logger.warning(f"Model file not found at {model_file_path}. Proceeding with untrained model.")
        # Don’t return None; proceed with fresh model
    return model

def score_password_variants(model, base_password, variants, device):
    """Score password variants using the model"""
    model.eval()
    tensors = text_to_tensor([base_password] + variants, device=device)
    with torch.no_grad():
        scores = model(tensors)
    return scores.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Crackernaut Password Variant Generator")
    parser.add_argument("password", type=str, help="Base password to generate variants from")
    parser.add_argument("depth", type=int, help="Depth of variant generation")
    args = parser.parse_args()

    config = load_configuration(CONFIG_FILE)
    device = setup_device()

    if config["model_embed_dim"] % config["model_num_heads"] != 0:
        logger.warning("model_embed_dim must be divisible by model_num_heads for transformer model. Adjusting model_embed_dim.")
        config["model_embed_dim"] = (config["model_embed_dim"] // config["model_num_heads"]) * config["model_num_heads"]

    model = load_model(config, device)
    if model is None:
        logger.error("Failed to load model.")
        exit(1)

    variants = generate_variants(base_pw, config["max_length"], config["chain_depth"], config=config)
    for variant in variants:
        print(variant)

if __name__ == "__main__":
    main()