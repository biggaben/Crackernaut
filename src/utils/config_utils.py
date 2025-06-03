import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('config_utils')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(PROJECT_ROOT, "config.json")
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
    "model_type": "transformer",  # New default option for transformer-based scoring
    "model_embed_dim": 64,
    "model_num_heads": 4,
    "model_num_layers": 3,
    "model_hidden_dim": 128,
    "model_dropout": 0.2,
    "lp_chunk_size": 1000000,      # Chunk size for list preparation
    "lp_output_dir": "clusters"     # Output directory for clusters
}

def load_configuration(config_file=CONFIG_FILE):
    """
    Load configuration from a JSON file or use default.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                logger.info(f"Configuration loaded from {config_file}")
                
                # Ensure all default keys exist by combining with defaults
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                        
                return config
        else:
            logger.warning(f"Configuration file {config_file} not found, using defaults")
            return DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return DEFAULT_CONFIG.copy()

def save_configuration(config, config_file=CONFIG_FILE):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_file: Path to save configuration
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Configuration saved to {config_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False

if __name__ == "__main__":
    # Create a default configuration file if one doesn't exist
    if not os.path.exists(CONFIG_FILE):
        save_configuration(DEFAULT_CONFIG)
        print("Created default configuration file: config.json")
    else:
        print("Configuration file already exists")