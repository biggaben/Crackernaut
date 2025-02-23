import json
import os

CONFIG_FILE = "config.json"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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
    "max_length": 20
}

def load_configuration():
    """
    Load configuration from a JSON file, falling back to defaults if unavailable or invalid.

    Returns:
        dict: Configuration settings with non-negative modification weights.
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
            print(f"Configuration loaded from {CONFIG_FILE}")
            for mod in config["modification_weights"]:
                config["modification_weights"][mod] = max(config["modification_weights"][mod], 0.0)
            return config
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print("Using default configuration.")
            return DEFAULT_CONFIG.copy()
        except PermissionError:
            print("Error: Permission denied.")
            return DEFAULT_CONFIG.copy()
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return DEFAULT_CONFIG.copy()
    else:
        print("Config file not found. Using defaults.")
        return DEFAULT_CONFIG.copy()

def save_configuration(config):
    """
    Save configuration to a JSON file.

    Args:
        config (dict): Configuration settings to save.
    """
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {CONFIG_FILE}")
    except PermissionError:
        print("Error: Permission denied.")
    except Exception as e:
        print(f"Error saving configuration: {e}")