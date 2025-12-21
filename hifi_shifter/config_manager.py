import json
import pathlib
import os

# Define config file path (in user's home directory)
CONFIG_FILE = pathlib.Path.home() / '.vocal_shifter_config.json'

def load_config():
    """Load configuration from JSON file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")
            return {}
    return {}

def save_config(config):
    """Save configuration to JSON file."""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save config: {e}")

def get_default_model_path():
    """Get the default model path from config."""
    config = load_config()
    return config.get('default_model_path')

def set_default_model_path(path):
    """Set the default model path in config."""
    config = load_config()
    config['default_model_path'] = str(path)
    save_config(config)
