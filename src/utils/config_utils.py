import yaml
from pathlib import Path
from typing import Dict, Any

def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    Update overrides base.
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML config with simple inheritance support.
    
    If config has a 'defaults' list, it loads those files first 
    (relative to the current config file) and merges them.
    
    Example:
        defaults:
          - base
    """
    path = Path(config_path)
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    if 'defaults' in config:
        defaults = config.pop('defaults')
        combined_config = {}
        
        for default_name in defaults:
            default_path = path.parent / f"{default_name}.yaml"
            if default_path.exists():
                default_config = load_config(str(default_path))
                combined_config = deep_merge(combined_config, default_config)
            else:
                raise FileNotFoundError(f"Default config not found: {default_path}")
                
        # Merge current config on top of defaults
        config = deep_merge(combined_config, config)
        
    return config
