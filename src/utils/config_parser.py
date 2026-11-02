"""
Configuration Parser & Validator
Loads and manages YAML configuration files for models, training, and corruption benchmarks.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

class ConfigDict(dict):
    """Dot-accessible dictionary for clean syntax, e.g. config.model.backbone."""
    def __getattr__(self, name: str) -> Any:
        value = self.get(name)
        if isinstance(value, dict):
            return ConfigDict(value)
        return value

    def __setattr__(self, name: str, value: Any):
        self[name] = value

def load_yaml_config(config_path: Union[str, Path]) -> ConfigDict:
    """
    Safely load a YAML config file and return a dot-accessible ConfigDict.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return ConfigDict(data)

def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Recursively merge override_config into base_config.
    """
    merged = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
