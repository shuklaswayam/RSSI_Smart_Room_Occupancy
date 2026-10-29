import os
from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load and parse master system configuration YAML."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Resolve project root relative paths
    root_dir = path.resolve().parent.parent
    if "paths" in cfg:
        for k, v in cfg["paths"].items():
            if isinstance(v, str) and not os.path.isabs(v):
                cfg["paths"][k] = str(root_dir / v)
    
    return cfg

def get_root_dir() -> Path:
    """Return the absolute path to the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent
