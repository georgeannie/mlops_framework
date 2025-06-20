from pathlib import Path
import os

def load_config(config_path):
    path = Path(str(config_path))  # ensure it's a Path object
    if path.suffix in [".yaml", ".yml"]:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.suffix == ".json":
        import json
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file type: {path.suffix}")