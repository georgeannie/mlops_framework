from pathlib import Path
import os
import argparse

#def resolve_config_path(env_var: str = "CONFIG_PATH", default_relative: str = "config.yaml") -> Path:
def resolve_config_path():
    """
    Resolves the configuration file path using the following priority:
    1. Environment variable (e.g., CONFIG_PATH)
    2. Default relative path from script location

    Args:
        env_var (str): The environment variable name to check for config path override.
        default_relative (str): Default relative path from script root.

    Returns:
        Path: Resolved absolute path to the configuration file.

    Raises:
        FileNotFoundError: If the resolved config path does not exist.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to job config YAML")
    args = parser.parse_args()
    return args.config

    # # Step 1: Compute default path relative to this file
    # default_path = Path(__file__).resolve().parent.parent.parent / default_relative

    # # Step 2: Override via environment variable (if set)
    # config_path = Path(os.getenv(env_var, str(default_path))).expanduser().resolve()

    # # Step 3: Check for file existence
    # if not config_path.exists():
    #     raise FileNotFoundError(f"Config file not found at: {config_path}")

    # print(f"[config] Using config path: {config_path}")
    # return config_path
