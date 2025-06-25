import time
import argparse
import subprocess
import yaml
from helper.azure_pipeline_runner import run_azure_pipeline
from helper.aws_pipeline_runner import run_aws_pipeline

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    provider = config['platform']['provider'].lower()

    if provider == 'azure':
        run_azure_pipeline(config)
    elif provider == 'aws':
        run_aws_pipeline(config)
    else:
        raise ValueError("Unsupported platform: use 'azure' or 'aws'")

if __name__ == "__main__":
    main()
