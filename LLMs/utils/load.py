import os
import json


def load_config(default_path: str = "config.json") -> dict:
    if not os.path.exists(default_path):
        raise FileNotFoundError(f'Default config is not found: {default_path}')

    with open(default_path) as f:
        return json.load(f)
