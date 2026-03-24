"""
src/utils/config.py — Loads configs/config.yaml
"""
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    for key, rel in cfg["paths"].items():
        abs_path = PROJECT_ROOT / rel
        abs_path.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = abs_path
    cfg["_project_root"] = PROJECT_ROOT
    return cfg
