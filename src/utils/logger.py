"""
src/utils/logger.py — Centralized logger (console + daily file)
"""
import logging
from datetime import datetime
from .config import load_config

def get_logger(name: str) -> logging.Logger:
    cfg     = load_config()
    log_dir = cfg["paths"]["logs"]
    logger  = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
