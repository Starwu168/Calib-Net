from __future__ import annotations
import yaml
from dataclasses import dataclass
from typing import Any, Dict

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get(cfg: Dict[str, Any], key: str, default=None):
    cur = cfg
    for k in key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
