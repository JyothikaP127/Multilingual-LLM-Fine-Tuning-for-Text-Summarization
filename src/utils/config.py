"""Minimal YAML config loading. Paths are resolved relative to the repo root
(the parent of the configs/ directory), never as absolute machine paths.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def load_yaml(relative_path: str) -> dict:
    path = CONFIGS_DIR / relative_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(relative_path: str) -> Path:
    return REPO_ROOT / relative_path
