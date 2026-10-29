"""Loads config/pipeline_config.json and resolves model paths.

Environment variables (optional):
  PIPELINE_CONFIG   path to the config file (default: <app root>/config/pipeline_config.json)
  MODELS_DIR        folder that holds the model files (default: <app root>/models)
  ALLOW_SETTINGS_MISMATCH=1   run even if the settings differ from what Track A was trained with
"""
from __future__ import annotations

import json
import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("PIPELINE_CONFIG", APP_ROOT / "config" / "pipeline_config.json"))
    with open(path) as f:
        cfg = json.load(f)
    models_dir = os.environ.get("MODELS_DIR")
    for key, rel in cfg["models"].items():
        p = Path(rel)
        if models_dir:
            p = Path(models_dir) / p.name
        elif not p.is_absolute():
            p = APP_ROOT / p
        cfg["models"][key] = p
    cfg["allow_settings_mismatch"] = os.environ.get("ALLOW_SETTINGS_MISMATCH", "0") == "1"
    return cfg
