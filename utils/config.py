# utils/config.py
# ---------------------------------------------------------
# YAML loader + simple path setup
# ---------------------------------------------------------
from __future__ import annotations
import os, io, copy
import yaml

def load_config(path: str) -> dict:
    with io.open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # basic interpolation for paths using already-defined keys
    cfg = _resolve_paths(cfg)
    return cfg

def ensure_dirs(cfg: dict):
    for k in ["work_dir", "log_dir", "ckpt_dir"]:
        d = cfg["paths"][k]
        os.makedirs(d, exist_ok=True)

def dump_config(cfg: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

def _resolve_paths(cfg: dict) -> dict:
    """Resolve ${...} placeholders for paths.* keys."""
    cfg = copy.deepcopy(cfg)
    wd = cfg["paths"]["work_dir"]
    for key in list(cfg["paths"].keys()):
        val = cfg["paths"][key]
        if isinstance(val, str):
            val = val.replace("${paths.work_dir}", wd)
            cfg["paths"][key] = val
    return cfg
