#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from training_engine import Trainer

_SECTION_KEYS = ("train_config", "train", "preprocessing_recipe", "preprocess", "preprocessing")


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def _flatten_sections(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    for key in _SECTION_KEYS:
        section = out.pop(key, None)
        if isinstance(section, dict):
            out.update(section)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    return _flatten_sections(_load_yaml(Path(path)))


def main():
    parser = argparse.ArgumentParser(description="Train ViK")
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    # Free throughput on Ampere+/Blackwell (does not change bf16 AMP recipe).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = load_config(args.config)
    cfg["device"] = "cuda"
    cfg.setdefault("fsdp", True)
    cfg.setdefault("recipe", "fast")

    for item in args.overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        try:
            value = yaml.safe_load(value)
        except Exception:
            pass
        cfg[key] = value

    print(
        f"[train] {cfg.get('model')} | {cfg.get('dataset')} | "
        f"gpus={cfg.get('num_gpus')} epochs={cfg.get('epochs')} "
        f"bs={cfg.get('batch_size')} amp={cfg.get('amp_dtype')}"
    )
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
