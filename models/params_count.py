from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.build import LKAT_VARIANTS, build_lkat


def count_params(model: nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_m(n: int | float) -> str:
    """Format a parameter count as millions (e.g. ``12.34M``)."""
    return f"{n / 1e6:.2f}M"


def count_lkat_variants(
    num_classes: int = 1000,
    image_size: int = 224,
) -> dict[str, int]:
    """Count trainable params for each LKAT variant."""
    out: dict[str, int] = {}
    for name in LKAT_VARIANTS:
        model = build_lkat(name, num_classes=num_classes, image_size=image_size)
        out[name] = count_params(model)
        del model
    return out


def count_kan_mlp_params(model: nn.Module) -> int:
    """Count params in per-block Triton RBF KAN MLP stacks only."""
    total = 0
    for blk in getattr(model, "blocks", []):
        total += count_params(blk.mlp)
    return total


BASELINES: dict[str, list[tuple[str, float, str]]] = {
    "ViT / DeiT (reference)": [
        ("DeiT-Tiny", 5.0, "embed=192, depth=12"),
        ("DeiT-Small", 22.0, "embed=384, depth=12"),
        ("DeiT-Base", 86.0, "embed=768, depth=12"),
    ],
}


def print_table() -> None:
    """Print LKAT parameter counts and literature baselines."""
    lkat = count_lkat_variants()

    print("=" * 78)
    print("LKAT parameter counts (T / S / B / L) — Triton RBF KAN")
    print("=" * 78)
    print(f"{'Model':<12} {'Params':>10} {'KAN MLP':>10}  Config")
    print("-" * 78)
    for name, n in lkat.items():
        cfg = LKAT_VARIANTS[name]
        model = build_lkat(name)
        kan_n = count_kan_mlp_params(model)
        del model
        cfg_str = (
            f"embed={cfg['embed_dim']}, depth={cfg['depth']}, "
            f"heads={cfg['num_heads']}, grid={cfg['kan_num_grids']}"
        )
        print(f"{name:<12} {format_m(n):>10} {format_m(kan_n):>10}  {cfg_str}")

    print()
    print("=" * 78)
    print("Optional literature baselines (scale reference only)")
    print("=" * 78)
    for family, rows in BASELINES.items():
        print(f"\n[{family}]")
        print(f"  {'Model':<22} {'Params':>8}  Notes")
        print(f"  {'-' * 60}")
        for model, mparams, notes in rows:
            print(f"  {model:<22} {mparams:>6.1f}M  {notes}")

    print()
    print("Notes:")
    print("  - LKAT = GatedLinearAttention + Triton RBF KAN (KANLayer / triton_rbf).")
    print("  - Counts measured from models.LKAT.LKAT + models.build.build_lkat.")


if __name__ == "__main__":
    print_table()
