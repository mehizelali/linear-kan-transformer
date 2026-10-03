from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .KANLayer_rbf_gating import KANLayerRBFGating
from .LKAT import LKAT


class KANRBFGating(nn.Module):
    """Two-layer RBF-KAN MLP using ``KANLayerRBFGating``."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_grids: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        drop: float = 0.0,
        use_layernorm: bool = True,
        use_compile: bool = True,
    ):
        """Build a two-layer gated RBF-KAN MLP."""
        super().__init__()
        self.use_compile = use_compile
        self._compile_key = ("KANRBFGating", id(self))
        self.fc1 = KANLayerRBFGating(
            dim,
            hidden_dim,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_layernorm=use_layernorm,
        )
        self.fc2 = KANLayerRBFGating(
            hidden_dim,
            dim,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_layernorm=use_layernorm,
        )
        self.drop = nn.Dropout(drop)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Run gated KAN layers with dropout."""
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated RBF-KAN MLP."""
        return self._forward_impl(x)


class LKATRBFGating(LKAT):
    """LKAT with RBF-gating KAN MLPs (``LKAT-B-RBF-Gating``)."""

    def __init__(self, *args, **kwargs):
        """Build LKAT then swap each block MLP for gated RBF-KAN."""
        kwargs = dict(kwargs)
        kwargs["use_base_update"] = False
        super().__init__(*args, **kwargs)
        for blk in self.blocks:
            old = blk.mlp
            blk.mlp = KANRBFGating(
                dim=old.fc1.input_dim,
                hidden_dim=old.fc1.output_dim,
                num_grids=old.fc1.num_grids,
                grid_min=float(old.fc1.rbf.grid_min),
                grid_max=float(old.fc1.rbf.grid_max),
                drop=float(old.drop.p) if isinstance(old.drop, nn.Dropout) else 0.0,
                use_layernorm=old.fc1.layernorm is not None,
                use_compile=bool(getattr(old, "use_compile", True)),
            )
        self._init_gat_resid_weights()

    def _init_gat_resid_weights(self) -> None:
        """Initialize gated residual linear weights."""
        for m in self.modules():
            if isinstance(m, KANLayerRBFGating):
                nn.init.trunc_normal_(m.gat_resid.weight, std=0.02)
                if m.gat_resid.bias is not None:
                    nn.init.zeros_(m.gat_resid.bias)


LKAT_RBF_GATING_VARIANTS = {
    "LKAT-B-RBF-Gating": dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        patch_size=16,
        kan_num_grids=4,
        kan_type="rbf",
        use_layernorm=False,
        num_registers=4,
        rope_mode="2dv1",
    ),
}


def _resolve_rbf_gating_variant(name: str) -> str:
    """Map a model name/alias to an RBF-gating variant key."""
    if name in LKAT_RBF_GATING_VARIANTS:
        return name
    key = name.replace("_", "-")
    aliases = {
        "LKAT-B-rbf-gating": "LKAT-B-RBF-Gating",
        "LKAT-B-RBF-gating": "LKAT-B-RBF-Gating",
    }
    if key in aliases:
        return aliases[key]
    if "rbf" in name.lower() and "gat" in name.lower() and "LKAT-B" in name.upper().replace("_", "-"):
        return "LKAT-B-RBF-Gating"
    raise ValueError(
        f"Unknown RBF-gating model '{name}'. Choose from {list(LKAT_RBF_GATING_VARIANTS)}"
    )


def build_lkat_rbf_gating_from_cfg(cfg: dict[str, Any], num_classes: int) -> LKATRBFGating:
    """Build an RBF-gating LKAT model from a YAML config dict."""
    from .build import _LKAT_CFG_KEYS, _apply_cfg_keys

    variant = _resolve_rbf_gating_variant(str(cfg.get("model", "LKAT-B-RBF-Gating")))
    model_cfg = dict(LKAT_RBF_GATING_VARIANTS[variant])
    _apply_cfg_keys(model_cfg, cfg, _LKAT_CFG_KEYS)
    return LKATRBFGating(
        img_size=int(cfg.get("input_size", 224)),
        in_chans=3,
        num_classes=num_classes,
        **model_cfg,
    )


def is_lkat_rbf_gating_model(name: str) -> bool:
    """Return True if the model name is an RBF-gating LKAT variant."""
    n = str(name)
    if n in LKAT_RBF_GATING_VARIANTS:
        return True
    low = n.lower().replace("_", "-")
    return "rbf-gating" in low or "rbf_gating" in n.lower()


__all__ = [
    "KANRBFGating",
    "KANLayerRBFGating",
    "LKATRBFGating",
    "LKAT_RBF_GATING_VARIANTS",
    "build_lkat_rbf_gating_from_cfg",
    "is_lkat_rbf_gating_model",
]
