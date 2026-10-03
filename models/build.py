from __future__ import annotations

from typing import Any

from .KATT import KATT
from .LKAT import LKAT

LKAT_VARIANTS = {
    "LKAT-T": dict(
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, patch_size=16, kan_num_grids=4,
        kan_type="rbf", use_layernorm=False, num_registers=4, rope_mode="2dv1",
    ),
    "LKAT-S": dict(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, patch_size=16, kan_num_grids=4,
        kan_type="rbf", use_layernorm=False, num_registers=4, rope_mode="2dv1",
    ),
    "LKAT-B": dict(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, patch_size=16, kan_num_grids=4,
        kan_type="rbf", use_layernorm=False, num_registers=4, rope_mode="2dv1",
    ),
    # ViT-L / DeiT-L width: embed=1024, depth=24, heads=16
    "LKAT-L": dict(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0, patch_size=16, kan_num_grids=4,
        kan_type="rbf", use_layernorm=False, num_registers=4, rope_mode="2dv1",
    ),
}

KATT_VARIANTS = {
    "KATT-T": dict(
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, patch_size=16, kan_num_grids=4,
        kan_grid_min=-1.0, kan_grid_max=1.0,
        num_registers=4, rope_mode="2dv1",
    ),
    "KATT-S": dict(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, patch_size=16, kan_num_grids=4,
        kan_grid_min=-1.0, kan_grid_max=1.0,
        num_registers=4, rope_mode="2dv1",
    ),
    "KATT-B": dict(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, patch_size=16, kan_num_grids=4,
        kan_grid_min=-1.0, kan_grid_max=1.0,
        num_registers=4, rope_mode="2dv1",
    ),
}


_VIT5_CFG_KEYS = {
    "embed_dim": "embed_dim",
    "depth": "depth",
    "num_heads": "num_heads",
    "mlp_ratio": "mlp_ratio",
    "patch_size": "patch_size",
    "num_registers": "num_registers",
    "reg_theta": "reg_theta",
    "qk_norm": "qk_norm",
    "qkv_bias": "qkv_bias",
    "flash": "flash",
    "sdpa": "sdpa",
    "drop_path_rate": "drop_path_rate",
    "init_scale": "init_scale",
    "layer_scale": "layer_scale",
}

_SHARED_CFG_KEYS = {
    "embed_dim": "embed_dim",
    "depth": "depth",
    "num_heads": "num_heads",
    "mlp_ratio": "mlp_ratio",
    "patch_size": "patch_size",
    "kan_num_grids": "kan_num_grids",
    "grid": "kan_num_grids",
    "kan_type": "kan_type",
    "kan_grid_min": "kan_grid_min",
    "kan_grid_max": "kan_grid_max",
    "kan_compile": "kan_compile",
    "use_base_update": "use_base_update",
    "use_layernorm": "use_layernorm",
    "drop_path_rate": "drop_path_rate",
    "use_cls_token": "use_cls_token",
    "num_registers": "num_registers",
    "use_abs_pos": "use_abs_pos",
    "attn_drop_rate": "attn_drop_rate",
}

_LKAT_CFG_KEYS = {
    **_SHARED_CFG_KEYS,
    "rope_mode": "rope_mode",
    "rope_theta": "rope_theta",
    "expand_k": "expand_k",
    "expand_v": "expand_v",
    "use_short_conv": "use_short_conv",
    "conv_size": "conv_size",
}

_KATT_CFG_KEYS = {
    **_SHARED_CFG_KEYS,
    "rope_mode": "rope_mode",
    "rope_theta": "rope_theta",
}

_VIG_CFG_KEYS = {
    **_SHARED_CFG_KEYS,
    "rope_mode": "rope_mode",
    "attn_model": "attn_model",
    "expand_k": "expand_k",
    "expand_v": "expand_v",
    "patch_embed_version": "patch_embed_version",
    "hidden_act": "hidden_act",
    "use_out_gate": "use_out_gate",
    "use_out_act": "use_out_act",
    "use_dirpe": "use_dirpe",
    "classification_mode": "classification_mode",
    "scan_mode": "scan_mode",
    "stride": "stride",
    "drop_path_rate": "drop_path_rate",
}

VIG_VARIANTS = {
    "ViG-T": dict(
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, patch_size=16, kan_num_grids=4,
        rope_mode="2dv1", attn_model="fused_recurrent", expand_k=0.5, expand_v=1.0,
        patch_embed_version="v2",
    ),
    "ViG-S": dict(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, patch_size=16, kan_num_grids=4,
        rope_mode="2dv1", attn_model="fused_recurrent", expand_k=0.5, expand_v=1.0,
        patch_embed_version="v2",
    ),
    "ViG-B": dict(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, patch_size=16, kan_num_grids=4,
        rope_mode="2dv1", attn_model="fused_recurrent", expand_k=0.5, expand_v=1.0,
        patch_embed_version="v2",
    ),
}


def lkat_tiny(**kw):
    """Build LKAT-Tiny."""
    return LKAT(embed_dim=192, depth=12, num_heads=3, **kw)


def lkat_small(**kw):
    """Build LKAT-Small."""
    return LKAT(embed_dim=384, depth=12, num_heads=6, **kw)


def lkat_base(**kw):
    """Build LKAT-Base."""
    return LKAT(embed_dim=768, depth=12, num_heads=12, **kw)


def lkat_large(**kw):
    """Build LKAT-Large."""
    return LKAT(embed_dim=1024, depth=24, num_heads=16, **kw)


def build_lkat(
    name: str = "LKAT-S",
    num_classes: int = 10,
    image_size: int = 224,
    **overrides,
) -> LKAT:
    """Build an LKAT variant by name with optional overrides."""
    if name not in LKAT_VARIANTS:
        raise ValueError(f"Unknown LKAT variant '{name}'. Choose from {list(LKAT_VARIANTS)}")
    cfg = {**LKAT_VARIANTS[name], **overrides}
    return LKAT(
        img_size=image_size,
        num_classes=num_classes,
        **cfg,
    )


def build_katt(
    name: str = "KATT-S",
    num_classes: int = 10,
    image_size: int = 224,
    **overrides,
) -> KATT:
    """Build a KATT variant by name with optional overrides."""
    if name not in KATT_VARIANTS:
        raise ValueError(f"Unknown KATT variant '{name}'. Choose from {list(KATT_VARIANTS)}")
    cfg = {**KATT_VARIANTS[name], **overrides}
    return KATT(
        img_size=image_size,
        num_classes=num_classes,
        **cfg,
    )


def _apply_cfg_keys(model_cfg: dict[str, Any], cfg: dict[str, Any], key_map: dict[str, str]) -> None:
    """Copy YAML keys into model constructor kwargs."""
    for yaml_key, ctor_key in key_map.items():
        if yaml_key in cfg:
            model_cfg[ctor_key] = cfg[yaml_key]
    if "dropout" in cfg:
        model_cfg["drop_rate"] = cfg["dropout"]
    model_cfg.setdefault("drop_path_rate", 0.0)
    model_cfg.setdefault("use_cls_token", True)
    model_cfg.setdefault("num_registers", 4)
    model_cfg.setdefault("rope_mode", "2dv1")
    if int(model_cfg.get("num_registers", 4)) < 1:
        raise ValueError("num_registers must be >= 1")
    if str(model_cfg.get("rope_mode", "2dv1")) not in ("2dv0", "2dv1"):
        raise ValueError("rope_mode must be '2dv0' or '2dv1'")


def build_lkat_from_cfg(cfg: dict[str, Any], num_classes: int) -> LKAT:
    """Build LKAT from a training YAML config."""
    variant = cfg.get("model", "LKAT-S")
    if variant not in LKAT_VARIANTS:
        raise ValueError(f"Unknown model '{variant}'. Choose from {list(LKAT_VARIANTS)}")

    model_cfg = dict(LKAT_VARIANTS[variant])
    _apply_cfg_keys(model_cfg, cfg, _LKAT_CFG_KEYS)
    return LKAT(
        img_size=int(cfg.get("input_size", 224)),
        in_chans=3,
        num_classes=num_classes,
        **model_cfg,
    )


def build_katt_from_cfg(cfg: dict[str, Any], num_classes: int) -> KATT:
    """Build KATT from a training YAML config."""
    variant = cfg.get("model", "KATT-S")
    if variant not in KATT_VARIANTS:
        raise ValueError(f"Unknown model '{variant}'. Choose from {list(KATT_VARIANTS)}")

    model_cfg = dict(KATT_VARIANTS[variant])
    _apply_cfg_keys(model_cfg, cfg, _KATT_CFG_KEYS)
    return KATT(
        img_size=int(cfg.get("input_size", 224)),
        in_chans=3,
        num_classes=num_classes,
        **model_cfg,
    )


def build_vig_from_cfg(cfg: dict[str, Any], num_classes: int):
    """Build ViG from a training YAML config."""
    from .VIG import ViG

    variant = cfg.get("model", "ViG-S")
    if variant not in VIG_VARIANTS:
        raise ValueError(f"Unknown model '{variant}'. Choose from {list(VIG_VARIANTS)}")

    model_cfg = dict(VIG_VARIANTS[variant])
    _apply_cfg_keys(model_cfg, cfg, _VIG_CFG_KEYS)
    if "dropout" in cfg:
        model_cfg["drop_rate"] = cfg["dropout"]
    model_cfg.setdefault("drop_path_rate", 0.1)
    if str(model_cfg.get("rope_mode", "2dv1")) not in ("2dv0", "2dv1", "none"):
        raise ValueError("rope_mode must be '2dv0', '2dv1', or 'none'")
    return ViG(
        img_size=int(cfg.get("input_size", 224)),
        channels=3,
        num_classes=num_classes,
        **model_cfg,
    )


def build_mixer(cfg: dict[str, Any], num_classes: int):
    """Build an MLP-Mixer from a training YAML config."""
    from timm.models.mlp_mixer import MlpMixer

    return MlpMixer(
        num_classes=num_classes,
        img_size=int(cfg.get("input_size", 224)),
        patch_size=int(cfg.get("patch_size", 16)),
        num_blocks=int(cfg.get("depth", 12)),
        embed_dim=int(cfg.get("embed_dim", 768)),
        drop_rate=float(cfg.get("dropout", 0.0)),
        drop_path_rate=float(cfg.get("drop_path_rate", 0.0)),
    )


def build_vit5_from_cfg(cfg: dict[str, Any], num_classes: int):
    """Build ViT-5 from a training YAML config."""
    from .vit5 import VIT5_VARIANTS, build_vit5

    variant = str(cfg.get("model", "ViT5-B"))
    if variant not in VIT5_VARIANTS:
        raise ValueError(f"Unknown model '{variant}'. Choose from {list(VIT5_VARIANTS)}")

    overrides: dict[str, Any] = {}
    for yaml_key, ctor_key in _VIT5_CFG_KEYS.items():
        if yaml_key in cfg:
            overrides[ctor_key] = cfg[yaml_key]
    if "dropout" in cfg:
        overrides["drop_rate"] = cfg["dropout"]

    return build_vit5(
        name=variant,
        num_classes=num_classes,
        image_size=int(cfg.get("input_size", 224)),
        **overrides,
    )


def build_model(cfg: dict[str, Any], num_classes: int):
    """Build a model from config (LKAT/KATT/ViT5/timm/etc.)."""
    timm_name = cfg.get("timm_model")
    if timm_name:
        import timm

        pretrained = cfg.get("pretrained", False)
        if pretrained is None:
            pretrained = False
        if str(timm_name).startswith("mixer") and ("embed_dim" in cfg or "depth" in cfg):
            if bool(pretrained):
                raise ValueError("pretrained weights unavailable for custom Mixer dims")
            return build_mixer(cfg, num_classes)
        kwargs: dict[str, Any] = {"num_classes": num_classes}
        if "drop_path_rate" in cfg:
            kwargs["drop_path_rate"] = float(cfg["drop_path_rate"])
        if "dropout" in cfg:
            kwargs["drop_rate"] = float(cfg["dropout"])
        return timm.create_model(
            str(timm_name),
            pretrained=bool(pretrained),
            **kwargs,
        )

    variant = str(cfg.get("model", "LKAT-S"))
    if variant in KATT_VARIANTS or variant.startswith("KATT"):
        return build_katt_from_cfg(cfg, num_classes)
    # Additive RBF-gating LKAT variant (separate module; must run before LKAT-* catch-all).
    from .LKAT_rbf_gating import is_lkat_rbf_gating_model, build_lkat_rbf_gating_from_cfg

    if is_lkat_rbf_gating_model(variant):
        return build_lkat_rbf_gating_from_cfg(cfg, num_classes)
    if variant in LKAT_VARIANTS or variant.startswith("LKAT"):
        return build_lkat_from_cfg(cfg, num_classes)
    if variant.startswith("ViT5"):
        return build_vit5_from_cfg(cfg, num_classes)
    if variant in VIG_VARIANTS or variant.startswith("ViG"):
        return build_vig_from_cfg(cfg, num_classes)

    raise ValueError(
        f"Unknown model '{variant}'. Use LKAT-*, KATT-*, ViG-*, ViT5-*, or timm_model=<id> "
        f"(Swin / PVT / ViT / Mixer)."
    )


__all__ = [
    "KATT_VARIANTS",
    "LKAT_VARIANTS",
    "VIG_VARIANTS",
    "build_katt",
    "build_katt_from_cfg",
    "build_lkat",
    "build_lkat_from_cfg",
    "build_model",
    "build_mixer",
    "build_vig_from_cfg",
    "build_vit5_from_cfg",
    "lkat_base",
    "lkat_large",
    "lkat_small",
    "lkat_tiny",
    "vit_base_gla_kan",
    "vit_small_gla_kan",
    "vit_tiny_gla_kan",
]
