from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .LKAT import (
    DropPath,
    KAN,
    PatchEmbed,
    RMSNorm,
    RotaryEmbeddingFast,
    RotaryEmbeddingFast2D,
    _to_2tuple,
)
from .KANLayer import KANLayer


class MultiHeadSelfAttention(nn.Module):
    """Full bidirectional MHSA with QK-RMSNorm + 2D RoPE on patch tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_mode: str = "2dv1",
        rope_theta: float = 10000.0,
        num_patches: int = 196,
        patch_resolution=(14, 14),
        qk_norm_eps: float = 1e-6,
        **kwargs,
    ):
        """Build MHSA with QK-norm and 2D RoPE."""
        super().__init__()
        _ = kwargs  # ignore legacy alibi_normalize / layerscale_init
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        if rope_mode not in ("2dv0", "2dv1"):
            raise ValueError(f"rope_mode must be '2dv0' or '2dv1', got {rope_mode!r}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.num_patches = int(num_patches)
        self.patch_resolution = _to_2tuple(patch_resolution)
        self.rope_mode = rope_mode

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if rope_mode == "2dv0":
            self.rotary = RotaryEmbeddingFast(
                embed_dims=self.head_dim,
                patch_resolution=self.patch_resolution,
                theta=rope_theta,
            )
        else:
            self.rotary = RotaryEmbeddingFast2D(
                embed_dims=self.head_dim,
                patch_resolution=self.patch_resolution,
                theta=rope_theta,
            )

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D RoPE to patch tokens; leave extras unchanged."""
        # q, k: (B, H, T, D)
        n = self.num_patches
        t = q.shape[2]
        extra = t - n
        if extra < 0:
            raise RuntimeError(f"seq_len {t} < num_patches {n}")
        if self.rope_mode == "2dv0":
            q_p = self.rotary(q[:, :, :n], self.patch_resolution)
            k_p = self.rotary(k[:, :, :n], self.patch_resolution)
            if extra:
                q = torch.cat([q_p, q[:, :, n:]], dim=2)
                k = torch.cat([k_p, k[:, :, n:]], dim=2)
            else:
                q, k = q_p, k_p
            return q, k
        qt = q.transpose(1, 2)
        kt = k.transpose(1, 2)
        q_p = self.rotary(qt[:, :n], self.patch_resolution)
        k_p = self.rotary(kt[:, :n], self.patch_resolution)
        if extra:
            qt = torch.cat([q_p, qt[:, n:]], dim=1)
            kt = torch.cat([k_p, kt[:, n:]], dim=1)
        else:
            qt, kt = q_p, k_p
        return qt.transpose(1, 2), kt.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention over tokens."""
        b, t, _ = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self._apply_rope(q, k)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(b, t, self.dim)
        return self.proj_drop(self.proj(x))


class KATTBlock(nn.Module):
    """Pre-RMSNorm block: MHSA + KAN MLP with DropPath residuals."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        kan_num_grids: int = 4,
        kan_grid_min: float = -2.0,
        kan_grid_max: float = 2.0,
        kan_compile: bool = True,
        use_base_update: bool = False,
        use_layernorm: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        rope_mode: str = "2dv1",
        rope_theta: float = 10000.0,
        num_patches: int = 196,
        patch_resolution=(14, 14),
    ):
        """Build one KATT encoder block."""
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} must be divisible by num_heads {num_heads}")

        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            rope_mode=rope_mode,
            rope_theta=rope_theta,
            num_patches=num_patches,
            patch_resolution=patch_resolution,
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = KAN(
            dim,
            mlp_hidden_dim,
            num_grids=kan_num_grids,
            grid_min=kan_grid_min,
            grid_max=kan_grid_max,
            drop=drop,
            use_base_update=use_base_update,
            use_layernorm=use_layernorm,
            use_compile=kan_compile,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Pre-RMSNorm MHSA then Pre-RMSNorm KAN with residuals."""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class KATT(nn.Module):
    """KATT backbone: MHSA + KAN."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        kan_num_grids: int = 4,
        kan_grid_min: float = -2.0,
        kan_grid_max: float = 2.0,
        kan_compile: bool = True,
        use_base_update: bool = False,
        use_layernorm: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_cls_token: bool = True,
        num_registers: int = 4,
        rope_mode: str = "2dv1",
        rope_theta: float = 10000.0,
        use_abs_pos: bool = True,
        **kwargs,
    ):
        """Build the full KATT vision backbone and classification head."""
        super().__init__()
        _ = kwargs  # ignore legacy alibi_normalize / layerscale_init / gate-lora flags
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if num_registers < 1:
            raise ValueError(f"num_registers must be >= 1, got {num_registers}")
        if rope_mode not in ("2dv0", "2dv1"):
            raise ValueError(f"rope_mode must be '2dv0' or '2dv1', got {rope_mode!r}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_cls_token = use_cls_token
        self.num_registers = int(num_registers)
        self.rope_mode = rope_mode
        self.use_abs_pos = bool(use_abs_pos)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        patch_resolution = (self.patch_embed.grid_size, self.patch_embed.grid_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_registers, embed_dim))

        num_extra = self.num_registers + (1 if use_cls_token else 0)
        num_tokens = num_patches + num_extra
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, num_tokens, embed_dim)) if self.use_abs_pos else None
        )
        self.pos_drop = nn.Dropout(drop_rate)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        dpr = [d.item() for d in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                KATTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    kan_num_grids=kan_num_grids,
                    kan_grid_min=kan_grid_min,
                    kan_grid_max=kan_grid_max,
                    kan_compile=kan_compile,
                    use_base_update=use_base_update,
                    use_layernorm=use_layernorm,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    rope_mode=rope_mode,
                    rope_theta=rope_theta,
                    num_patches=num_patches,
                    patch_resolution=patch_resolution,
                )
                for i in range(depth)
            ]
        )

        self.norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        """Initialize tokens, norms, and non-KAN linear layers."""
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        skip_linears: set[nn.Module] = set()
        for m in self.modules():
            if isinstance(m, KANLayer):
                skip_linears.add(m.dim_linear)
                if getattr(m, "grid_linear", None) is not None:
                    skip_linears.add(m.grid_linear)
                if getattr(m, "use_base_update", False) and hasattr(m, "base_linear"):
                    skip_linears.add(m.base_linear)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in skip_linears:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, RMSNorm)):
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Embed patches, add tokens/pos, and run all encoder blocks."""
        x = self.patch_embed(x)
        b = x.shape[0]
        x = torch.cat([x, self.register_tokens.expand(b, -1, -1)], dim=1)
        if self.cls_token is not None:
            x = torch.cat([x, self.cls_token.expand(b, -1, -1)], dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify an image with the KATT backbone."""
        x = self.forward_features(x)
        if self.cls_token is not None:
            pooled = x[:, -1]
        else:
            pooled = x[:, : self.num_patches].mean(dim=1)
        return self.head(pooled)


__all__ = [
    "MultiHeadSelfAttention",
    "KATTBlock",
    "KATT",
]


if __name__ == "__main__":
    from .build import KATT_VARIANTS, build_katt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_katt("KATT-S", num_classes=10).to(device)
    assert model.num_registers == 4
    assert isinstance(model.norm, RMSNorm)
    assert isinstance(model.blocks[0].norm1, RMSNorm)
    feats = model.forward_features(torch.randn(2, 3, 224, 224, device=device))
    assert feats.shape[1] == 196 + 4 + 1
    y = model(torch.randn(2, 3, 224, 224, device=device))
    print(y.shape, "RMSNorm + QK-RMSNorm + RoPE ok")
    for name, cfg in KATT_VARIANTS.items():
        m = build_katt(name, num_classes=10)
        assert m.num_heads == cfg["num_heads"]
        print(name, "ok")
