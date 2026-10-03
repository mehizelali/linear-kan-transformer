from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Mlp, PatchEmbed

try:  # optional; variants set flash=False so this is never required
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:  # pragma: no cover
    flash_attn_qkvpacked_func = None


def broadcat(freqss, dim = -1):
    """Broadcast and concatenate frequency tensors along dim."""
    num_freqss = len(freqss)
    shape_lens = set(list(map(lambda t: len(t.shape), freqss)))
    assert len(shape_lens) == 1, 'freqss must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), freqss)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_freqss), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    freqss = list(map(lambda t: t[0].expand(*t[1]), zip(freqss, expandable_shapes)))
    return torch.cat(freqss, dim = dim)

def rotate_half(x):
    """Rotate pairs of features for RoPE."""
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

class VisionRotaryEmbedding(nn.Module):
    """2D rotary embedding for vision tokens."""

    def __init__(
        self,
        dim,
        pt_seq_len=14,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        """Build rotary frequency buffers for a patch grid."""
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.pt_seq_len=pt_seq_len
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        """Apply 2D RoPE to a token sequence."""
        ft_seq_len = int(np.sqrt(x.shape[1]))
        t = torch.arange(ft_seq_len, device=x.device, dtype=x.dtype) / ft_seq_len * self.pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, self.freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2) # 14*32
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1) # 14*14*64

        freqs_cos = freqs.cos().view(-1, 1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, 1, freqs.shape[-1])
        return  x * freqs_cos + rotate_half(x) * freqs_sin

def rotate_freqs(freqs, angle_deg):
    """Rotate RoPE frequency grid by an angle in degrees."""
    assert freqs.ndim == 4 and freqs.shape[0] == freqs.shape[1], "Input must have shape (n, n, d1, d2)"
    n, _, d1, d2 = freqs.shape
    freq_type = freqs.dtype
    angle_rad = math.radians(angle_deg)

    # Reshape from (n, n, d1, d2) → (n, n, d1 * d2)
    freqs = freqs.reshape(n, n, -1)

    # Permute to (1, C, H, W) where C = d1 * d2
    freqs = freqs.permute(2, 0, 1).unsqueeze(0)

    # Rotation matrix (2x3)
    theta = torch.tensor([
        [ math.cos(angle_rad), -math.sin(angle_rad), 0.0],
        [ math.sin(angle_rad),  math.cos(angle_rad), 0.0]
    ], dtype=torch.float32, device=freqs.device).unsqueeze(0)

    freqs = freqs.to(torch.float32)

    # Build sampling grid
    grid = F.affine_grid(theta, freqs.size(), align_corners=True)

    # Rotate using bilinear interpolation, with border padding
    rotated = F.grid_sample(freqs, grid, mode='bilinear', padding_mode='border', align_corners=True)

    # Convert back: (1, C, H, W) → (H, W, C)
    rotated = rotated.squeeze(0).permute(1, 2, 0).to(freq_type)

    # Reshape back to (n, n, d1, d2)
    return rotated.reshape(n, n, d1, d2)


class Attention(nn.Module):
    """Multi-head attention with optional RoPE, QK-norm, and flash/SDPA."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., flash=True, sdpa=False,
                 rope_size=0, rope_reg_size=0, num_registers=0, reg_theta=10000, qk_norm=False):
        """Build attention projections and optional RoPE modules."""
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.flash = flash
        self.num_registers = num_registers
        self.rope = VisionRotaryEmbedding(head_dim//2, rope_size) if rope_size > 0 else None
        self.rope_reg = VisionRotaryEmbedding(head_dim//2, rope_reg_size, theta=reg_theta) if rope_reg_size > 0 else None

        self.qk_norm = qk_norm
        self.sdpa = sdpa
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(self, x):
        """Compute multi-head attention over tokens."""
        B, N, C = x.shape
        reg_idx = N - self.num_registers

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)

        if self.qk_norm:
            qk_dtype = q.dtype
            q = self.q_norm(q).to(qk_dtype)
            k = self.k_norm(k).to(qk_dtype)

        if self.rope is not None:
            q = torch.cat((q[:, :1], self.rope(q[:, 1: reg_idx]), q[:, reg_idx:]), dim=1)
            k = torch.cat((k[:, :1], self.rope(k[:, 1: reg_idx]), k[:, reg_idx:]), dim=1)
        if self.rope_reg is not None:
            q = torch.cat((q[:, :1], q[:, 1: reg_idx], self.rope_reg(q[:, reg_idx:])), dim=1)
            k = torch.cat((k[:, :1], k[:, 1: reg_idx], self.rope_reg(k[:, reg_idx:])), dim=1)
        
        if self.flash:
            qkv = torch.stack([q, k, v], dim=2)
            x = flash_attn_qkvpacked_func(qkv).reshape(B, N, C)
        elif getattr(self, "sdpa", False):
            # Mathematically identical to the branch below, but PyTorch's fused
            # kernel never materializes the B x H x N x N attention matrix.
            # Upstream keeps that explicit matmul, which OOMs at large batch.
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
            ).transpose(1, 2).reshape(B, N, C)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RMSNorm(nn.Module):
    """Root-mean-square layer normalization."""

    def __init__(self, hidden_size, eps=1e-6):
        """Create an RMSNorm with learnable scale."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """Normalize hidden states with RMS and apply scale."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class SwiGLU(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False,):
        """Build a SwiGLU MLP."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Apply SwiGLU feed-forward."""
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    """Transformer block with optional LayerScale."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp, init_values=1e-4,
                 flash=True, rope_size=0, rope_reg_size=0, reg_theta=10000, num_registers=0, qk_norm=False, layer_scale=True, sdpa=False):
        """Build attention + MLP with residual DropPath."""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, flash=flash,
            rope_size=rope_size, rope_reg_size=rope_reg_size, num_registers=num_registers, qk_norm=qk_norm, reg_theta=reg_theta, sdpa=sdpa)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        """Apply Pre-Norm attention and MLP with residuals."""
        if self.layer_scale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x        
    
class vit_models(nn.Module):
    """Vision Transformer with LayerScale (ViT-5)."""
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, ape=True,
                 block_layers=Block, Patch_layer=PatchEmbed, act_layer=nn.GELU, Attention_block=Attention, Mlp_block=Mlp,
                 init_scale=1e-4, flash=True, rope=False, num_registers=0, qk_norm=False, reg_theta=10000, layer_scale=True, sdpa=False, **kwargs):
        """Build a ViT-5 style vision transformer."""
        super().__init__()       
        self.dropout_rate = drop_rate  
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_registers = num_registers

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, num_registers, embed_dim)) if num_registers > 0 else None

        rope_reg_size = int(num_registers ** 0.5)
        assert rope_reg_size ** 2 == num_registers, "num_registers must be a square number"

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) if ape else None

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale,
                flash=flash, rope_size=img_size // patch_size if rope else 0,
                rope_reg_size=rope_reg_size, num_registers=num_registers,
                qk_norm=qk_norm, reg_theta=reg_theta, layer_scale=layer_scale, sdpa=sdpa)
            for i in range(depth)])
           
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        
        trunc_normal_(self.cls_token, std=.02)
        if ape:
            trunc_normal_(self.pos_embed, std=.02)
        if num_registers > 0:
            trunc_normal_(self.reg_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize Linear and LayerNorm weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Parameter names excluded from weight decay."""
        return {'pos_embed', 'cls_token', 'reg_token'}

    def get_classifier(self):
        """Return the classification head module."""
        return self.head
    
    def get_num_layers(self):
        """Return the number of transformer blocks."""
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        """Replace the classification head for a new num_classes."""
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """Embed patches and return the CLS token features."""
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        registers = self.reg_token.expand(B, -1, -1) if self.reg_token is not None else None
        
        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
        if registers is not None:
            x = torch.cat((x, registers), dim=1)
            
        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        """Classify an image with ViT-5."""
        x = self.forward_features(x)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        return x

def deit_small_patch16_LS(img_size=224, **kwargs):
    """Build DeiT-Small with LayerScale."""
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, flash=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block, **kwargs)
    return model

def deit_base_patch16_LS(img_size=224, **kwargs):
    """Build DeiT-Base with LayerScale."""
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, flash=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block, **kwargs)
    return model

def deit_large_patch16_LS(img_size=224, **kwargs):
    """Build DeiT-Large with LayerScale."""
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, flash=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block, **kwargs)
    return model

def vit5_small(img_size=224, **kwargs):
    """Build ViT-5 Small."""
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=False,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Block, rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

def vit5_base(img_size=224, **kwargs):
    """Build ViT-5 Base."""
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=False,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Block, rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

def vit5_large(img_size=224, **kwargs):
    """Build ViT-5 Large."""
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=False,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Block, rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

def vit5_xlarge(img_size=224, **kwargs):
    """Build ViT-5 XLarge."""
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=1152, depth=28, num_heads=16, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=False,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Block, rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

# Named variants for this repo's T/S/B convention.
VIT5_VARIANTS: dict[str, dict] = {
    "ViT5-T": dict(embed_dim=192, depth=12, num_heads=3),
    "ViT5-S": dict(embed_dim=384, depth=12, num_heads=6),
    "ViT5-B": dict(embed_dim=768, depth=12, num_heads=12),
    # matches upstream vit5_large (1024/24/16) and the LKAT-L / Mixer-L slot
    "ViT5-L": dict(embed_dim=1024, depth=24, num_heads=16),
}

# Flags shared by every upstream vit5_* model (see models_vit5.py).
_VIT5_COMMON = dict(
    patch_size=16,
    mlp_ratio=4,
    qkv_bias=False,
    num_registers=4,
    flash=False,
    rope=True,
    rope_reg=True,
    reg_theta=100,
    qk_norm=True,
)


def build_vit5(
    name: str = "ViT5-B",
    num_classes: int = 100,
    image_size: int = 224,
    **overrides,
):
    """Build a ViT-5 variant by name."""
    if name not in VIT5_VARIANTS:
        raise ValueError(f"Unknown ViT-5 variant '{name}'. Choose from {list(VIT5_VARIANTS)}")
    cfg = {**_VIT5_COMMON, **VIT5_VARIANTS[name], **overrides}
    return vit_models(
        img_size=image_size,
        num_classes=num_classes,
        norm_layer=partial(RMSNorm, eps=1e-6),
        block_layers=Block,
        **cfg,
    )


__all__ = [
    "vit_models",
    "Block",
    "Attention",
    "RMSNorm",
    "SwiGLU",
    "VIT5_VARIANTS",
    "build_vit5",
    "vit5_small",
    "vit5_base",
    "vit5_large",
    "vit5_xlarge",
]
