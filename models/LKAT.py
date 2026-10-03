from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from fla.layers import GatedLinearAttention
    from fla.layers.utils import get_layer_cache, get_unpad_data, index_first_axis, pad_input, update_layer_cache
    from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "LKAT requires flash-linear-attention (import name: fla). "
        "Install with:  uv pip install flash-linear-attention"
    ) from e

from .KANLayer import KANLayer


def _to_2tuple(x):
    """Convert a value to an (h, w) int tuple."""
    if isinstance(x, (tuple, list)):
        return (int(x[0]), int(x[1]))
    return (int(x), int(x))


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape rotary freqs so they broadcast against x."""
    ndim = x.ndim
    if freqs_cis.shape[-1] == x.shape[-1]:
        shape = [1 if i in (0, 2) else d for i, d in enumerate(x.shape)]
    else:
        shape = [d if i != 0 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Create an RMSNorm with learnable scale."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x with RMS and apply the scale weight."""
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


class RotaryEmbeddingFast2D(nn.Module):
    """2D rotary position embedding for patch tokens."""

    def __init__(self, embed_dims, patch_resolution, theta=10000.0, base_size=14):
        """Build 2D RoPE for a patch grid."""
        super().__init__()
        self.dim = embed_dims
        self.patch_resolution = _to_2tuple(patch_resolution)
        self.theta = theta
        self.base_size = base_size
        h, w = self.patch_resolution
        self.scale_h = base_size / h
        self.scale_w = base_size / w
        self.freqs_cis = None

    def compute_position_embedding(self, step: int = 1, bias=0.0):
        """Compute complex 2D rotary frequencies for the patch grid."""
        h, w = self.patch_resolution
        end = h * w
        flat = torch.arange(0, end)
        x_pos = bias + (flat % w) * step
        y_pos = bias + (flat // w) * step
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 4)[: self.dim // 4].float() / self.dim))
        x_freqs = torch.outer(self.scale_w * x_pos, freqs).float()
        y_freqs = torch.outer(self.scale_h * y_pos, freqs).float()
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
        return freqs_cis.reshape(end, -1)

    def apply_rotary_emb_single(self, xq: torch.Tensor, freqs_cis: torch.Tensor):
        """Apply complex rotary embedding to one tensor."""
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
        out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        return out.type_as(xq)

    def forward(self, x, patch_resolution):
        """Apply 2D RoPE, rebuilding freqs if the grid size changed."""
        patch_resolution = _to_2tuple(patch_resolution)
        if patch_resolution != self.patch_resolution or self.freqs_cis is None:
            self.patch_resolution = patch_resolution
            self.scale_h = self.base_size / patch_resolution[0]
            self.scale_w = self.base_size / patch_resolution[1]
            self.freqs_cis = self.compute_position_embedding().to(device=x.device, dtype=torch.complex64)
        return self.apply_rotary_emb_single(x, self.freqs_cis)


class RotaryEmbeddingFast(nn.Module):
    """2D rotary embedding using cos/sin buffers (v0 mode)."""

    def __init__(self, embed_dims, patch_resolution, theta=10000.0):
        """Build cos/sin RoPE buffers for a patch grid."""
        super().__init__()
        self.half_dim = embed_dims // 2
        self.patch_resolution = _to_2tuple(patch_resolution)
        self.theta = theta
        freqs_cos, freqs_sin = self.compute_position_embedding()
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def compute_position_embedding(self):
        """Compute cos/sin position tables for height and width."""
        frequency = 1.0 / (self.theta ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        h, w = self.patch_resolution
        th = torch.arange(h) / h * self.half_dim
        tw = torch.arange(w) / w * self.half_dim
        position_h = (th[:, None] @ frequency[None, :]).repeat(1, 2)
        position_w = (tw[:, None] @ frequency[None, :]).repeat(1, 2)
        height = position_h[:, None, :].expand(h, w, self.half_dim)
        width = position_w[None, :, :].expand(h, w, self.half_dim)
        position = torch.cat((height, width), dim=-1)
        return position.cos().view(-1, position.shape[-1]), position.sin().view(-1, position.shape[-1])

    def forward(self, x, patch_resolution):
        """Apply cos/sin RoPE, rebuilding buffers if the grid size changed."""
        patch_resolution = _to_2tuple(patch_resolution)
        if patch_resolution != self.patch_resolution:
            self.patch_resolution = patch_resolution
            freqs_cos, freqs_sin = self.compute_position_embedding()
            self.register_buffer("freqs_cos", freqs_cos.to(device=x.device, dtype=x.dtype))
            self.register_buffer("freqs_sin", freqs_sin.to(device=x.device, dtype=x.dtype))
        inputs = x
        x = x.reshape(*x.shape[:3], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1).reshape_as(inputs)
        return inputs * self.freqs_cos + x * self.freqs_sin


class DropPath(nn.Module):
    """Stochastic depth (per-sample path drop)."""

    def __init__(self, drop_prob: float = 0.0):
        """Create a DropPath with the given drop probability."""
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Drop residual paths during training."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed(nn.Module):
    """Convolutional patch embedding for images."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
    ):
        """Create a non-overlapping patch projection."""
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """Project an image to a sequence of patch tokens."""
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class KAN(nn.Module):
    """Two-layer RBF-KAN MLP used as the block feed-forward."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_grids: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        drop: float = 0.0,
        use_base_update: bool = False,
        use_layernorm: bool = True,
        use_compile: bool = True,
    ):
        """Build a two-layer Triton RBF-KAN MLP."""
        super().__init__()
        self.use_compile = use_compile
        self._compile_key = ("KAN", id(self))
        self.fc1 = KANLayer(
            dim,
            hidden_dim,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_layernorm=use_layernorm,
            use_base_update=use_base_update,
        )
        self.fc2 = KANLayer(
            hidden_dim,
            dim,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_layernorm=use_layernorm,
            use_base_update=use_base_update,
        )
        self.drop = nn.Dropout(drop)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Run fc1 → dropout → fc2 → dropout."""
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two-layer KAN MLP."""
        return self._forward_impl(x)


class GatedLinearAttentionViT(GatedLinearAttention):
    """GLA attention with QK-RMSNorm and 2D RoPE for vision tokens."""

    def __init__(
        self,
        *args,
        rope_mode: str = "2dv1",
        rope_theta: float = 10000.0,
        num_patches: int = 196,
        patch_resolution=(14, 14),
        num_extra: int = 0,
        qk_norm_eps: float = 1e-6,
        **kwargs,
    ):
        """Wrap FLA GLA with vision RoPE and QK norms."""
        kwargs.setdefault("fuse_norm", False)
        super().__init__(*args, **kwargs)
        if rope_mode not in ("2dv0", "2dv1"):
            raise ValueError(f"rope_mode must be '2dv0' or '2dv1', got {rope_mode!r}")
        self.rope_mode = rope_mode
        self.num_patches = int(num_patches)
        self.num_extra = int(num_extra)
        self.patch_resolution = _to_2tuple(patch_resolution)

        self.q_norm = RMSNorm(self.head_k_dim, eps=qk_norm_eps)
        self.k_norm = RMSNorm(self.head_k_dim, eps=qk_norm_eps)

        self.fuse_norm_and_gate = False
        if hasattr(self, "g_norm_swish_gate"):
            del self.g_norm_swish_gate
        self.g_norm = nn.LayerNorm(self.head_v_dim)

        if rope_mode == "2dv0":
            self.rotary = RotaryEmbeddingFast(
                embed_dims=self.head_k_dim,
                patch_resolution=self.patch_resolution,
                theta=rope_theta,
            )
        else:
            self.rotary = RotaryEmbeddingFast2D(
                embed_dims=self.head_k_dim,
                patch_resolution=self.patch_resolution,
                theta=rope_theta,
            )

    def _apply_rope(self, q, k):
        """Apply 2D RoPE to patch tokens; leave extras unchanged."""
        n = self.num_patches
        extra = q.shape[1] - n
        if extra < 0:
            raise RuntimeError(f"seq_len {q.shape[1]} < num_patches {n}")
        if self.rope_mode == "2dv0":
            qh = q.transpose(1, 2)
            kh = k.transpose(1, 2)
            q_p = self.rotary(qh[:, :, :n], self.patch_resolution)
            k_p = self.rotary(kh[:, :, :n], self.patch_resolution)
            if extra:
                qh = torch.cat([q_p, qh[:, :, n:]], dim=2)
                kh = torch.cat([k_p, kh[:, :, n:]], dim=2)
            else:
                qh, kh = q_p, k_p
            return qh.transpose(1, 2), kh.transpose(1, 2)
        q_p = self.rotary(q[:, :n], self.patch_resolution)
        k_p = self.rotary(k[:, :n], self.patch_resolution)
        if extra:
            q = torch.cat([q_p, q[:, n:]], dim=1)
            k = torch.cat([k_p, k[:, n:]], dim=1)
        else:
            q, k = q_p, k_p
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        """Run gated linear attention over vision tokens."""
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if hidden_states.shape[1] <= 64 else self.mode

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is None and attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            conv_state_q = conv_state_k = conv_state_v = None
        gk = self.gk_proj(hidden_states)

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        if self.num_kv_groups > 1:
            k, gk = (repeat(x, "... (h d) -> ... (h g) d", g=self.num_kv_groups, d=self.head_k_dim) for x in (k, gk))
            v = repeat(v, "... (h d) -> ... (h g) d", g=self.num_kv_groups, d=self.head_v_dim)
        else:
            k, gk = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (k, gk))
            v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self._apply_rope(q, k)

        gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        if self.feature_map_fn is not None:
            q, k = map(self.feature_map_fn, (q, k))

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gla(
                q=q, k=k, v=v, gk=gk,
                initial_state=recurrent_state, output_final_state=use_cache,
                state_v_first=True, cu_seqlens=cu_seqlens,
            )
        elif mode == "fused_chunk":
            o, recurrent_state = fused_chunk_gla(
                q=q, k=k, v=v, g=gk,
                initial_state=recurrent_state, output_final_state=use_cache,
            )
        elif mode == "chunk":
            o, recurrent_state = chunk_gla(
                q=q, k=k, v=v, g=gk,
                initial_state=recurrent_state, output_final_state=use_cache,
                state_v_first=True, cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        update_layer_cache(
            self, past_key_values, recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, "... (h d) -> ... h d", d=self.head_v_dim)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, "... h d -> ... (h d)")
            else:
                o = rearrange(self.g_norm(o), "... h d -> ... (h d)")
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), "... h d -> ... (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values


GatedLinearAttentionRoPE = GatedLinearAttentionViT


class LKATBlock(nn.Module):
    """Pre-LN block with GLA attention and Triton RBF-KAN MLP."""

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
        use_layernorm: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        layer_idx: int | None = None,
        rope_mode: str = "2dv1",
        rope_theta: float = 10000.0,
        num_patches: int = 196,
        patch_resolution=(14, 14),
        num_extra: int = 0,
    ):
        """Build one LKAT encoder block."""
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} must be divisible by num_heads {num_heads}")

        self.attn = GatedLinearAttentionViT(
            hidden_size=dim,
            num_heads=num_heads,
            expand_k=expand_k,
            expand_v=expand_v,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            mode="fused_recurrent",
            layer_idx=layer_idx,
            rope_mode=rope_mode,
            rope_theta=rope_theta,
            num_patches=num_patches,
            patch_resolution=patch_resolution,
            num_extra=num_extra,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(drop)
        # Encoder Pre-LN: must be LayerNorm (not RMSNorm).
        self.norm1 = nn.LayerNorm(dim)

        # Triton RBF KAN only (KANLayer → models/triton_rbf.py).
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
        self.norm2 = nn.LayerNorm(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """Apply Pre-LN GLA then Pre-LN KAN with residual DropPath."""
        # Pre-LN (LayerNorm) → attn / KAN, then residual.
        attn_out, *_ = self.attn(self.norm1(x))
        attn_out = self.proj_drop(self.attn_drop(attn_out))
        x = x + self.drop_path(attn_out)

        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_out)
        return x


class LKAT(nn.Module):
    """LKAT backbone: GLA + Triton RBF KAN (Pre-LN LayerNorm encoder)."""

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
        kan_type: str = "rbf",
        kan_grid_min: float = -2.0,
        kan_grid_max: float = 2.0,
        kan_compile: bool = True,
        use_base_update: bool = False,
        use_layernorm: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_cls_token: bool = True,
        num_registers: int = 4,
        rope_mode: str = "2dv1",
        rope_theta: float = 10000.0,
        use_abs_pos: bool = True,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        **kwargs,
    ):
        """Build the full LKAT vision backbone and classification head."""
        super().__init__()
        _ = kwargs
        if str(kan_type).lower() not in ("rbf", "triton", "triton_rbf"):
            raise ValueError(
                f"LKAT only supports Triton RBF KAN (kan_type='rbf'), got {kan_type!r}"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}) "
                f"as in ViT"
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

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_registers, embed_dim))

        num_extra = self.num_registers + (1 if use_cls_token else 0)
        num_tokens = num_patches + num_extra

        if self.use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(drop_rate)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        dpr = [d.item() for d in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                LKATBlock(
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
                    expand_k=expand_k,
                    expand_v=expand_v,
                    use_short_conv=use_short_conv,
                    conv_size=conv_size,
                    layer_idx=i,
                    rope_mode=rope_mode,
                    rope_theta=rope_theta,
                    num_patches=num_patches,
                    patch_resolution=patch_resolution,
                    num_extra=num_extra,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
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
            if isinstance(m, GatedLinearAttention):
                for sm in m.modules():
                    if isinstance(sm, nn.Linear):
                        skip_linears.add(sm)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in skip_linears:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, RMSNorm)):
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward_features(self, x):
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
        x = self.norm(x)
        return x

    def forward(self, x):
        """Classify an image with the LKAT backbone."""
        x = self.forward_features(x)
        if self.cls_token is not None:
            pooled = x[:, -1]
        else:
            pooled = x[:, : self.num_patches].mean(dim=1)
        return self.head(pooled)

