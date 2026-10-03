from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


def _as_fp32_grid(grid: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Cast grid/weight to contiguous float32 on device."""
    if grid.device != device or grid.dtype != torch.float32 or not grid.is_contiguous():
        return grid.detach().to(device=device, dtype=torch.float32).contiguous()
    return grid


def triton_available() -> bool:
    """Return True if Triton and CUDA are available."""
    return _HAS_TRITON and torch.cuda.is_available()


def rbf_basis_ref(x: torch.Tensor, grid: torch.Tensor, denominator: float) -> torch.Tensor:
    """Reference Gaussian RBF basis in float32."""
    return torch.exp(-((x.float().unsqueeze(-1) - grid.float()) / float(denominator)) ** 2)


def rbf_grid_linear_ref(
    x: torch.Tensor,
    grid: torch.Tensor,
    grid_weight: torch.Tensor,
    denominator: float,
) -> torch.Tensor:
    """Reference Linear(G->1) over RBF basis."""
    basis = rbf_basis_ref(x, grid, denominator)
    w = grid_weight.float().reshape(1, grid.numel())
    return torch.nn.functional.linear(basis, w).squeeze(-1)


if _HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 256}, num_warps=2),
            triton.Config({"BLOCK": 512}, num_warps=4),
            triton.Config({"BLOCK": 1024}, num_warps=4),
            triton.Config({"BLOCK": 2048}, num_warps=8),
            triton.Config({"BLOCK": 4096}, num_warps=8),
        ],
        key=["G"],
    )
    @triton.jit
    def _rbf_grid_lin_fwd_kernel(
        x_ptr,
        grid_ptr,
        w_ptr,
        out_ptr,
        n_elements,
        inv_d,
        G: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fused weighted RBF sum (Triton forward)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv = tl.full((), inv_d, dtype=tl.float32)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for g in range(G):
            c = tl.load(grid_ptr + g).to(tl.float32)
            w = tl.load(w_ptr + g).to(tl.float32)
            t = (x - c) * inv
            acc += w * tl.exp(-(t * t))
        tl.store(out_ptr + offs, acc, mask=mask)

    @triton.jit
    def _rbf_grid_lin_bwd_kernel(
        x_ptr,
        grid_ptr,
        w_ptr,
        dout_ptr,
        dx_ptr,
        dw_ptr,
        n_elements,
        inv_d,
        G: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fused weighted RBF gradients (Triton backward)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dout = tl.load(dout_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv = tl.full((), inv_d, dtype=tl.float32)
        inv2 = inv * inv
        jac = tl.zeros((BLOCK,), dtype=tl.float32)
        for g in range(G):
            c = tl.load(grid_ptr + g).to(tl.float32)
            w = tl.load(w_ptr + g).to(tl.float32)
            diff = x - c
            t = diff * inv
            rbf = tl.exp(-(t * t))
            jac += w * rbf * (-2.0 * diff * inv2)
            tl.atomic_add(dw_ptr + g, tl.sum(dout * rbf, axis=0))
        tl.store(dx_ptr + offs, dout * jac, mask=mask)


def _launch_grid_lin_fwd(
    x_flat: torch.Tensor,
    grid: torch.Tensor,
    weight: torch.Tensor,
    inv_d: float,
    out: torch.Tensor,
) -> None:
    """Launch fused weighted-RBF forward kernel."""
    n = x_flat.numel()
    G = int(grid.numel())
    grid_launch = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _rbf_grid_lin_fwd_kernel[grid_launch](
        x_flat, grid, weight, out, n, float(inv_d), G=G,
    )


def _launch_grid_lin_bwd(
    x_flat: torch.Tensor,
    grid: torch.Tensor,
    weight: torch.Tensor,
    dout_flat: torch.Tensor,
    inv_d: float,
    dx: torch.Tensor,
    dw: torch.Tensor,
) -> None:
    """Launch fused weighted-RBF backward kernel."""
    n = x_flat.numel()
    G = int(grid.numel())
    _rbf_grid_lin_bwd_kernel[(triton.cdiv(n, 1024),)](
        x_flat, grid, weight, dout_flat, dx, dw,
        n, float(inv_d), G=G, BLOCK=1024, num_warps=4,
    )


class _RBFGridLinearTritonFn(torch.autograd.Function):
    """Autograd bridge for fused weighted-RBF Triton kernels."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, grid: torch.Tensor, weight: torch.Tensor, denominator: float):
        """Run fused weighted-RBF forward and save tensors for backward."""
        assert x.is_cuda and grid.is_cuda and weight.is_cuda
        G = int(grid.numel())
        w = weight.reshape(-1).contiguous()
        if w.numel() != G:
            raise ValueError(f"grid Linear weight has {w.numel()} elems, grid has {G}")
        inv_d = 1.0 / float(denominator)
        x_flat = x.reshape(-1).contiguous()
        grid_c = _as_fp32_grid(grid, x.device)
        w_c = _as_fp32_grid(w, x.device)
        out_flat = torch.empty(x_flat.numel(), device=x.device, dtype=torch.float32)
        _launch_grid_lin_fwd(x_flat, grid_c, w_c, inv_d, out_flat)
        ctx.save_for_backward(x_flat, grid_c, w_c)
        ctx.inv_d = inv_d
        ctx.shape = x.shape
        ctx.weight_shape = tuple(weight.shape)
        ctx.x_dtype = x.dtype
        ctx.w_dtype = weight.dtype
        return out_flat.view_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Compute input and weight gradients for fused weighted-RBF."""
        x_flat, grid_c, w_c = ctx.saved_tensors
        dout = grad_out.reshape(-1).contiguous().float()
        dx_flat = torch.empty(x_flat.numel(), device=x_flat.device, dtype=torch.float32)
        dw = torch.zeros_like(w_c)
        _launch_grid_lin_bwd(x_flat, grid_c, w_c, dout, ctx.inv_d, dx_flat, dw)
        dx = dx_flat.view(ctx.shape).to(dtype=ctx.x_dtype)
        return dx, None, dw.view(ctx.weight_shape).to(dtype=ctx.w_dtype), None


def rbf_grid_linear_triton(
    x: torch.Tensor,
    grid: torch.Tensor,
    grid_weight: torch.Tensor,
    denominator: float,
) -> torch.Tensor:
    """Fused weighted RBF contraction used by KANLayer."""
    if not _HAS_TRITON or not x.is_cuda:
        raise RuntimeError("rbf_grid_linear_triton requires Triton + CUDA tensor.")
    if grid.device != x.device:
        grid = grid.to(device=x.device)
    if grid_weight.device != x.device:
        grid_weight = grid_weight.to(device=x.device)
    return _RBFGridLinearTritonFn.apply(x, grid, grid_weight, float(denominator))
