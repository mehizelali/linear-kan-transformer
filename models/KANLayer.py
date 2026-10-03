from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_rbf import rbf_grid_linear_triton, triton_available


class RadialBasisFunction(nn.Module):
    """Fixed Gaussian RBF grid centers and width (non-trainable)."""

    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        denominator: float | None = None,
    ):
        """Build evenly spaced RBF centers."""
        super().__init__()
        if not triton_available():
            raise RuntimeError("Triton + CUDA required for RBF KAN (only backend).")
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / max(num_grids - 1, 1)


class KANLayer(nn.Module):
    """Single RBF-KAN layer: grid mix then channel projection."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = False,
        use_layernorm: bool = True,
        base_activation=F.silu,
    ) -> None:
        """Construct one KAN layer."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grids = num_grids
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)

        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.grid_linear = nn.Linear(num_grids, 1, bias=False)
        self.dim_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.reset_parameters()
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def reset_parameters(self) -> None:
        """Initialize grid_linear and dim_linear weights."""
        scale = 1.0 / (self.num_grids * self.input_dim)
        nn.init.trunc_normal_(self.grid_linear.weight, mean=0, std=scale)
        nn.init.trunc_normal_(self.dim_linear.weight, mean=0, std=scale)

    def _rbf(self, x_n: torch.Tensor) -> torch.Tensor:
        """Fused weighted RBF contraction via Triton."""
        return rbf_grid_linear_triton(
            x_n, self.rbf.grid, self.grid_linear.weight, self.rbf.denominator
        )

    def forward(self, x: torch.Tensor, use_layernorm: bool = True) -> torch.Tensor:
        """Apply RBF-KAN mapping from input_dim to output_dim."""
        x_n = self.layernorm(x) if (self.layernorm is not None and use_layernorm) else x
        reduced = self._rbf(x_n)
        w = self.dim_linear.weight
        if reduced.dtype != w.dtype:
            reduced = reduced.to(dtype=w.dtype)
        ret = self.dim_linear(reduced)
        if ret.dtype != x_n.dtype:
            ret = ret.to(dtype=x_n.dtype)
        if self.use_base_update:
            ret = ret + self.base_linear(self.base_activation(x))
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2,
    ):
        """Return the learned univariate RBF curve for one (input, output) edge."""
        ng = self.rbf.num_grids
        h = float(self.rbf.denominator)
        if not (0 <= input_index < self.input_dim):
            raise IndexError(f"input_index {input_index} not in [0, {self.input_dim})")
        if not (0 <= output_index < self.output_dim):
            raise IndexError(f"output_index {output_index} not in [0, {self.output_dim})")

        # Effective FastKAN-style spline weights for this edge: [G]
        w_grid = self.grid_linear.weight.detach().reshape(-1)  # [G]
        w_dim = self.dim_linear.weight.detach()[output_index, input_index]
        w = (w_dim * w_grid).to(dtype=torch.float32)

        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts,
            dtype=torch.float32,
        )
        grid = self.rbf.grid.detach().to(dtype=torch.float32)
        with torch.no_grad():
            # φ: [num_pts, G]
            basis = torch.exp(-((x[:, None] - grid[None, :]) / h) ** 2)
            y = (w * basis).sum(-1)
        return x, y


__all__ = ["KANLayer", "RadialBasisFunction"]
