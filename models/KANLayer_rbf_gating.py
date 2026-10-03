from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .KANLayer import KANLayer


class KANLayerRBFGating(KANLayer):
    """KANLayer with a single-linear gated residual on the RBF output."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_layernorm: bool = True,
        base_activation=F.silu,
    ) -> None:
        """Construct a gated RBF-KAN layer."""
        # Keep base_update off; gating replaces that branch.
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_base_update=False,
            use_layernorm=use_layernorm,
            base_activation=base_activation,
        )
        self.gat_resid = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, use_layernorm: bool = True) -> torch.Tensor:
        """Apply RBF-KAN then add gated residual."""
        ret = super().forward(x, use_layernorm=use_layernorm)
        gat_resid = self.gat_resid(x)
        if gat_resid.dtype != ret.dtype:
            gat_resid = gat_resid.to(dtype=ret.dtype)
        return ret + gat_resid * torch.sigmoid(gat_resid)


__all__ = ["KANLayerRBFGating"]
