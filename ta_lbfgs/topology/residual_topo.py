"""
Axis 3 — Residual Stream Topology.

Probes inter-layer Jacobian coupling via output-norm ratios and classifies
each layer as 'coupled' (early layers with significant cross-layer Hessian
interaction) or 'block_diag' (late layers in the NTK lazy regime).

For LoRA adapters in the coupled zone a lightweight inter-layer buffer
captures cross-layer (s, y) pairs at rank-r × r cost.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor


class LoRAInterLayerBuffer:
    """Lightweight cross-layer curvature buffer for LoRA adapter pairs."""

    def __init__(self, max_pairs: int = 5) -> None:
        self.max_pairs = max_pairs
        self._pairs: List[Tuple[Tensor, Tensor]] = []

    def add(self, s: Tensor, y: Tensor) -> None:
        self._pairs.append((s.clone(), y.clone()))
        if len(self._pairs) > self.max_pairs:
            self._pairs.pop(0)

    def pairs(self) -> List[Tuple[Tensor, Tensor]]:
        return list(self._pairs)


class ResidualTopologyBuilder:
    """
    Cross-layer Jacobian norm estimator and coupling classifier.

    Estimates ‖∂x_{l'}/∂x_l‖ using output-norm ratios from a single
    forward pass and marks layer pairs whose coupling exceeds *threshold*
    as 'coupled'.  Early layers (l < n_layers / 3) typically fall in the
    coupled zone; later layers converge to a block-diagonal regime.

    Args:
        n_layers: Total number of transformer layers.
        threshold: Jacobian-norm ratio above which layers are coupled.
    """

    def __init__(self, n_layers: int, threshold: float = 0.05) -> None:
        self.n_layers = n_layers
        self.threshold = threshold
        # [n_layers, n_layers] estimated ‖∂x_{l'}/∂x_l‖
        self.jacobian_norms: Tensor = torch.zeros(n_layers, n_layers)
        # set of (l, l') pairs with significant coupling
        self.coupled_zone: Set[Tuple[int, int]] = set()
        # optional LoRA inter-layer buffers: {(l, lp): buffer}
        self._lora_buffers: Dict[Tuple[int, int], LoRAInterLayerBuffer] = {}

    # ------------------------------------------------------------------
    # Jacobian-norm probing
    # ------------------------------------------------------------------
    @torch.no_grad()
    def probe_jacobian_norms(
        self,
        model,
        x_sample: Optional[Tensor],
        layer_outputs: List[Tensor],
    ) -> None:
        """
        Estimate inter-layer Jacobian coupling from layer output norms.

        Uses output-norm ratios as a cheap proxy for ‖∂x_{l'}/∂x_l‖.
        Marks layer pairs as coupled when the ratio exceeds self.threshold.

        Args:
            model: The nn.Module (unused; reserved for future hook-based probing).
            x_sample: A representative input sample (unused in norm-ratio mode).
            layer_outputs: List of length n_layers with each layer's output tensor.
        """
        n = min(self.n_layers, len(layer_outputs))
        for l in range(n):
            norm_l = layer_outputs[l].norm().clamp(min=1e-8).item()
            for lp in range(l + 1, n):
                norm_lp = layer_outputs[lp].norm().clamp(min=1e-8).item()
                ratio = norm_lp / norm_l
                self.jacobian_norms[l, lp] = ratio
                if ratio > self.threshold:
                    self.coupled_zone.add((l, lp))
                else:
                    self.coupled_zone.discard((l, lp))

    # ------------------------------------------------------------------
    # Coupling query
    # ------------------------------------------------------------------
    def is_coupled(self, l: int, lp: int) -> bool:
        """Return True if layers l and l' are in the coupled zone."""
        return (l, lp) in self.coupled_zone

    def hessian_strategy(self, l: int) -> str:
        """
        Return the Hessian approximation strategy for layer l.

        Early layers (l < n_layers / 3) use a banded inter-layer
        approximation; later layers use standard block-diagonal treatment.
        """
        if l < self.n_layers // 3:
            return "coupled"
        return "block_diag"

    # ------------------------------------------------------------------
    # LoRA inter-layer buffers
    # ------------------------------------------------------------------
    def get_lora_buffer(self, l: int, lp: int) -> LoRAInterLayerBuffer:
        """Return (creating if needed) the LoRA inter-layer buffer for (l, lp)."""
        key = (l, lp)
        if key not in self._lora_buffers:
            self._lora_buffers[key] = LoRAInterLayerBuffer()
        return self._lora_buffers[key]
