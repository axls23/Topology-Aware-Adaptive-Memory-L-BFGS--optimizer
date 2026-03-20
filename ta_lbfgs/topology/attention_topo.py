"""
Axis 1 — Attention Interaction Topology.

Derives per-head topology masks from secant pair accumulation and
classifies each head into one of four empirically observed types:
local (sliding window), global, causal (triangular), or sink (BOS-dominated).
Each type implies a distinct Hessian block strategy used by the outer loop.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


_HEAD_TYPES = ("local", "global", "causal", "sink")
_HESSIAN_STRATEGY = {"local": "banded", "sink": "kfac", "global": "kfac", "causal": "kfac"}


class AttentionTopologyBuilder:
    """
    Per-head topology mask builder for multi-head attention layers.

    Accumulates secant pairs (s, y) per (layer, head, projection) triple,
    derives binary masks after a warm-up window, and classifies each head
    into a type that determines the Hessian approximation strategy.

    Args:
        model: The nn.Module being optimized (may be None for standalone use).
        window_size: Local attention window radius used for head classification.
        warmup_steps: Number of outer steps before first mask derivation.
    """

    HEAD_TYPES = _HEAD_TYPES

    def __init__(
        self,
        model,
        window_size: int = 128,
        warmup_steps: int = 50,
    ) -> None:
        self.model = model
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        # {(layer_idx, head_idx, proj): C_secant}
        self.secant_accum: Dict[Tuple[int, int, str], Tensor] = {}
        # {(layer_idx, head_idx, proj): M_h  (bool mask)}
        self.masks: Dict[Tuple[int, int, str], Tensor] = {}
        # {(layer_idx, head_idx): head_type str}
        self.head_type: Dict[Tuple[int, int], str] = {}
        self._outer_step: int = 0
        self._val_loss_history: list = []

    # ------------------------------------------------------------------
    # Secant accumulation (EMA outer product of |s| * |y|)
    # ------------------------------------------------------------------
    def accumulate_secant(
        self, layer_idx: int, head_idx: int, proj: str, s: Tensor, y: Tensor
    ) -> None:
        """Update the running secant accumulation matrix for one head projection."""
        key = (layer_idx, head_idx, proj)
        outer = (s.unsqueeze(1) * y.unsqueeze(0)).abs()
        if key not in self.secant_accum:
            self.secant_accum[key] = outer.clone()
        else:
            self.secant_accum[key] = 0.9 * self.secant_accum[key] + 0.1 * outer

    # ------------------------------------------------------------------
    # Mask derivation
    # ------------------------------------------------------------------
    def derive_mask(
        self, layer_idx: int, head_idx: int, proj: str, threshold: float = 0.01
    ) -> Optional[Tensor]:
        """Derive a binary topology mask from the accumulated secant matrix."""
        key = (layer_idx, head_idx, proj)
        C = self.secant_accum.get(key)
        if C is None:
            return None
        M = (C / C.max().clamp(min=1e-8)) > threshold
        self.masks[key] = M
        return M

    # ------------------------------------------------------------------
    # Head classification
    # ------------------------------------------------------------------
    def classify_head(
        self, layer_idx: int, head_idx: int, attn_weights: Tensor
    ) -> str:
        """
        Classify an attention head from its average attention weight matrix.

        Args:
            layer_idx: Layer index.
            head_idx: Head index within the layer.
            attn_weights: (seq, seq) average attention probability matrix.

        Returns:
            One of 'local', 'global', 'causal', or 'sink'.
        """
        T = attn_weights.size(0)
        bos_mass = float(attn_weights[:, 0].mean().item())

        if bos_mass > 0.5:
            head_type = "sink"
        else:
            # Sum of all attention weight within ±window_size diagonals
            local_mass = 0.0
            for offset in range(-self.window_size, self.window_size + 1):
                if abs(offset) >= T:
                    continue
                diag = attn_weights.diagonal(offset)
                if diag.numel() > 0:
                    local_mass += float(diag.sum().item())

            total_mass = float(attn_weights.sum().item())
            if total_mass > 0 and local_mass / total_mass > 0.8:
                head_type = "local"
            else:
                head_type = "global"

        self.head_type[(layer_idx, head_idx)] = head_type
        return head_type

    # ------------------------------------------------------------------
    # Hessian strategy dispatch
    # ------------------------------------------------------------------
    def hessian_strategy(self, layer_idx: int, head_idx: int) -> str:
        """Return the Hessian approximation strategy for the given head."""
        ht = self.head_type.get((layer_idx, head_idx), "global")
        return _HESSIAN_STRATEGY.get(ht, "kfac")

    # ------------------------------------------------------------------
    # Topology re-derive scheduling
    # ------------------------------------------------------------------
    def should_rederive(self, val_loss: Optional[float] = None) -> bool:
        """
        Return True if masks should be re-derived.

        Triggers at warm-up completion (step == warmup_steps) and every
        100 steps thereafter, plus an emergency re-derive if validation
        loss variance exceeds 2-sigma of its rolling mean.
        """
        self._outer_step += 1
        if self._outer_step == self.warmup_steps:
            return True
        if self._outer_step > self.warmup_steps and (self._outer_step % 100) == 0:
            return True

        if val_loss is not None:
            self._val_loss_history.append(val_loss)
            if len(self._val_loss_history) > 20:
                self._val_loss_history.pop(0)
            if len(self._val_loss_history) >= 5:
                vals = self._val_loss_history
                mu = sum(vals) / len(vals)
                sigma = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
                if abs(val_loss - mu) > 2.0 * sigma and sigma > 0:
                    return True
        return False
