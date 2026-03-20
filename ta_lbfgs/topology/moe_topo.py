"""
Axis 2 — Expert Routing Topology.

Maintains per-expert (s, y) curvature buffers, load-proportional window
sizing, TTL-gated expiry for dormant experts, and a co-activation matrix
that tracks which expert pairs are frequently activated together.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor


class MoETopologyBuilder:
    """
    Per-expert topology and curvature buffer manager for MoE models.

    Each expert maintains an independent L-BFGS (s,y) buffer whose window
    size scales with the expert's activation frequency.  Dormant experts
    (TTL exceeded) have their buffers purged to prevent stale curvature.

    Args:
        n_experts: Total number of experts in the MoE layer.
        top_k: Number of experts activated per token.
        m_max: Maximum curvature buffer window size.
        ttl_expire: Steps of inactivity after which a buffer is purged.
    """

    def __init__(
        self,
        n_experts: int,
        top_k: int,
        m_max: int = 20,
        ttl_expire: int = 50,
    ) -> None:
        self.n_experts = n_experts
        self.top_k = top_k
        self.m_max = m_max
        self.ttl_expire = ttl_expire
        # steps since last activation (0 = active this step)
        self.ttl: Dict[int, int] = {e: 0 for e in range(n_experts)}
        # rolling exponential activation frequency in [0, 1]
        self.load_freq: Tensor = torch.zeros(n_experts)
        # per-expert (s, y) curvature buffer
        self._buffers: Dict[int, List[Tuple[Tensor, Tensor]]] = {
            e: [] for e in range(n_experts)
        }
        # n_experts × n_experts co-activation matrix
        self.coactivation: Tensor = torch.zeros(n_experts, n_experts)

    # ------------------------------------------------------------------
    # Forward-pass hook
    # ------------------------------------------------------------------
    def on_forward(self, active_experts: List[int]) -> None:
        """Update TTL, load frequency, and co-activation for one forward pass."""
        active_set = set(active_experts)
        for e in range(self.n_experts):
            if e in active_set:
                self.ttl[e] = 0
                self.load_freq[e] = 0.99 * self.load_freq[e] + 0.01
            else:
                self.ttl[e] += 1
                self.load_freq[e] = self.load_freq[e] * 0.99

        for i in active_experts:
            for j in active_experts:
                self.coactivation[i, j] = (
                    0.99 * self.coactivation[i, j] + 0.01
                )

    # ------------------------------------------------------------------
    # Window sizing
    # ------------------------------------------------------------------
    def window_for_expert(self, e: int, m_min: int = 3) -> int:
        """Return load-proportional buffer window for expert e."""
        p = float(self.load_freq[e].item())
        return max(m_min, min(self.m_max, round(p * self.m_max)))

    # ------------------------------------------------------------------
    # Curvature pair management
    # ------------------------------------------------------------------
    def add_pair(self, e: int, s: Tensor, y: Tensor) -> None:
        """Add a curvature pair for expert e (no-op if expert is inactive)."""
        if self.ttl[e] > 0:
            return  # expert not active this step — skip stale pair
        m = self.window_for_expert(e)
        self._buffers[e].append((s.clone(), y.clone()))
        if len(self._buffers[e]) > m:
            self._buffers[e].pop(0)

    def get_buffer(self, e: int) -> List[Tuple[Tensor, Tensor]]:
        """Return the curvature buffer for expert e."""
        return self._buffers[e]

    # ------------------------------------------------------------------
    # Stale buffer expiry
    # ------------------------------------------------------------------
    def expire_stale(self, ttl_expire: int = None) -> int:
        """
        Purge buffers for experts that have exceeded their TTL.

        Returns:
            Number of expert buffers that were cleared.
        """
        limit = ttl_expire if ttl_expire is not None else self.ttl_expire
        cleared = 0
        for e in range(self.n_experts):
            if self.ttl[e] > limit and self._buffers[e]:
                self._buffers[e].clear()
                cleared += 1
        return cleared
