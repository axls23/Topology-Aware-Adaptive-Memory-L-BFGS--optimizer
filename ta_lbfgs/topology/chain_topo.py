"""
Axis 4 — Reasoning Chain Dependency Topology.

Monitors gradient-norm dynamics during chain-of-thought inference to
detect *pivot events* (reasoning direction reversals) and tracks which
segment of the reasoning process is currently active.

Pivot detection uses a rolling-window spike test (grad_norm > μ + σ·pivot_sigma).
PRM (process reward model) scores provide an additional signal: high scores
confirm topology stability; low scores can trigger re-evaluation.

Segment-conditioned window scaling lets the outer loop apply different
L-BFGS memory sizes depending on whether we are in reasoning, answer, or
verification mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_SEGMENTS = ("reasoning", "answer", "verify")
_WINDOW_SCALE = {"reasoning": 0.5, "answer": 1.0, "verify": 0.7}


class ChainTopologyController:
    """
    Reasoning-chain aware topology scheduler.

    Args:
        pivot_sigma: Number of standard deviations above the rolling mean
            that constitutes a pivot event (default 3.0).
        prm_ttl_scale: Reserved for future PRM-gated TTL modulation.
    """

    SEGMENTS = _SEGMENTS

    def __init__(
        self,
        pivot_sigma: float = 3.0,
        prm_ttl_scale: int = 100,
    ) -> None:
        self.pivot_sigma = pivot_sigma
        self.prm_ttl_scale = prm_ttl_scale
        self.current_segment: str = "reasoning"
        # topology_valid: True once a pivot event has been confirmed;
        # False in the initial quiescent state or on zero-variance history.
        self.topology_valid: bool = False
        self._grad_norm_history: List[float] = []

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------
    def on_outer_step(
        self,
        grad_norm: float,
        prm_score: Optional[float] = None,
        mean_entropy: Optional[float] = None,
    ) -> None:
        """
        Process one outer optimisation step.

        Checks whether the current grad_norm constitutes a pivot event
        relative to the rolling history, then appends it to the history.
        Optionally updates segment and topology validity from PRM score
        and entropy-derived reasoning state.

        Args:
            grad_norm: Gradient norm at the current outer step.
            prm_score: Optional process-reward-model score in [0, 1].
            mean_entropy: Optional mean token entropy for chain segmentation.
        """
        history = self._grad_norm_history

        # Pivot detection uses the history *before* adding the new value
        if len(history) >= 5:
            mu = sum(history) / len(history)
            var = sum((x - mu) ** 2 for x in history) / len(history)
            sigma = var ** 0.5
            if sigma > 0 and grad_norm > mu + self.pivot_sigma * sigma:
                self._on_pivot_detected()
                return  # history cleared inside; skip append

        history.append(grad_norm)
        if len(history) > 20:
            history.pop(0)

        if mean_entropy is not None:
            if mean_entropy > 2.5:
                self.current_segment = "reasoning"
            elif mean_entropy < 1.0:
                self.current_segment = "answer"
            else:
                self.current_segment = "verify"

        # PRM score gating
        if prm_score is not None:
            if prm_score >= 0.3:
                self.topology_valid = True
                if mean_entropy is None:
                    self.current_segment = "answer"
            else:
                # Low quality: stay in reasoning, do not mark valid
                self.current_segment = "reasoning"

    def on_outer_step_with_snapshot(self, snap: Dict[str, Any], grad_norm: float) -> None:
        entropy = snap.get("logit_entropy")
        mean_entropy: Optional[float] = None
        if entropy is not None:
            try:
                mean_entropy = float(entropy.detach().mean().item())
            except Exception:
                mean_entropy = None
        self.on_outer_step(grad_norm=grad_norm, prm_score=None, mean_entropy=mean_entropy)

    # ------------------------------------------------------------------
    # Pivot handling
    # ------------------------------------------------------------------
    def _on_pivot_detected(self) -> None:
        """Record a confirmed pivot event and reset the gradient history."""
        self.topology_valid = True
        self._grad_norm_history = []

    # ------------------------------------------------------------------
    # Window scale query
    # ------------------------------------------------------------------
    def window_scale(self) -> float:
        """Return the memory-window multiplier for the current segment."""
        return _WINDOW_SCALE.get(self.current_segment, 1.0)
