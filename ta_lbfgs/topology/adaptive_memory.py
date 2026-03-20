"""
Adaptive Memory Sizing.

Maps the condition number κ of a layer's local landscape to its
L-BFGS history window size m_l. Higher κ → larger memory to
preserve directionality in narrow ravines.
"""

import math
from typing import List

import torch


# ADDS: log-clamped adaptive history window mapping with explicit min/max bounds.
# REMOVES: raw log(kappa)+base formula that could drift from the specified schedule.
def compute_window(kappa: float, m_min: int = 3, m_max: int = 20) -> int:
    """Log-clamped adaptive window. kappa=1->3, kappa=64->6, kappa=1e6->20."""
    if kappa <= 1.0:
        return m_min
    return max(m_min, min(m_max, math.ceil(math.log2(kappa))))


class AitkenAccelerator:
    """Sequence acceleration for hyperparameter updates in narrow valleys."""

    def __init__(self):
        self._hist: List[torch.Tensor] = []

    # ADDS: stable Aitken delta-squared acceleration with guarded denominator.
    # REMOVES: absence of any sequence acceleration utility in adaptive memory module.
    def step(self, x: torch.Tensor) -> torch.Tensor:
        self._hist.append(x.clone())
        if len(self._hist) < 3:
            return x
        x0, x1, x2 = self._hist[-3], self._hist[-2], self._hist[-1]
        denom = x2 - 2 * x1 + x0
        mask = denom.abs() > 1e-10
        safe = torch.where(denom >= 0, denom.clamp(min=1e-12), denom.clamp(max=-1e-12))
        return torch.where(mask, x0 - (x1 - x0) ** 2 / safe, x2)


# ADDS: compatibility wrapper that now delegates to the stricter compute_window schedule.
# REMOVES: old compute_memory_size direct raw-formula implementation.
def compute_memory_size(
    kappa: float,
    m_base: int = 5,
    m_min: int = 3,
    m_max: int = 20,
) -> int:
    _ = m_base
    return compute_window(kappa, m_min=m_min, m_max=m_max)
