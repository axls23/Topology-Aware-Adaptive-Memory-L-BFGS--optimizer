"""
Adaptive Memory Sizing.

Maps the condition number κ of a layer's local landscape to its
L-BFGS history window size m_l. Higher κ → larger memory to
preserve directionality in narrow ravines.
"""

import math


def compute_memory_size(
    kappa: float,
    m_base: int = 5,
    m_min: int = 3,
    m_max: int = 20,
) -> int:
    """
    Compute adaptive memory size from condition number.

    Formula: m_l = clip(floor(log(κ)) + m_base, m_min, m_max)

    Rationale: well-conditioned regions (κ ≈ 1) need minimal history,
    while ill-conditioned ravines (κ >> 1) benefit from more curvature
    pairs to maintain directional memory.

    Args:
        kappa: Condition number of the layer's local landscape.
        m_base: Base memory size.
        m_min: Minimum allowed memory.
        m_max: Maximum allowed memory.

    Returns:
        Integer memory size m_l.
    """
    if kappa <= 1.0:
        return m_min

    log_kappa = math.log(max(kappa, 1.0 + 1e-12))
    m_l = int(math.floor(log_kappa) + m_base)

    return max(m_min, min(m_l, m_max))
