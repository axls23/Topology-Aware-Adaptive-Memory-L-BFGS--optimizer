"""
Persistent Homology for Loss Landscape Topology.

Uses giotto-tda to compute exact persistence diagrams on the
PCA-projected loss subspace, replacing heuristic secant-based
saddle detection with mathematically rigorous topological invariants.

Key outputs:
  - Persistence diagram (birth-death pairs for critical points)
  - Persistent saddle count (number of significant saddles)
  - Memory boost signal (additional m to add to L-BFGS memory)
  - Betti numbers β₀ (connected components), β₁ (loops/holes)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from gtda.homology import CubicalPersistence, VietorisRipsPersistence
    from gtda.diagrams import NumberOfPoints, Amplitude

    _GTDA_AVAILABLE = True
except ImportError:
    _GTDA_AVAILABLE = False


def is_available() -> bool:
    """Return True if giotto-tda is installed and importable."""
    return _GTDA_AVAILABLE


# ── Core persistence computation ─────────────────────────────────


def compute_loss_persistence(
    loss_grid: np.ndarray,
    homology_dimensions: Tuple[int, ...] = (0, 1),
) -> np.ndarray:
    """Compute the persistence diagram of a 2D loss landscape grid.

    Args:
        loss_grid: 2D array of shape (H, W) representing sampled loss values
            on the PCA-projected active subspace.
        homology_dimensions: Which Betti numbers to compute. (0,) for
            connected components only, (0, 1) to also capture loops.

    Returns:
        Persistence diagram as array of shape (N, 3) where each row
        is (birth, death, dimension). Returns empty array if giotto-tda
        is not available.
    """
    if not _GTDA_AVAILABLE:
        return np.empty((0, 3), dtype=np.float64)

    if loss_grid.ndim != 2 or loss_grid.size < 4:
        return np.empty((0, 3), dtype=np.float64)

    # CubicalPersistence expects (batch, H, W) input.
    grid_batch = loss_grid[np.newaxis, :, :].astype(np.float64)

    cp = CubicalPersistence(
        homology_dimensions=homology_dimensions,
        coeff=2,
        n_jobs=1,
    )
    diagrams = cp.fit_transform(grid_batch)  # shape (1, n_features, 3)
    diagram = diagrams[0]  # (n_features, 3)

    # Remove infinite-death features (boundary artifacts).
    finite_mask = np.isfinite(diagram[:, 1])
    return diagram[finite_mask]


def compute_pointcloud_persistence(
    points: np.ndarray,
    homology_dimensions: Tuple[int, ...] = (0, 1),
    max_edge_length: float = float("inf"),
) -> np.ndarray:
    """Compute persistence on a point cloud (e.g., gradient sketches).

    Args:
        points: 2D array of shape (N, D) — N points in D dimensions.
        homology_dimensions: Which Betti numbers to compute.
        max_edge_length: Maximum edge length in the Vietoris-Rips complex.

    Returns:
        Persistence diagram as array of shape (M, 3).
    """
    if not _GTDA_AVAILABLE:
        return np.empty((0, 3), dtype=np.float64)

    if points.ndim != 2 or points.shape[0] < 3:
        return np.empty((0, 3), dtype=np.float64)

    pts_batch = points[np.newaxis, :, :].astype(np.float64)

    vr = VietorisRipsPersistence(
        homology_dimensions=homology_dimensions,
        max_edge_length=max_edge_length,
        coeff=2,
        n_jobs=1,
    )
    diagrams = vr.fit_transform(pts_batch)
    diagram = diagrams[0]

    finite_mask = np.isfinite(diagram[:, 1])
    return diagram[finite_mask]


# ── Topological feature extraction ──────────────────────────────


def count_persistent_features(
    diagram: np.ndarray,
    dimension: int = 0,
    persistence_threshold: float = 0.1,
) -> int:
    """Count features with persistence above threshold in a given dimension.

    Args:
        diagram: Persistence diagram of shape (N, 3).
        dimension: Homology dimension to filter (0=components, 1=loops).
        persistence_threshold: Minimum (death - birth) to count a feature
            as significant.

    Returns:
        Number of persistent features.
    """
    if diagram.size == 0:
        return 0

    dim_mask = diagram[:, 2] == dimension
    filtered = diagram[dim_mask]
    if filtered.size == 0:
        return 0

    persistences = filtered[:, 1] - filtered[:, 0]
    return int(np.sum(persistences > persistence_threshold))


def count_persistent_saddles(
    diagram: np.ndarray,
    persistence_threshold: float = 0.1,
) -> int:
    """Count saddle-type critical points with significant persistence.

    In 2D cubical persistence, dimension-1 features correspond to
    saddle points connecting distinct basins. High persistence saddles
    indicate real structural features of the loss landscape.

    Args:
        diagram: Persistence diagram from compute_loss_persistence.
        persistence_threshold: Minimum persistence to count.

    Returns:
        Number of significant saddles.
    """
    return count_persistent_features(diagram, dimension=1, persistence_threshold=persistence_threshold)


def betti_numbers(
    diagram: np.ndarray,
    persistence_threshold: float = 0.1,
) -> Dict[int, int]:
    """Compute Betti numbers from a persistence diagram.

    β₀ = number of connected components (basins)
    β₁ = number of 1-cycles (loops / holes in the landscape)

    Args:
        diagram: Persistence diagram of shape (N, 3).
        persistence_threshold: Minimum persistence to count.

    Returns:
        Dict mapping dimension → Betti number.
    """
    result: Dict[int, int] = {}
    if diagram.size == 0:
        return result
    dims = set(int(d) for d in diagram[:, 2])
    for d in dims:
        result[d] = count_persistent_features(diagram, d, persistence_threshold)
    return result


def max_persistence(diagram: np.ndarray, dimension: int = 1) -> float:
    """Return the maximum persistence value in a given dimension.

    This measures the "most prominent" topological feature — useful
    for gauging how rugged the loss landscape is.
    """
    if diagram.size == 0:
        return 0.0
    dim_mask = diagram[:, 2] == dimension
    filtered = diagram[dim_mask]
    if filtered.size == 0:
        return 0.0
    return float(np.max(filtered[:, 1] - filtered[:, 0]))


def total_persistence(diagram: np.ndarray, dimension: int = 1) -> float:
    """Sum of all persistences in a given dimension (topological energy)."""
    if diagram.size == 0:
        return 0.0
    dim_mask = diagram[:, 2] == dimension
    filtered = diagram[dim_mask]
    if filtered.size == 0:
        return 0.0
    return float(np.sum(filtered[:, 1] - filtered[:, 0]))


# ── Memory sizing integration ───────────────────────────────────


def persistence_memory_boost(
    diagram: np.ndarray,
    persistence_threshold: float = 0.1,
    max_boost: int = 5,
) -> int:
    """Compute additional L-BFGS memory needed based on persistence.

    The idea: each significant saddle in the loss landscape represents
    a direction where the optimizer needs extra curvature history.
    More saddles → more memory.

    Formula: boost = min(max_boost, persistent_saddle_count)

    Args:
        diagram: Persistence diagram from compute_loss_persistence.
        persistence_threshold: Minimum persistence for a saddle to count.
        max_boost: Maximum additional memory to add.

    Returns:
        Integer memory boost ∈ [0, max_boost].
    """
    saddle_count = count_persistent_saddles(diagram, persistence_threshold)
    return min(max_boost, saddle_count)


def enhanced_memory_size(
    kappa: float,
    diagram: np.ndarray,
    m_min: int = 3,
    m_max: int = 20,
    persistence_threshold: float = 0.1,
) -> int:
    """Compute L-BFGS memory size using both κ and persistent homology.

    Combines the log-clamped κ formula from updates.pdf Section 1.5A
    with the persistence-based boost.

    m_l = clip(ceil(log₂(κ)) + persistent_saddle_count, m_min, m_max)

    Args:
        kappa: Condition number estimate for the layer.
        diagram: Persistence diagram of the local loss subspace.
        m_min: Minimum memory size.
        m_max: Maximum memory size.
        persistence_threshold: Minimum persistence for saddles.

    Returns:
        Memory size ∈ [m_min, m_max].
    """
    if kappa <= 1.0:
        base_m = m_min
    else:
        base_m = math.ceil(math.log2(max(1.0, kappa)))

    boost = persistence_memory_boost(diagram, persistence_threshold)
    return max(m_min, min(m_max, base_m + boost))


# ── Loss subspace sampling ──────────────────────────────────────


def sample_loss_grid(
    loss_fn,
    center: np.ndarray,
    directions: np.ndarray,
    grid_size: int = 32,
    scale: float = 1.0,
) -> np.ndarray:
    """Sample a 2D loss grid around a point in parameter space.

    Projects the loss function onto a 2D plane spanned by the top-2
    principal gradient directions.

    Args:
        loss_fn: Callable that takes a 1D parameter perturbation vector
            and returns the scalar loss value.
        center: 1D array — the current parameter point.
        directions: 2D array of shape (2, D) — the two principal
            directions (e.g., from PCA of gradient history).
        grid_size: Number of samples per axis (total = grid_size²).
        scale: Range of the grid in each direction.

    Returns:
        2D array of shape (grid_size, grid_size) with loss values.
    """
    alphas = np.linspace(-scale, scale, grid_size)
    grid = np.zeros((grid_size, grid_size), dtype=np.float64)

    for i, a in enumerate(alphas):
        for j, b in enumerate(alphas):
            perturbation = a * directions[0] + b * directions[1]
            grid[i, j] = loss_fn(center + perturbation)

    return grid


# ── Dashboard payload ────────────────────────────────────────────


def persistence_to_payload(
    diagram: np.ndarray,
    persistence_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Convert a persistence diagram to a JSON-serializable payload for the UI.

    Returns:
        Dict with keys:
          - diagram: list of [birth, death, dimension] triples
          - betti: {0: β₀, 1: β₁}
          - saddle_count: number of persistent saddles
          - max_persistence: maximum persistence in dim 1
          - total_persistence: sum of persistences in dim 1
    """
    if diagram.size == 0:
        return {
            "diagram": [],
            "betti": {},
            "saddle_count": 0,
            "max_persistence": 0.0,
            "total_persistence": 0.0,
        }

    return {
        "diagram": diagram.tolist(),
        "betti": betti_numbers(diagram, persistence_threshold),
        "saddle_count": count_persistent_saddles(diagram, persistence_threshold),
        "max_persistence": max_persistence(diagram),
        "total_persistence": total_persistence(diagram),
    }
