"""Online layerwise Randomized SVD utilities for streaming trajectory projection."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class RSVDProjectionResult:
    """Projection state returned after each online update."""

    layer_name: str
    coordinates: np.ndarray
    explained_variance_ratio: float
    cumulative_energy: np.ndarray
    sample_count: int


class OnlineRandomizedSVD:
    """
    Online randomized SVD over a stream of high-dimensional vectors.

    Uses a Gaussian sketch matrix and an exponentially weighted covariance
    update in sketch-space so we never store historical vectors.
    """

    def __init__(
        self,
        input_dim: int,
        n_components: int = 3,
        sketch_dim: int = 24,
        forgetting_factor: float = 0.98,
        seed: int = 0,
    ):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if sketch_dim < n_components:
            raise ValueError("sketch_dim must be >= n_components")
        if not (0.0 < forgetting_factor < 1.0):
            raise ValueError("forgetting_factor must be in (0, 1)")

        self.input_dim = int(input_dim)
        self.n_components = int(n_components)
        self.sketch_dim = int(sketch_dim)
        self.forgetting_factor = float(forgetting_factor)
        self.sample_count = 0

        rng = np.random.default_rng(seed)
        omega = rng.standard_normal((self.input_dim, self.sketch_dim))
        self.omega = omega / np.sqrt(self.sketch_dim)

        self.mean = np.zeros(self.input_dim, dtype=np.float64)
        self.sketch_cov = np.zeros((self.sketch_dim, self.sketch_dim), dtype=np.float64)
        self._basis = np.eye(self.input_dim, self.n_components, dtype=np.float64)
        self._cumulative_energy = np.zeros(self.n_components, dtype=np.float64)
        self._explained_ratio = 0.0

    def update(self, vector: np.ndarray) -> RSVDProjectionResult:
        """Consume one new observation and update projection basis incrementally."""
        x = np.asarray(vector, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.input_dim:
            raise ValueError(
                f"Expected vector of length {self.input_dim}, got {x.shape[0]}"
            )

        self.sample_count += 1
        alpha = 1.0 - self.forgetting_factor

        self.mean = self.forgetting_factor * self.mean + alpha * x
        centered = x - self.mean
        sketch = centered @ self.omega

        outer = np.outer(sketch, sketch)
        self.sketch_cov = self.forgetting_factor * self.sketch_cov + alpha * outer

        evals, evecs = np.linalg.eigh(self.sketch_cov)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        evecs = evecs[:, order]

        top_eval = evals[: self.n_components]
        top_evec = evecs[:, : self.n_components]

        # Lift sketch-space directions back to input-space and orthonormalize.
        basis_raw = self.omega @ top_evec
        q, _ = np.linalg.qr(basis_raw, mode="reduced")
        if q.shape[1] < self.n_components:
            pad = np.eye(self.input_dim, self.n_components - q.shape[1])
            q = np.concatenate([q, pad], axis=1)
        self._basis = q[:, : self.n_components]

        total = float(np.sum(evals))
        if total > 0:
            self._cumulative_energy = np.cumsum(top_eval) / total
            self._explained_ratio = float(np.sum(top_eval) / total)
        else:
            self._cumulative_energy = np.zeros(self.n_components, dtype=np.float64)
            self._explained_ratio = 0.0

        coords = centered @ self._basis
        return RSVDProjectionResult(
            layer_name="",
            coordinates=coords,
            explained_variance_ratio=self._explained_ratio,
            cumulative_energy=self._cumulative_energy.copy(),
            sample_count=self.sample_count,
        )


class LayerwiseOnlineRSVD:
    """Manages per-layer online RSVD trackers and returns monitored-layer state."""

    def __init__(
        self,
        monitored_layer: str,
        n_components: int = 3,
        sketch_dim: int = 24,
        forgetting_factor: float = 0.98,
        warning_threshold: float = 0.8,
        seed: int = 0,
    ):
        self.monitored_layer = monitored_layer
        self.n_components = n_components
        self.sketch_dim = sketch_dim
        self.forgetting_factor = forgetting_factor
        self.warning_threshold = warning_threshold
        self.seed = seed
        self._trackers: Dict[str, OnlineRandomizedSVD] = {}

    def update(self, layer_name: str, vector: np.ndarray) -> Optional[Dict[str, object]]:
        """Update the RSVD tracker for one layer and emit monitored-layer summary."""
        vec = np.asarray(vector, dtype=np.float64).reshape(-1)
        tracker = self._trackers.get(layer_name)
        if tracker is None:
            tracker = OnlineRandomizedSVD(
                input_dim=vec.shape[0],
                n_components=self.n_components,
                sketch_dim=max(self.sketch_dim, self.n_components),
                forgetting_factor=self.forgetting_factor,
                seed=self.seed + len(self._trackers),
            )
            self._trackers[layer_name] = tracker

        result = tracker.update(vec)

        if layer_name != self.monitored_layer:
            return None

        ratio = float(result.explained_variance_ratio)
        is_reliable = ratio >= self.warning_threshold
        status = "Reliable" if is_reliable else "Low-Fidelity"
        warning = None
        if not is_reliable:
            warning = (
                f"Top-{self.n_components} components explain only {ratio * 100.0:.1f}% variance. "
                "Landscape rendering may be inaccurate in low dimensions."
            )

        return {
            "layer": layer_name,
            "coords": result.coordinates.astype(np.float64),
            "explained_variance_ratio": ratio,
            "cumulative_energy": result.cumulative_energy.astype(np.float64),
            "sample_count": result.sample_count,
            "status": status,
            "warning": warning,
            "warning_threshold": self.warning_threshold,
            "n_components": self.n_components,
        }
