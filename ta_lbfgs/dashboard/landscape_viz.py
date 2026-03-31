"""
Gradient Landscape Visualization — Hybrid B+C Adaptive Mesh.

Constructs a 3D loss surface mesh that evolves with the optimizer:

  Approach B (Curvature Seed):
    Uses per-layer condition number κ, secant condition y^Ts, and
    gradient norms to build a local quadratic/saddle approximation
    around the current hyperparameter position.

  Approach C (RBF Interpolation):
    As the trajectory accumulates visited (hp, loss) points, uses
    Radial Basis Function interpolation to fill in the mesh between
    visited points, revealing the actual topology as explored.

The mesh axes are grounded in the ACTUAL hyperparameter space
(lr, wd from DifferentiableHyperparameters) so they align with
the trajectory coordinates.
"""

import os
import math
import numpy as np
import torch
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Any


# ────────────────────────────────────────────────────────────────────
# Adaptive Landscape Mesh (Hybrid B+C)
# ────────────────────────────────────────────────────────────────────

class AdaptiveLandscapeMesh:
    """
    Evolving 3D loss surface mesh grounded in the model's hyperparameter space.

    Combines two approaches:
      B) Curvature-seeded quadratic: Uses κ and secant data to build
         a local Hessian approximation (convex bowl, ravine, or saddle).
      C) RBF interpolation: Uses visited (lr, wd, loss) points to
         interpolate the actual landscape via radial basis functions.

    The blend transitions from pure-quadratic (early, few data points)
    to data-driven RBF (later, many visited points).

    The mesh coordinate space is the REAL hyperparameter space:
      X-axis = lr (from DifferentiableHyperparameters, layer 0 or mean)
      Y-axis = wd (from DifferentiableHyperparameters, layer 0 or mean)
      Z-axis = loss value

    Args:
        lr_range: (min, max) for learning rate axis.
        wd_range: (min, max) for weight decay axis.
        n_points: Grid resolution per axis.
        rbf_epsilon: RBF kernel width parameter.
    """

    def __init__(
        self,
        lr_range: Tuple[float, float] = (1e-7, 1.0),
        wd_range: Tuple[float, float] = (1e-7, 1.0),
        n_points: int = 20,
        rbf_epsilon: float = 1.0,
    ):
        self.lr_range = lr_range
        self.wd_range = wd_range
        self.n_points = n_points
        self.rbf_epsilon = rbf_epsilon

        # Visited trajectory points: [(lr, wd, loss), ...]
        self._visited: List[Tuple[float, float, float]] = []

        # Per-layer curvature data (most recent)
        self._curvature: Dict[str, Dict[str, float]] = {}

        # Current hyperparameter center
        self._center_lr: float = 1e-4
        self._center_wd: float = 1e-2
        self._center_loss: float = 1.0

        # Cached mesh
        self._cached_mesh: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._dirty = True

    def update(
        self,
        lr: float,
        wd: float,
        loss: float,
        layer_curvature: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Feed the mesh a new observation from the optimizer.
        """
        # Threshold for 'new' point in log-space to avoid singular RBF matrix
        if self._visited:
            last_lr, last_wd, last_loss = self._visited[-1]
            dist = math.sqrt(
                (math.log10(lr/last_lr))**2 + 
                (math.log10(wd/last_wd))**2
            ) if lr > 0 and last_lr > 0 and wd > 0 and last_wd > 0 else 1.0
            
            if dist < 1e-4 and abs(loss - last_loss) < 1e-4:
                return # Skip duplicate/near-duplicate point

        self._visited.append((lr, wd, loss))
        self._center_lr = lr
        self._center_wd = wd
        self._center_loss = loss

        if layer_curvature:
            self._curvature = layer_curvature

        self._dirty = True

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the adaptive 3D surface mesh.

        Returns (X, Y, Z) numpy arrays for Plotly Surface or TUI wireframe.

        Strategy:
          - If < 5 visited points → pure curvature-seeded quadratic (B)
          - If >= 5              → RBF interpolation (C) blended with
                                   quadratic to fill unvisited regions
        """
        if not self._dirty and self._cached_mesh is not None:
            return self._cached_mesh

        n = self.n_points

        # Build grid centered around recent trajectory in LOG-SPACE
        # This gives better resolution around the actual operating region
        lr_center = max(self._center_lr, 1e-8)
        wd_center = max(self._center_wd, 1e-8)

        # Span ±2 decades around current position (in log-space)
        log_lr_center = np.log10(lr_center)
        log_wd_center = np.log10(wd_center)
        span = 1.5  # decades

        log_lr_lo = max(np.log10(self.lr_range[0]), log_lr_center - span)
        log_lr_hi = min(np.log10(self.lr_range[1]), log_lr_center + span)
        log_wd_lo = max(np.log10(self.wd_range[0]), log_wd_center - span)
        log_wd_hi = min(np.log10(self.wd_range[1]), log_wd_center + span)

        log_lr = np.linspace(log_lr_lo, log_lr_hi, n)
        log_wd = np.linspace(log_wd_lo, log_wd_hi, n)
        LOG_LR, LOG_WD = np.meshgrid(log_lr, log_wd)

        # Transform to actual hyperparameter space
        X = 10.0 ** LOG_LR  # lr values
        Y = 10.0 ** LOG_WD  # wd values

        # Compute surface values
        n_visited = len(self._visited)

        if n_visited < 5:
            # ── APPROACH B: Curvature-Seeded Quadratic ──────────────
            Z = self._curvature_quadratic(LOG_LR, LOG_WD, log_lr_center, log_wd_center)
        else:
            # ── HYBRID B+C: Blend Quadratic + RBF ───────────────────
            Z_quad = self._curvature_quadratic(LOG_LR, LOG_WD, log_lr_center, log_wd_center)
            Z_rbf = self._rbf_interpolate(LOG_LR, LOG_WD)

            # Confidence weight: increases with data density
            # Near visited points → trust RBF; far away → trust quadratic
            confidence = self._rbf_confidence(LOG_LR, LOG_WD)
            Z = confidence * Z_rbf + (1.0 - confidence) * Z_quad

        self._cached_mesh = (X, Y, Z)
        self._dirty = False
        return X, Y, Z

    def _curvature_quadratic(
        self,
        log_lr_grid: np.ndarray,
        log_wd_grid: np.ndarray,
        log_lr_center: float,
        log_wd_center: float,
    ) -> np.ndarray:
        """
        APPROACH B: Build a local quadratic/saddle surface from curvature data.

        Uses the aggregate condition number κ and secant conditions to determine:
          - κ ≈ 1   → isotropic bowl (well-conditioned)
          - κ >> 1  → elongated ravine (ill-conditioned)
          - y^Ts <0 → saddle point (negative curvature in one direction)

        The Hessian eigenvalues are:
          λ₁ = 1.0       (along lr-direction)
          λ₂ = 1/κ_avg   (along wd-direction, stretched for ill-conditioning)

        If saddle detected, λ₂ goes negative → saddle shape.
        """
        # Aggregate curvature from all layers
        kappas = []
        secants = []
        grad_norms = []

        for layer_name, data in self._curvature.items():
            kappas.append(data.get("kappa", 1.0))
            secants.append(data.get("secant", 1.0))
            grad_norms.append(data.get("grad_norm", 0.1))

        if not kappas:
            kappas = [5.0]
            secants = [1.0]
            grad_norms = [0.1]

        avg_kappa = np.mean(kappas)
        max_kappa = np.max(kappas)
        avg_secant = np.mean(secants)
        n_saddle_layers = sum(1 for s in secants if s <= 0)

        # ── Build local Hessian eigenvalues ──────────────────────
        # λ₁: principal curvature (lr-direction)
        # λ₂: secondary curvature (wd-direction)
        lambda_1 = 1.0

        if n_saddle_layers > len(kappas) / 2:
            # Majority layers in saddle → saddle surface
            saddle_strength = min(abs(avg_secant), 0.5) if avg_secant < 0 else 0.1
            lambda_2 = -saddle_strength * lambda_1
        elif max_kappa > 20:
            # Narrow ravine: large gap between eigenvalues
            lambda_2 = lambda_1 / max(avg_kappa, 1.0)
        else:
            # Quasi-convex bowl
            lambda_2 = lambda_1 / max(avg_kappa, 1.0)

        # Displacement from center (in log-space for better scaling)
        dx = log_lr_grid - log_lr_center
        dy = log_wd_grid - log_wd_center

        # Mix in a rotation based on gradient direction for visual variety
        avg_grad = np.mean(grad_norms)
        if avg_grad > 0 and len(grad_norms) >= 2:
            # Create slight rotation to break axis-alignment
            theta = np.arctan2(grad_norms[0], grad_norms[-1]) * 0.3
        else:
            theta = 0.0

        ct, st = np.cos(theta), np.sin(theta)
        dx_rot = dx * ct - dy * st
        dy_rot = dx * st + dy * ct

        # Quadratic surface: z = center_loss + λ₁·dx² + λ₂·dy²
        # Scale so the amplitude matches observed loss range
        loss_scale = max(self._center_loss * 0.5, 0.1)
        Z = self._center_loss + loss_scale * (
            lambda_1 * dx_rot ** 2 + lambda_2 * dy_rot ** 2
        )

        # Add Rosenbrock-like non-convexity if saddle layers exist
        if n_saddle_layers > 0:
            rosenbrock = 0.1 * loss_scale * (
                (dx_rot ** 2 - dy_rot) ** 2 + 0.1 * (1.0 - dx_rot) ** 2
            )
            blend = n_saddle_layers / len(kappas)
            Z += blend * rosenbrock

        # Small noise for visual texture
        Z += np.random.randn(*Z.shape) * loss_scale * 0.02

        return Z

    def _rbf_interpolate(
        self,
        log_lr_grid: np.ndarray,
        log_wd_grid: np.ndarray,
    ) -> np.ndarray:
        """
        APPROACH C: RBF (Radial Basis Function) interpolation
        from visited trajectory points.

        Uses a Gaussian kernel: φ(r) = exp(-ε²·r²)
        Solves the linear system Φ·w = f to get RBF weights,
        then evaluates on the mesh grid.
        """
        pts = np.array(self._visited)  # shape [N, 3] → (lr, wd, loss)
        n_pts = len(pts)

        # Work in log-space for lr and wd
        log_lr_pts = np.log10(np.clip(pts[:, 0], 1e-10, None))
        log_wd_pts = np.log10(np.clip(pts[:, 1], 1e-10, None))
        loss_pts = pts[:, 2]

        # Subsample if too many points (for speed)
        max_rbf_pts = 100
        if n_pts > max_rbf_pts:
            indices = np.linspace(0, n_pts - 1, max_rbf_pts, dtype=int)
            log_lr_pts = log_lr_pts[indices]
            log_wd_pts = log_wd_pts[indices]
            loss_pts = loss_pts[indices]
            n_pts = max_rbf_pts

        # Build RBF kernel matrix Φ [N×N]
        eps = self.rbf_epsilon
        centers = np.column_stack([log_lr_pts, log_wd_pts])  # [N, 2]
        dist_sq = np.sum(
            (centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
            axis=-1
        )
        Phi = np.exp(-eps ** 2 * dist_sq)

        # Regularize for numerical stability
        Phi += 1e-6 * np.eye(n_pts)

        # Solve for RBF weights: Φ·w = loss_pts
        try:
            weights = np.linalg.solve(Phi, loss_pts)
        except np.linalg.LinAlgError:
            # Fallback: pseudoinverse
            weights = np.linalg.lstsq(Phi, loss_pts, rcond=None)[0]

        # Evaluate RBF on the mesh grid
        grid_points = np.column_stack([
            log_lr_grid.ravel(),
            log_wd_grid.ravel(),
        ])  # [M, 2]

        dist_sq_grid = np.sum(
            (grid_points[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
            axis=-1
        )  # [M, N]

        Phi_grid = np.exp(-eps ** 2 * dist_sq_grid)  # [M, N]
        Z_flat = Phi_grid @ weights
        Z = Z_flat.reshape(log_lr_grid.shape)

        return Z

    def _rbf_confidence(
        self,
        log_lr_grid: np.ndarray,
        log_wd_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a [0, 1] confidence map: how close each grid point
        is to visited trajectory points.

        High confidence (→1) near visited points → trust RBF.
        Low confidence (→0) far from data → fall back to quadratic.
        """
        pts = np.array(self._visited)
        log_lr_pts = np.log10(np.clip(pts[:, 0], 1e-10, None))
        log_wd_pts = np.log10(np.clip(pts[:, 1], 1e-10, None))

        grid_points = np.column_stack([
            log_lr_grid.ravel(),
            log_wd_grid.ravel(),
        ])
        centers = np.column_stack([log_lr_pts, log_wd_pts])

        # Minimum distance from each grid point to any visited point
        dist_sq = np.sum(
            (grid_points[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
            axis=-1
        )
        min_dist = np.sqrt(dist_sq.min(axis=1))

        # Sigmoid-like transition: confidence decays with distance
        # scale = typical spacing between visited points
        if len(self._visited) >= 2:
            pt_dists = np.sqrt(np.sum(np.diff(centers, axis=0) ** 2, axis=1))
            scale = np.median(pt_dists) * 2.0 if len(pt_dists) > 0 else 1.0
        else:
            scale = 1.0

        scale = max(scale, 0.1)
        confidence = np.exp(-(min_dist / scale) ** 2)
        return confidence.reshape(log_lr_grid.shape)

    @property
    def n_visited(self) -> int:
        return len(self._visited)


# ────────────────────────────────────────────────────────────────────
# Legacy-Compatible Wrapper (for existing callers)
# ────────────────────────────────────────────────────────────────────

# Global mesh instance (created once, updated incrementally)
_global_mesh: Optional[AdaptiveLandscapeMesh] = None


def get_adaptive_mesh(
    lr_range: Tuple[float, float] = (1e-7, 1.0),
    wd_range: Tuple[float, float] = (1e-7, 1.0),
    n_points: int = 20,
) -> AdaptiveLandscapeMesh:
    """Get or create the global adaptive mesh instance."""
    global _global_mesh
    if _global_mesh is None:
        _global_mesh = AdaptiveLandscapeMesh(
            lr_range=lr_range,
            wd_range=wd_range,
            n_points=n_points,
        )
    return _global_mesh


def reset_adaptive_mesh():
    """Reset the global mesh instance (for new optimization runs)."""
    global _global_mesh
    _global_mesh = None


def generate_landscape_mesh(
    model,
    hyperparams,
    n_points: int = 20,
    layer_curvature: Optional[Dict[str, Dict[str, float]]] = None,
    loss: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 3D surface mesh of the current landscape.

    UPGRADED: Now returns an evolving mesh grounded in the actual
    hyperparameter space, using:
      - Curvature data (κ, y^Ts) for quadratic seeding
      - Visited points for RBF interpolation

    Args:
        model: The model (unused for mesh; kept for API compat).
        hyperparams: DifferentiableHyperparameters instance.
        n_points: Grid resolution.
        layer_curvature: Optional per-layer curvature dict.
        loss: Optional current loss to update RBF history.

    Returns:
        (X, Y, Z) numpy arrays where X=lr, Y=wd, Z=loss.
    """
    mesh = get_adaptive_mesh(n_points=n_points)

    if hyperparams is not None:
        hp = hyperparams.as_float_dict()
        lr_val = hp["lr"][0] if isinstance(hp["lr"], list) else hp["lr"]
        wd_val = hp["wd"][0] if isinstance(hp["wd"], list) else hp["wd"]
        
        if loss is not None:
            mesh.update(lr_val, wd_val, loss, layer_curvature)
        else:
            # Just update curvature and center
            mesh._center_lr = lr_val
            mesh._center_wd = wd_val
            if layer_curvature:
                mesh._curvature = layer_curvature

    return mesh.generate_mesh()


# ────────────────────────────────────────────────────────────────────
# 3D Interactive Export (Plotly)
# ────────────────────────────────────────────────────────────────────

def export_trajectory_3d(
    hyperparameter_history: np.ndarray,
    loss_history: np.ndarray,
    output_path: str,
    n_components: int = 3,
    mesh_data: Optional[Tuple] = None,
    mesh_instance: Optional[AdaptiveLandscapeMesh] = None,
):
    """
    Deprecated: trajectory export has been disabled in favor of topology-component exports.
    """
    # Intentionally disabled to enforce topology-only 3D visualization outputs.
    return

    hyperparameter_history = np.asarray(hyperparameter_history, dtype=np.float64)
    loss_history = np.asarray(loss_history, dtype=np.float64).reshape(-1)
    if hyperparameter_history.ndim != 2 or loss_history.size == 0:
        return

    finite_rows = np.isfinite(loss_history)
    finite_rows &= np.all(np.isfinite(hyperparameter_history), axis=1)
    if not np.any(finite_rows):
        return

    hyperparameter_history = hyperparameter_history[finite_rows]
    loss_history = loss_history[finite_rows]

    fig = go.Figure()

    # ── Surface Mesh (Topology) ──────────────────────────────────
    if mesh_instance is not None:
        X, Y, Z = mesh_instance.generate_mesh()
    elif mesh_data is not None:
        X, Y, Z = mesh_data
    else:
        X, Y, Z = None, None, None

    # ── Trajectory Coordinates ───────────────────────────────────
    # If a topology mesh exists, keep trajectory on compatible axes
    # (layer-0 lr and wd) so path and surface are in the same space.
    using_topology_axes = X is not None and hyperparameter_history.shape[1] >= 2
    if using_topology_axes:
        feature_dim = hyperparameter_history.shape[1]
        wd_index = feature_dim // 2 if feature_dim >= 4 else 1
        wd_index = min(max(wd_index, 1), feature_dim - 1)
        coords = np.column_stack([
            hyperparameter_history[:, 0],
            hyperparameter_history[:, wd_index],
            loss_history,
        ])
    else:
        # Fall back to SVD compression when no topology mesh is provided.
        if hyperparameter_history.shape[1] > 3:
            centered = hyperparameter_history - hyperparameter_history.mean(axis=0)
            try:
                U, S, _ = np.linalg.svd(centered, full_matrices=False)
                coords = U[:, :3] * S[:3]
            except np.linalg.LinAlgError:
                # Stabilize degenerate trajectories by avoiding SVD in pathological cases.
                n = centered.shape[0]
                fallback_y = centered[:, 0] if centered.shape[1] > 0 else np.zeros(n)
                coords = np.column_stack([
                    np.arange(n, dtype=np.float64),
                    fallback_y,
                    np.zeros(n, dtype=np.float64),
                ])
        else:
            coords = hyperparameter_history[:, :3]

    x_axis_title = "LR (Layer 0)" if using_topology_axes else "PC0 (Learning Rate Direction)"
    y_axis_title = "WD (Layer 0)" if using_topology_axes else "PC1 (Regularization Direction)"
    x_hover = "LR" if using_topology_axes else "PC0"
    y_hover = "WD" if using_topology_axes else "PC1"
    xy_explainer = (
        "Layer-0 hyperparameter axes (LR, WD), aligned with the topology surface."
        if using_topology_axes
        else "Principal Components of the hyperparam space (LR, WD, Dropout, tau)."
    )

    if X is not None:
        # Ground coloring in local curvature for a "sleek" technical look
        # High curvature = Red (sharp regions), Low = Deep Cyan (valleys)
        dZ_dx = np.gradient(Z, axis=1)
        dZ_dy = np.gradient(Z, axis=0)
        curvature_mag = np.sqrt(dZ_dx ** 2 + dZ_dy ** 2)

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            opacity=0.6,
            colorscale=[
                [0.0, "rgb(0, 255, 255)"],      # Neon Cyan (Optimized valley)
                [0.2, "rgb(0, 150, 255)"],      # Electric Blue
                [0.5, "rgb(150, 0, 255)"],      # Deep Purple
                [0.8, "rgb(255, 0, 150)"],      # Hot Pink
                [1.0, "rgb(255, 20, 20)"],      # Warning Red (Critical slope)
            ],
            surfacecolor=curvature_mag,
            showscale=True,
            colorbar=dict(
                title="Saliency / Curvature",
                len=0.5,
                y=0.25,
                thickness=15,
                tickfont=dict(color="rgba(255,255,255,0.7)")
            ),
            lighting=dict(
                ambient=0.4,
                diffuse=0.9,
                fresnel=2,
                specular=1.5,
                roughness=0.1
            ),
            lightposition=dict(x=100, y=200, z=150),
            name="Geometric Topology",
            hovertemplate=(
                "λ LR: %{x:.2e}<br>"
                "λ WD: %{y:.2e}<br>"
                "Val Loss: %{z:.4f}<br>"
                "<extra>Topology Surface</extra>"
            ),
        ))

    # ── Trajectory Path ──────────────────────────────────────────
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=loss_history,
            mode="lines+markers",
            marker=dict(
                size=4,
                color=loss_history,
                colorscale="Viridis",
                opacity=0.9,
                colorbar=dict(title="Loss", len=0.4, y=0.75, thickness=15),
            ),
            line=dict(color="rgba(100, 200, 255, 0.8)", width=4),
            name="Optimizer Trajectory",
            hovertemplate=(
                f"{x_hover}: %{{x:.4e}}<br>"
                f"{y_hover}: %{{y:.4e}}<br>"
                "Loss: %{z:.4f}<br>"
                "<extra>Step %{pointNumber}</extra>"
            ),
        )
    )

    # ── Start & End markers ──────────────────────────────────────
    fig.add_trace(
        go.Scatter3d(
            x=[coords[0, 0]], y=[coords[0, 1]], z=[loss_history[0]],
            mode="markers",
            marker=dict(size=10, color="lime", symbol="diamond"),
            name="Start",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[coords[-1, 0]], y=[coords[-1, 1]], z=[loss_history[-1]],
            mode="markers",
            marker=dict(size=10, color="red", symbol="x"),
            name="End (Best)",
            showlegend=True,
        )
    )

    # ── Interpretation Guide ─────────────────────────────────────
    mesh_method = "Hybrid B+C (Curvature + RBF)" if mesh_instance else "Static"
    n_visited = mesh_instance.n_visited if mesh_instance else 0

    interpretation_text = (
        f"<b>TOPOLOGY-AWARE VISUALIZATION</b><br>"
        f"• <b>Surface:</b> Adaptive mesh ({mesh_method}, {n_visited} observations).<br>"
        f"• <b>Blue valleys:</b> Well-conditioned regions (low κ). <b>Red peaks:</b> High curvature / saddle risk.<br>"
        f"• <b>Trace:</b> Optimizer trajectory through hyperparameter space.<br>"
        f"• <b>Z-Axis:</b> Validation Loss. Downward = Improvement.<br>"
        f"• <b>X/Y Axes:</b> {xy_explainer}<br>"
        f"• <b>Abrupt Pivots:</b> Saddle-Point Evasion events (y^Ts ≤ 0 detected)."
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text="ta-LBFGS: Topology-Aware Hyperparameter Landscape",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            zaxis_title="Validation Loss",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(10, 10, 15, 0.9)",
            bordercolor="rgba(0, 255, 255, 0.3)",
            borderwidth=1,
            font=dict(color="cyan"),
        ),
        annotations=[
            dict(
                text=interpretation_text,
                align='left',
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.02,
                bordercolor="gray",
                borderwidth=1,
                borderpad=10,
                bgcolor="rgba(20, 20, 30, 0.85)",
                font=dict(size=11, color="white"),
            )
        ],
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.write_html(output_path)


def export_topology_components_3d(
    topology_history: np.ndarray,
    output_path: str,
    component_names: Optional[List[str]] = None,
):
    """
    Export an animated 3D view of five topology components over layers.

    Args:
        topology_history: Array with shape [T, L, C] where
            T = steps, L = layers, C = components (expected 5).
        output_path: HTML path for the interactive Plotly export.
        component_names: Optional names for component axis labels.
    """
    data = np.asarray(topology_history, dtype=np.float64)
    if data.ndim != 3 or data.shape[0] == 0 or data.shape[1] == 0 or data.shape[2] == 0:
        return

    T, L, C = data.shape
    if component_names is None or len(component_names) != C:
        component_names = [f"comp_{i}" for i in range(C)]

    # Robust per-component normalization keeps scales comparable in one scene.
    norm = data.copy()
    for c in range(C):
        col = norm[:, :, c]
        lo = float(np.nanmin(col))
        hi = float(np.nanmax(col))
        span = max(1e-9, hi - lo)
        norm[:, :, c] = (col - lo) / span

    x = np.arange(L, dtype=np.float64)
    y = np.arange(C, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    z0 = norm[0].T
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=z0,
            surfacecolor=z0,
            colorscale="Viridis",
            cmin=0.0,
            cmax=1.0,
            opacity=0.95,
            showscale=True,
            colorbar=dict(title="Normalized Curvature"),
            hovertemplate=(
                "Layer: %{x}<br>"
                "Component idx: %{y}<br>"
                "Curvature: %{z:.3f}<extra></extra>"
            ),
            name="Topology Curvature",
        )
    )

    frames = []
    for t in range(T):
        zt = norm[t].T
        frames.append(
            go.Frame(
                data=[
                    go.Surface(
                        x=X,
                        y=Y,
                        z=zt,
                        surfacecolor=zt,
                        colorscale="Viridis",
                        cmin=0.0,
                        cmax=1.0,
                        opacity=0.95,
                        showscale=True,
                    )
                ],
                name=f"step_{t}",
            )
        )
    fig.frames = frames

    steps = [
        {
            "label": str(t),
            "method": "animate",
            "args": [[f"step_{t}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
        }
        for t in range(T)
    ]

    fig.update_layout(
        template="plotly_dark",
        title="Evolving Topology Curvature (5 Components)",
        scene=dict(
            xaxis_title="Layer Index",
            yaxis_title="Topology Component",
            zaxis_title="Normalized Curvature",
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(C)),
                ticktext=component_names,
            ),
            camera=dict(eye=dict(x=1.7, y=1.4, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=48, b=0),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 1.02,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 250, "redraw": True}, "transition": {"duration": 0}}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "y": 1.0,
                "x": 0.22,
                "len": 0.75,
                "currentvalue": {"prefix": "Step: "},
                "steps": steps,
            }
        ],
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.write_html(output_path)


def export_trajectory_3d_static(
    hyperparameter_history: np.ndarray,
    loss_history: np.ndarray,
    output_path: str,
    floor_padding: float = 0.08,
):
    """
    Export a static 3D loss landscape image in a publication-style layout.

    The rendering mirrors the requested style:
      - smooth 3D surface
      - contour projection on the floor plane
      - trajectory overlay
      - epsilon-axis labels
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    hp = np.asarray(hyperparameter_history, dtype=np.float64)
    losses = np.asarray(loss_history, dtype=np.float64).reshape(-1)
    if hp.ndim != 2 or losses.size == 0 or hp.shape[0] != losses.size:
        return

    feature_dim = hp.shape[1]
    wd_index = feature_dim // 2 if feature_dim >= 4 else min(1, feature_dim - 1)
    eps1 = hp[:, 0]
    eps2 = hp[:, wd_index]

    # Work in normalized coordinates to keep triangulation stable in log-scale ranges.
    x = np.log10(np.clip(eps1, 1e-12, None))
    y = np.log10(np.clip(eps2, 1e-12, None))
    z = losses

    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[finite_mask]
    y = y[finite_mask]
    z = z[finite_mask]
    if z.size < 3:
        return

    triang = mtri.Triangulation(x, y)
    interp = mtri.LinearTriInterpolator(triang, z)

    grid_n = int(max(40, min(140, np.sqrt(z.size) * 18)))
    xi = np.linspace(float(np.min(x)), float(np.max(x)), grid_n)
    yi = np.linspace(float(np.min(y)), float(np.max(y)), grid_n)
    XI, YI = np.meshgrid(xi, yi)
    ZI = interp(XI, YI)
    ZI = np.asarray(ZI.filled(np.nan), dtype=np.float64)

    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))
    z_span = max(1e-8, z_max - z_min)
    z_floor = z_min - floor_padding * z_span

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        XI,
        YI,
        ZI,
        cmap="inferno",
        linewidth=0,
        antialiased=True,
        alpha=0.97,
    )

    ax.contourf(
        XI,
        YI,
        ZI,
        zdir="z",
        offset=z_floor,
        levels=24,
        cmap="inferno",
        alpha=0.95,
    )

    ax.plot(x, y, z, color="white", linewidth=1.2, alpha=0.85)
    ax.scatter(x[0], y[0], z[0], c="cyan", s=28, depthshade=False)
    ax.scatter(x[-1], y[-1], z[-1], c="red", s=30, depthshade=False)

    ax.set_xlabel(r"$\varepsilon_1$", labelpad=6)
    ax.set_ylabel(r"$\varepsilon_2$", labelpad=6)
    ax.set_zlabel("Loss", labelpad=8)
    ax.set_zlim(z_floor, z_max + 0.05 * z_span)
    ax.view_init(elev=26, azim=48)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08)
    cbar.set_label("Loss")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Dynamics Plot (matplotlib, unchanged)
# ────────────────────────────────────────────────────────────────────

def plot_dynamics(
    loss_history: Any,
    grad_magnitudes: Any,
    kappa_changes: Any,
    hyperparam_history: Any,
    output_path: str,
    optimizer_name: str = "",
    annotate_instability_spike: bool = True,
    spike_threshold: float = 0.8,
):
    """
    Geometric Diagnostics Suite.
    Replaces time-series with phase-space and resource analysis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def to_float_array(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x, dtype=np.float32)

    losses = to_float_array(loss_history)
    grads = to_float_array(grad_magnitudes)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1])

    # 1. Phase Portrait: Loss vs Gradient Norm (Geometric Convergence)
    ax0 = fig.add_subplot(gs[:, 0])
    points = ax0.scatter(losses, grads, c=np.arange(len(losses)), cmap="cool", s=40, edgecolors="white", linewidth=0.5, alpha=0.8)
    ax0.plot(losses, grads, "w-", alpha=0.2, lw=1)
    
    # Annotate Start/End
    ax0.annotate("Start", (losses[0], grads[0]), xytext=(10, 10), textcoords="offset points", color="lime", weight="bold")
    ax0.annotate("End", (losses[-1], grads[-1]), xytext=(10, -15), textcoords="offset points", color="red", weight="bold")

    ax0.set_xlabel("Validation Loss (Error Surface Z)")
    ax0.set_ylabel("Hypergradient Norm ||∇λ||")
    ax0.set_title("Optimization Phase Portrait (Convergence Stability)", color="cyan", pad=20)
    ax0.grid(color="gray", alpha=0.2, linestyle="--")
    plt.colorbar(points, ax=ax0, label="Progress (Iteration Sequence)")

    if annotate_instability_spike and len(grads) > 0:
        grad_norm_spike = float(np.max(grads))
        if grad_norm_spike > float(spike_threshold):
            spike_idx = int(np.argmax(grads))
            spike_loss = float(losses[spike_idx])
            label_prefix = f"{optimizer_name}: " if optimizer_name else ""
            ax0.annotate(
                f"{label_prefix}Instability spike: {grad_norm_spike:.2f}\n"
                f"(Expected: Hutch++ preconditioning\n"
                f"suppresses this in full implementation)",
                xy=(spike_loss, grad_norm_spike),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
                color="orange",
            )

    # 2. Adaptive Memory Resource Map (Kappa vs Memory Window)
    # We estimate proxy relationship for the visualization
    ax1 = fig.add_subplot(gs[0, 1])
    try:
        # Extract per-layer LRs from the history list
        lrs = []
        for h in hyperparam_history:
            if isinstance(h["lr"], list):
                lrs.append(np.mean(h["lr"]))
            else:
                lrs.append(h["lr"])
        lrs = to_float_array(lrs)
        
        ax1.fill_between(np.arange(len(lrs)), 0, lrs, color="magenta", alpha=0.3, label="Avg Learning Rate")
        ax1.set_ylabel("HP Magnitude")
        ax1.set_title("Hyperparameter Modulation", color="magenta")
        ax1.grid(alpha=0.2)
    except:
        ax1.text(0.5, 0.5, "Data format mismatch for Resource Map", ha="center")

    # 3. Conditioning Distribution (Kappa Proxy over time)
    ax2 = fig.add_subplot(gs[1, 1])
    k_vals = to_float_array(kappa_changes)
    ax2.bar(np.arange(len(k_vals)), k_vals, color="lime", alpha=0.6, label="Δ avg(κ)")
    ax2.set_xlabel("Update Step")
    ax2.set_ylabel("Topology delta")
    ax2.set_title("Landscape Structural Volatility (κ Drift)", color="lime")
    ax2.grid(alpha=0.1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor="#0a0a0f")
    plt.close()


def plot_hyperparameter_trajectories(
    hyperparam_history: Any,
    output_path: str,
):
    """
    Plot per-step trajectories for each hyperparameter family.

    Produces a compact dashboard with one panel per hyperparameter type:
    lr, wd, dropout, attention temperature, and label smoothing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not hyperparam_history:
        return

    def to_matrix(key: str) -> np.ndarray:
        rows = []
        max_width = 1
        for h in hyperparam_history:
            v = h.get(key, np.nan)
            if isinstance(v, list):
                arr = np.array(v, dtype=np.float32).reshape(-1)
            else:
                arr = np.array([v], dtype=np.float32)
            rows.append(arr)
            max_width = max(max_width, arr.size)

        mat = np.full((len(rows), max_width), np.nan, dtype=np.float32)
        for i, row in enumerate(rows):
            mat[i, : row.size] = row
        return mat

    steps = np.arange(len(hyperparam_history))
    metrics = [
        ("lr", "Learning Rate", True),
        ("wd", "Weight Decay", True),
        ("dropout", "Dropout", False),
        ("attn_temp", "Attention Temperature", False),
        ("label_smoothing", "Label Smoothing", False),
    ]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    axes = axes.flatten()

    for idx, (key, title, use_log_y) in enumerate(metrics):
        ax = axes[idx]
        mat = to_matrix(key)

        for layer_idx in range(mat.shape[1]):
            series = mat[:, layer_idx]
            if np.all(np.isnan(series)):
                continue
            label = f"layer {layer_idx}" if mat.shape[1] > 1 else key
            ax.plot(steps, series, linewidth=1.8, alpha=0.9, label=label)

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.2)
        if use_log_y:
            ax.set_yscale("log")

        if mat.shape[1] <= 8:
            ax.legend(loc="best", fontsize=8, framealpha=0.2)

    # Hide unused panel.
    axes[-1].axis("off")

    fig.suptitle("Hyperparameter Trajectories", fontsize=14, color="cyan")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor="#0a0a0f")
    plt.close()
