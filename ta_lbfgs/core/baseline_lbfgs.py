"""
Baseline L-BFGS Implementation.

Adapted from hjmshi/PyTorch-LBFGS (https://github.com/hjmshi/PyTorch-LBFGS).
Provides the core two-loop recursion, curvature updates, and line search
logic that the LayerwiseTaLBFGS optimizer builds upon.

Original Authors: Hao-Jun Michael Shi and Dheevatsa Mudigere
Adapted for ta-LBFGS: Block-diagonal layerwise usage.
"""

import torch
import numpy as np
from functools import reduce
from copy import deepcopy
from torch.optim import Optimizer
from typing import Dict, Optional, Tuple


def is_legal(v: torch.Tensor) -> bool:
    """Check that tensor contains no NaN or Inf values."""
    return not torch.isnan(v).any() and not torch.isinf(v).any()


def polyinterp(points, x_min_bound=None, x_max_bound=None):
    """
    Polynomial interpolation for line search step size selection.

    Gives the minimizer and minimum of the interpolating polynomial over
    given points based on function and derivative information.
    Falls back to bisection if no valid critical points exist.

    Based on polyinterp.m from Mark Schmidt's minFunc.

    Args:
        points: Array of shape [N, 3] with columns [x, f(x), f'(x)].
                Use np.nan for unknown f or f' values.
        x_min_bound: Minimum bracket bound (default: min of points).
        x_max_bound: Maximum bracket bound (default: max of points).

    Returns:
        x_sol: Minimizer of interpolating polynomial.
    """
    no_points = points.shape[0]
    order = np.sum(1 - np.isnan(points[:, 1:3]).astype("int")) - 1

    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])

    if x_min_bound is None:
        x_min_bound = x_min
    if x_max_bound is None:
        x_max_bound = x_max

    eps = 1e-12

    # Explicit quadratic interpolation
    if no_points == 2 and order == 2:
        if points[0, 0] == 0:
            denom = 2 * (points[1, 1] - points[0, 1] - points[0, 2] * points[1, 0])
            if abs(denom) < eps:
                x_sol = (x_max_bound + x_min_bound) / 2
            else:
                x_sol = -points[0, 2] * points[1, 0] ** 2 / denom
        else:
            dx = points[0, 0] - points[1, 0]
            if abs(dx) < eps:
                x_sol = (x_max_bound + x_min_bound) / 2
            else:
                a = -(
                    points[0, 1]
                    - points[1, 1]
                    - points[0, 2] * dx
                ) / (dx ** 2)
                if abs(a) < eps:
                    x_sol = (x_max_bound + x_min_bound) / 2
                else:
                    x_sol = points[0, 0] - points[0, 2] / (2 * a)

        x_sol = np.minimum(np.maximum(x_min_bound, x_sol), x_max_bound)

    # Explicit cubic interpolation
    elif no_points == 2 and order == 3:
        dx = points[0, 0] - points[1, 0]
        if abs(dx) < eps:
            x_sol = (x_max_bound + x_min_bound) / 2
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                d1 = points[0, 2] + points[1, 2] - 3 * ((points[0, 1] - points[1, 1]) / dx)
                disc = d1 ** 2 - points[0, 2] * points[1, 2]

            if not np.isfinite(d1) or disc < 0:
                x_sol = (x_max_bound + x_min_bound) / 2
            else:
                d2 = np.sqrt(disc)
                denom = points[1, 2] - points[0, 2] + 2 * d2
                if abs(denom) < eps or not np.isfinite(denom):
                    x_sol = (x_max_bound + x_min_bound) / 2
                else:
                    x_sol = points[1, 0] - (points[1, 0] - points[0, 0]) * (
                        (points[1, 2] + d2 - d1) / denom
                    )
                    x_sol = np.minimum(np.maximum(x_min_bound, x_sol), x_max_bound)

    # General polynomial via linear system
    else:
        A = np.zeros((0, order + 1))
        b = np.zeros((0, 1))

        for i in range(no_points):
            if not np.isnan(points[i, 1]):
                constraint = np.zeros((1, order + 1))
                for j in range(order, -1, -1):
                    constraint[0, order - j] = points[i, 0] ** j
                A = np.append(A, constraint, 0)
                b = np.append(b, points[i, 1])

        for i in range(no_points):
            if not np.isnan(points[i, 2]):
                constraint = np.zeros((1, order + 1))
                for j in range(order):
                    constraint[0, j] = (order - j) * points[i, 0] ** (order - j - 1)
                A = np.append(A, constraint, 0)
                b = np.append(b, points[i, 2])

        if A.shape[0] != A.shape[1] or np.linalg.matrix_rank(A) != A.shape[0]:
            x_sol = (x_min_bound + x_max_bound) / 2
        else:
            coeff = np.linalg.solve(A, b)
            dcoeff = np.zeros(order)
            for i in range(len(coeff) - 1):
                dcoeff[i] = coeff[i] * (order - i)

            crit_pts = np.array([x_min_bound, x_max_bound])
            crit_pts = np.append(crit_pts, points[:, 0])

            if not np.isinf(dcoeff).any():
                roots = np.roots(dcoeff)
                crit_pts = np.append(crit_pts, roots)

            f_min = np.inf
            x_sol = (x_min_bound + x_max_bound) / 2
            for crit_pt in crit_pts:
                if np.isreal(crit_pt) and x_min_bound <= crit_pt <= x_max_bound:
                    F_cp = np.polyval(coeff, crit_pt)
                    if np.isreal(F_cp) and F_cp < f_min:
                        x_sol = np.real(crit_pt)
                        f_min = np.real(F_cp)

    if not np.isfinite(x_sol):
        return (x_max_bound + x_min_bound) / 2
    return x_sol


class FullBatchLBFGS(Optimizer):
    """
    L-BFGS optimizer adapted from hjmshi/PyTorch-LBFGS.

    Supports Armijo backtracking and Wolfe bracketing line searches,
    Powell damping for curvature pair stability, and configurable
    history size.

    This class is used as the **per-block engine** inside the
    LayerwiseTaLBFGS wrapper. Each transformer layer block gets its
    own FullBatchLBFGS instance with an independent history buffer.

    Args:
        params: Iterable of parameters to optimize.
        lr: Step length / learning rate (default: 1.0).
        history_size: Number of curvature pairs to store (default: 10).
        line_search: Line search strategy ('Wolfe', 'Armijo', 'None').
        dtype: Torch data type (default: torch.float).
        debug: Enable debug logging (default: False).
    """

    def __init__(
        self,
        params,
        lr=1.0,
        history_size=10,
        line_search="Wolfe",
        dtype=torch.float,
        debug=False,
        damping=False,
        damping_eps=1e-2,
        curvature_threshold=0.2,
        secant_topology_enabled=False,
        secant_topology_warmup_steps=10,
        secant_topology_top_k=16,
        secant_topology_percentile=95.0,
        secant_symmetrize_enabled=True,
        secant_symmetry_assert_enabled=True,
        use_spectral_scaler=True,
        spectral_mu=0.2,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if history_size < 0:
            raise ValueError(f"Invalid history size: {history_size}")
        if line_search not in ["Armijo", "Wolfe", "None"]:
            raise ValueError(f"Invalid line search: {line_search}")

        defaults = dict(
            lr=lr,
            history_size=history_size,
            line_search=line_search,
            dtype=dtype,
            debug=debug,
            damping=damping,
            damping_eps=damping_eps,
            curvature_threshold=curvature_threshold,
            secant_topology_enabled=secant_topology_enabled,
            secant_topology_warmup_steps=secant_topology_warmup_steps,
            secant_topology_top_k=secant_topology_top_k,
            secant_topology_percentile=secant_topology_percentile,
            secant_symmetrize_enabled=secant_symmetrize_enabled,
            secant_symmetry_assert_enabled=secant_symmetry_assert_enabled,
            use_spectral_scaler=use_spectral_scaler,
            spectral_mu=spectral_mu,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "FullBatchLBFGS doesn't support per-parameter options "
                "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

        state = self.state["global_state"]
        state.setdefault("n_iter", 0)
        state.setdefault("curv_skips", 0)
        state.setdefault("fail_skips", 0)
        state.setdefault("H_diag", 1)
        state.setdefault("fail", True)

        state["old_dirs"] = []  # y vectors (gradient differences)
        state["old_stps"] = []  # s vectors (parameter differences)
        state["old_alpha_scales"] = []
        state["topology_mask_indices"] = None
        state["topology_num_params"] = 0
        state["secant_topology_enabled"] = bool(secant_topology_enabled)
        state["secant_topology_warmup_steps"] = int(max(1, secant_topology_warmup_steps))
        state["secant_topology_top_k"] = int(max(1, secant_topology_top_k))
        state["secant_topology_percentile"] = float(secant_topology_percentile)
        state["secant_symmetrize_enabled"] = bool(secant_symmetrize_enabled)
        state["secant_symmetry_assert_enabled"] = bool(secant_symmetry_assert_enabled)
        state["use_spectral_scaler"] = bool(use_spectral_scaler)
        state["spectral_mu"] = float(spectral_mu)
        state["secant_topology_num_params"] = int(self._numel())
        state["secant_warmup_count"] = 0
        state["secant_edge_scores"] = {}
        state["secant_mask_coo"] = None
        state["secant_active_mask"] = None

    def _update_secant_covariance(
        self,
        s_dense: torch.Tensor,
        y_dense: torch.Tensor,
    ) -> None:
        """
        Accumulate secant-derived topology estimator C_secant over warmup.

        Mask application follows Report Section 4.1 Eq. C_secant.
        """
        state = self.state["global_state"]
        if not state.get("secant_topology_enabled", False):
            return

        warmup_steps = state.get("secant_topology_warmup_steps", 10)
        if state.get("secant_warmup_count", 0) >= warmup_steps:
            return

        abs_s = s_dense.abs()
        abs_y = y_dense.abs()
        n = int(abs_s.numel())
        top_k = min(max(1, state.get("secant_topology_top_k", 16)), n)

        s_idx = torch.topk(abs_s, k=top_k).indices.tolist()
        y_idx = torch.topk(abs_y, k=top_k).indices.tolist()
        active = sorted(set(s_idx + y_idx))
        if not active:
            return

        active_idx = torch.tensor(active, device=s_dense.device, dtype=torch.long)
        s_k = abs_s[active_idx]
        y_k = abs_y[active_idx]

        # Symmetric secant estimator: C += (|s y^T| + |s y^T|^T) / 2
        outer_product = torch.abs(torch.outer(s_k, y_k))
        if state.get("secant_symmetrize_enabled", True):
            c_step = 0.5 * (outer_product + outer_product.transpose(0, 1))
        else:
            c_step = outer_product

        edge_scores: Dict[Tuple[int, int], float] = state.get("secant_edge_scores", {})
        c_cpu = c_step.detach().cpu()
        for i_local, i_global in enumerate(active):
            row = c_cpu[i_local]
            for j_local, j_global in enumerate(active):
                val = float(row[j_local].item())
                if val <= 0.0:
                    continue
                a, b = (int(i_global), int(j_global)) if i_global <= j_global else (int(j_global), int(i_global))
                edge_scores[(a, b)] = edge_scores.get((a, b), 0.0) + val

        state["secant_edge_scores"] = edge_scores
        state["secant_warmup_count"] = state.get("secant_warmup_count", 0) + 1

        if state["secant_warmup_count"] >= warmup_steps:
            self._finalize_secant_topology_mask(device=s_dense.device, dtype=s_dense.dtype)

    def _finalize_secant_topology_mask(self, device: torch.device, dtype: torch.dtype) -> None:
        """Build sparse adjacency mask M from warmup secant covariance scores."""
        state = self.state["global_state"]
        edge_scores: Dict[Tuple[int, int], float] = state.get("secant_edge_scores", {})
        n = int(state.get("secant_topology_num_params", self._numel()))
        W = max(1, int(state.get("secant_warmup_count", 1)))

        if not edge_scores:
            state["secant_mask_coo"] = None
            state["secant_active_mask"] = None
            return

        # Normalize C_secant by warmup count W before thresholding.
        for key in list(edge_scores.keys()):
            edge_scores[key] = edge_scores[key] / float(W)

        if state.get("secant_symmetry_assert_enabled", True):
            active_nodes = sorted({i for (i, j) in edge_scores.keys()} | {j for (i, j) in edge_scores.keys()})
            if active_nodes:
                compact_idx = {node: idx for idx, node in enumerate(active_nodes)}
                C_secant = torch.zeros((len(active_nodes), len(active_nodes)), dtype=torch.float32)
                for (i, j), value in edge_scores.items():
                    ii = compact_idx[i]
                    jj = compact_idx[j]
                    C_secant[ii, jj] = float(value)
                    C_secant[jj, ii] = float(value)
                assert torch.allclose(C_secant, C_secant.T, atol=1e-6), (
                    "C_secant must be symmetric before mask derivation"
                )

        keys = list(edge_scores.keys())
        vals = torch.tensor([edge_scores[k] for k in keys], dtype=torch.float32)
        q = max(0.0, min(1.0, state.get("secant_topology_percentile", 95.0) / 100.0))
        tau = torch.quantile(vals, q=q)
        keep = vals >= tau
        if not torch.any(keep):
            keep[torch.argmax(vals)] = True

        kept_keys = [keys[i] for i, flag in enumerate(keep.tolist()) if flag]
        idx = torch.tensor(kept_keys, dtype=torch.long, device=device).t().contiguous()
        values = torch.ones(idx.size(1), dtype=dtype, device=device)
        adj = torch.sparse_coo_tensor(idx, values, size=(n, n), device=device).coalesce()
        adj = (adj + adj.transpose(0, 1)).coalesce()

        active_idx = torch.unique(adj.indices().reshape(-1))
        active_mask = torch.zeros(n, dtype=dtype, device=device)
        active_mask[active_idx] = 1.0

        state["secant_mask_coo"] = adj
        state["secant_active_mask"] = active_mask
        state["secant_edge_scores"] = {}

    @staticmethod
    def compute_alpha_k(
        s_tilde_k: torch.Tensor,
        y_hat_k: torch.Tensor,
        B_k_s: torch.Tensor,
        mu: float = 0.2,
    ) -> torch.Tensor:
        """
        Dynamic spectral scaler alpha_k.

        Interpolates toward B_k on non-convex cliffs.
        See Report Section 4.3 and rho_k^alpha definition.
        Powell damping reference: Powell (1978), Numerical Analysis 144-157.
        """
        raw_curvature = s_tilde_k.dot(y_hat_k)
        predicted_curvature = s_tilde_k.dot(B_k_s)

        if raw_curvature >= mu * predicted_curvature:
            return torch.tensor(1.0, device=raw_curvature.device, dtype=raw_curvature.dtype)

        denom = (predicted_curvature - raw_curvature) + 1e-10
        alpha_k = ((1.0 - mu) * predicted_curvature) / denom
        return torch.clamp(alpha_k, min=0.0, max=1.0)

    @staticmethod
    def compute_rho_alpha(
        alpha_k: torch.Tensor,
        s_tilde_k: torch.Tensor,
        y_hat_k: torch.Tensor,
        G: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute rho_k^alpha under Euclidean or metric-weighted inner product."""
        if G is None:
            inner_product = s_tilde_k.dot(y_hat_k)
        else:
            inner_product = s_tilde_k.dot(torch.mv(G, y_hat_k))
        return alpha_k / (inner_product + 1e-8)

    def _project_secant_pair(
        self,
        s_dense: torch.Tensor,
        y_dense: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply secant topology mask projection to incoming (s, y) pairs after warmup."""
        state = self.state["global_state"]
        if not state.get("secant_topology_enabled", False):
            return s_dense, y_dense

        if state.get("secant_warmup_count", 0) <= state.get("secant_topology_warmup_steps", 10):
            return s_dense, y_dense

        mask = state.get("secant_active_mask")
        if mask is None:
            return s_dense, y_dense

        if mask.device != s_dense.device:
            mask = mask.to(device=s_dense.device, dtype=s_dense.dtype)
        else:
            mask = mask.to(dtype=s_dense.dtype)

        return s_dense * mask, y_dense * mask

    def set_topology_mask(self, edge_indices: Optional[torch.Tensor], num_params: int):
        """Set sparse topology mask as COO edge indices of shape [2, E]."""
        state = self.state["global_state"]
        if edge_indices is None:
            state["topology_mask_indices"] = None
            state["topology_num_params"] = 0
            return

        if edge_indices.dim() != 2 or edge_indices.size(0) != 2:
            raise ValueError("edge_indices must have shape [2, E]")

        state["topology_mask_indices"] = edge_indices.long()
        state["topology_num_params"] = int(num_params)

    def _compress_with_topology(self, vec: torch.Tensor) -> torch.Tensor:
        """Project dense parameter vector into edge-space features."""
        state = self.state["global_state"]
        edge_idx = state.get("topology_mask_indices")
        if edge_idx is None or edge_idx.numel() == 0:
            return vec

        u = edge_idx[0].to(vec.device)
        v = edge_idx[1].to(vec.device)
        return 0.5 * (vec[u] + vec[v])

    def _expand_from_topology(self, edge_vec: torch.Tensor, target_numel: int) -> torch.Tensor:
        """Lift edge-space vector back to dense parameter vector by edge scatter averaging."""
        state = self.state["global_state"]
        edge_idx = state.get("topology_mask_indices")
        if edge_idx is None or edge_idx.numel() == 0:
            return edge_vec

        out = torch.zeros(target_numel, device=edge_vec.device, dtype=edge_vec.dtype)
        counts = torch.zeros(target_numel, device=edge_vec.device, dtype=edge_vec.dtype)

        u = edge_idx[0].to(edge_vec.device)
        v = edge_idx[1].to(edge_vec.device)

        out.index_add_(0, u, edge_vec)
        out.index_add_(0, v, edge_vec)
        ones = torch.ones_like(edge_vec)
        counts.index_add_(0, u, ones)
        counts.index_add_(0, v, ones)

        mask = counts > 0
        out[mask] = out[mask] / counts[mask]
        return out

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0
            )
        return self._numel_cache

    def _gather_flat_grad(self):
        """Flatten and concatenate all parameter gradients."""
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_update(self, step_size, update):
        """Apply update to parameters."""
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.add_(update[offset : offset + numel].view_as(p.data), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _copy_params(self):
        """Deep copy current parameters."""
        return [deepcopy(p.data) for p in self._params]

    def _load_params(self, current_params):
        """Restore parameters from a saved copy."""
        for param, saved in zip(self._params, current_params):
            param.data[:] = saved

    def two_loop_recursion(self, vec):
        """
        L-BFGS two-loop recursion: compute H_k * vec.

        This is the core of the L-BFGS algorithm. It approximates
        the product of the inverse Hessian with a vector using only
        stored curvature pairs (s, y).

        Sparse/topology-aware variant follows Report Section 5.2.
        Reference: Nocedal (1980), Mathematics of Computation 35(151).

        Args:
            vec: 1-D gradient tensor to apply H_k inverse to.

        Returns:
            r: Search direction tensor (-H_k * g_k).
        """
        group = self.param_groups[0]
        history_size = group["history_size"]

        state = self.state["global_state"]
        old_dirs = state.get("old_dirs")  # y vectors
        old_stps = state.get("old_stps")  # s vectors
        old_alpha_scales = state.get("old_alpha_scales", [])
        H_diag = state.get("H_diag")

        num_old = len(old_dirs)

        if "rho" not in state:
            state["rho"] = [None] * history_size
            state["alpha"] = [None] * history_size
        rho = state["rho"]
        alpha = state["alpha"]

        # Compute rho_i^alpha = alpha_i / <s_i, y_i>.
        for i in range(num_old):
            denom = old_stps[i].dot(old_dirs[i])
            if torch.abs(denom) < 1e-12:
                denom = torch.tensor(1e-12, device=denom.device, dtype=denom.dtype)
            alpha_scale = old_alpha_scales[i] if i < len(old_alpha_scales) else 1.0
            rho[i] = alpha_scale / denom

        q = vec.clone()

        # First loop: newest → oldest
        for i in range(num_old - 1, -1, -1):
            alpha[i] = old_dirs[i].dot(q) * rho[i]
            q.add_(old_stps[i], alpha=-alpha[i])

        # Multiply by initial Hessian approximation H_0 = H_diag * I
        r = torch.mul(q, H_diag)

        # Second loop: oldest → newest
        for i in range(num_old):
            beta = old_stps[i].dot(r) * rho[i]
            r.add_(old_dirs[i], alpha=alpha[i] - beta)

        return r

    def approximate_Bs(self, s, old_dirs, old_stps, H_diag):
        """
        Lightweight direct-Hessian action approximation B_k s.

        Uses a diagonal base B_0 plus secant-informed corrections from history.
        """
        inv_h = 1.0 / (H_diag + 1e-12)
        Bs = inv_h * s.clone()

        for s_i, y_i in zip(old_stps, old_dirs):
            s_i_dot_s_i = s_i.dot(s_i) + 1e-12
            proj = s_i.dot(s) / s_i_dot_s_i
            Bs = Bs + proj * (y_i - inv_h * s_i)

        return Bs

    def curvature_update(self, flat_grad, eps=1e-2, damping=False):
        """
        Update curvature pair history with new gradient information.

        Implements the secant equation update with optional Powell damping
        for numerical stability in non-convex landscapes.

        Args:
            flat_grad: Current flattened gradient vector.
            eps: Threshold for curvature pair acceptance / damping.
            damping: Whether to use Powell damping.

        Returns:
            dict with 'skipped' (bool) and 'ys' (float) for monitoring.
        """
        group = self.param_groups[0]
        history_size = group["history_size"]
        debug = group["debug"]

        state = self.state["global_state"]
        old_dirs = state.get("old_dirs")
        old_stps = state.get("old_stps")
        old_alpha_scales = state.setdefault("old_alpha_scales", [])
        mu = float(state.get("spectral_mu", group.get("curvature_threshold", 0.2)))

        # Compute s_k and y_k
        flat_grad_old = state.get("flat_grad")
        if flat_grad_old is None:
            state["flat_grad"] = flat_grad.clone()
            return {"skipped": True, "ys": 0.0}

        y_dense = flat_grad.sub(flat_grad_old)  # y_k = g_{k+1} - g_k
        s_dense = state.get("d").mul(state.get("t"))  # s_k = alpha_k * d_k

        self._update_secant_covariance(s_dense, y_dense)
        s_dense, y_dense = self._project_secant_pair(s_dense, y_dense)

        y = self._compress_with_topology(y_dense)
        s = self._compress_with_topology(s_dense)

        ys = y.dot(s)  # y_k^T s_k (secant condition)
        result = {"skipped": False, "ys": ys.item()}

        H_diag = state.get("H_diag", torch.tensor(1.0, device=s.device, dtype=s.dtype))
        if not torch.is_tensor(H_diag):
            H_diag = torch.tensor(float(H_diag), device=s.device, dtype=s.dtype)
        Bs = self.approximate_Bs(s, old_dirs, old_stps, H_diag)
        sBs = s.dot(Bs)

        if damping:
            # Powell damping with direct-Hessian action estimate B_k s.
            if ys < mu * sBs:
                if debug:
                    print(f"Damping applied: ys={ys:.6f}, sBs={sBs:.6f}")
                denom = (sBs - ys) + 1e-12
                theta = (1 - mu) * sBs / denom
                theta = torch.clamp(theta, min=0.0, max=1.0)
                y_damped = theta * y + (1 - theta) * Bs
                y = y_damped
                ys = y.dot(s)

        if state.get("use_spectral_scaler", True):
            alpha_k = self.compute_alpha_k(s, y, Bs, mu=mu)
            _ = self.compute_rho_alpha(alpha_k, s, y, G=None)
        else:
            alpha_k = torch.tensor(1.0, device=ys.device, dtype=ys.dtype)

        if ys > 1e-10:
            # Accept curvature pair
            if len(old_dirs) == history_size:
                old_dirs.pop(0)
                old_stps.pop(0)
                old_alpha_scales.pop(0)
            old_dirs.append(y)
            old_stps.append(s)
            alpha_scalar = float(alpha_k.item())
            old_alpha_scales.append(alpha_scalar)
            if alpha_scalar < -1e-8 or alpha_scalar > 1.0 + 1e-8:
                raise RuntimeError(f"alpha_k out of range [0, 1]: {alpha_scalar:.6f}")

            # Update initial Hessian scaling: H_0 = (y^T s) / (y^T y)
            state["H_diag"] = ys / y.dot(y)
        else:
            # Reject pair — secant condition violated (possible saddle region)
            state["curv_skips"] += 1
            result["skipped"] = True

        state["flat_grad"] = flat_grad.clone()
        return result

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Loss value from closure (if provided).
        """
        if closure is None:
            raise ValueError("FullBatchLBFGS requires a closure for step()")

        group = self.param_groups[0]
        lr = group["lr"]
        line_search = group["line_search"]
        state = self.state["global_state"]

        # Evaluate initial loss and gradient
        loss = closure()
        flat_grad = self._gather_flat_grad()
        grad_comp = self._compress_with_topology(flat_grad)

        if not is_legal(flat_grad):
            state["fail"] = True
            return loss

        # Compute search direction via two-loop recursion
        if len(state["old_dirs"]) == 0:
            d_comp = -grad_comp  # Steepest descent for first iteration
        else:
            d_comp = -self.two_loop_recursion(grad_comp)

        d = self._expand_from_topology(d_comp, flat_grad.numel())

        # Store direction and gradient for curvature update
        state["d"] = d
        state["flat_grad_prev"] = flat_grad.clone()

        # Compute directional derivative
        gtd = flat_grad.dot(d)

        if gtd > 0:
            # Not a descent direction — fall back to steepest descent
            d = -flat_grad
            gtd = flat_grad.dot(d)

        state["t"] = torch.tensor(lr, dtype=flat_grad.dtype)

        if line_search == "None":
            # Fixed step size
            self._add_update(lr, d)
            loss = closure()
            flat_grad_new = self._gather_flat_grad()
        elif line_search == "Armijo":
            loss, flat_grad_new = self._armijo_line_search(
                closure, d, flat_grad, loss, gtd
            )
        else:  # Wolfe
            loss, flat_grad_new = self._wolfe_line_search(
                closure, d, flat_grad, loss, gtd
            )

        # Update curvature pairs
        self.curvature_update(
            flat_grad_new,
            eps=group.get("damping_eps", 1e-2),
            damping=group.get("damping", False),
        )

        state["n_iter"] += 1
        state["fail"] = False

        return loss

    def _armijo_line_search(self, closure, d, g, f0, gtd, c1=1e-4, max_ls=25):
        """Armijo backtracking line search."""
        t = 1.0
        for _ in range(max_ls):
            self._add_update(t, d)
            f_new = closure()
            g_new = self._gather_flat_grad()

            if f_new <= f0 + c1 * t * gtd:
                self.state["global_state"]["t"] = torch.tensor(t)
                return f_new, g_new

            # Backtrack
            self._add_update(-t, d)  # undo
            t *= 0.5

        # Failed — accept last step
        self._add_update(t, d)
        self.state["global_state"]["t"] = torch.tensor(t)
        return closure(), self._gather_flat_grad()

    def _wolfe_line_search(self, closure, d, g, f0, gtd, c1=1e-4, c2=0.9, max_ls=25):
        """Armijo-Wolfe bracketing line search."""
        t = 1.0
        f_prev = f0
        gtd_prev = gtd
        done = False
        bracket_low = 0.0
        bracket_high = float("inf")
        best_t = 0.0
        best_f = f0
        best_g = g

        for ls_iter in range(max_ls):
            t_curr = t
            self._add_update(t_curr, d)
            f_new = closure()
            g_new = self._gather_flat_grad()
            gtd_new = g_new.dot(d)

            if not is_legal(g_new) or not torch.isfinite(f_new):
                # Undo unstable trial and shrink step aggressively.
                self._add_update(-t_curr, d)
                bracket_high = min(bracket_high, t_curr)
                t = max(1e-12, 0.5 * (bracket_low + bracket_high) if bracket_high < float("inf") else 0.5 * t_curr)
                continue

            if f_new < best_f:
                best_t = t_curr
                best_f = f_new
                best_g = g_new

            if f_new > f0 + c1 * t_curr * gtd:
                # Armijo violated — bracket
                bracket_high = t_curr
                t_next = polyinterp(
                    np.array(
                        [
                            [bracket_low, f_prev.item() if hasattr(f_prev, 'item') else f_prev, gtd_prev.item() if hasattr(gtd_prev, 'item') else gtd_prev], 
                            [t_curr, f_new.item() if hasattr(f_new, 'item') else f_new, gtd_new.item() if hasattr(gtd_new, 'item') else gtd_new]
                        ]
                    ),
                    x_min_bound=bracket_low,
                    x_max_bound=bracket_high,
                )
                # Undo trial so each iteration starts from the same base params.
                self._add_update(-t_curr, d)
                if not np.isfinite(t_next):
                    t_next = 0.5 * (bracket_low + bracket_high)
                t = float(np.clip(t_next, max(1e-12, bracket_low + 1e-12), max(1e-12, bracket_high - 1e-12)))
                continue

            if abs(gtd_new) <= -c2 * gtd:
                # Strong Wolfe satisfied
                done = True
                best_t = t_curr
                best_f = f_new
                best_g = g_new
                break

            if gtd_new >= 0:
                bracket_high = t_curr
            else:
                bracket_low = t_curr
                f_prev = f_new
                gtd_prev = gtd_new

            if bracket_high < float("inf"):
                t_next = 0.5 * (bracket_low + bracket_high)
            else:
                t_next = 2.0 * t_curr

            # Undo trial so each iteration starts from the same base params.
            self._add_update(-t_curr, d)
            t = max(1e-12, float(t_next))

        if not done:
            # Apply the best valid point seen during search; if none improved,
            # take a tiny conservative step to keep iteration progress stable.
            t_apply = best_t if best_t > 0 else min(1e-3, max(1e-12, t))
            self._add_update(t_apply, d)
            f_new = closure()
            g_new = self._gather_flat_grad()
            t = t_apply

        self.state["global_state"]["t"] = torch.tensor(t)
        return f_new, g_new

    def resize_history(self, new_size: int):
        """
        Dynamically resize the history buffer (adaptive memory).

        Preserves the most recent curvature pairs when shrinking.

        Args:
            new_size: New maximum history size.
        """
        group = self.param_groups[0]
        state = self.state["global_state"]
        old_size = group["history_size"]

        if new_size == old_size:
            return

        group["history_size"] = new_size

        # Trim if shrinking
        if new_size < len(state["old_dirs"]):
            excess = len(state["old_dirs"]) - new_size
            state["old_dirs"] = state["old_dirs"][excess:]
            state["old_stps"] = state["old_stps"][excess:]
            state["old_alpha_scales"] = state["old_alpha_scales"][excess:]

        # Reset rho/alpha caches
        state["rho"] = [None] * new_size
        state["alpha"] = [None] * new_size

    def clear_history(self):
        """Clear stored curvature pairs and related caches."""
        state = self.state["global_state"]
        history_size = self.param_groups[0]["history_size"]
        state["old_dirs"] = []
        state["old_stps"] = []
        state["old_alpha_scales"] = []
        state["rho"] = [None] * history_size
        state["alpha"] = [None] * history_size
        state["flat_grad"] = None
        state["flat_grad_prev"] = None

    @property
    def history_count(self) -> int:
        """Current number of stored curvature pairs."""
        return len(self.state["global_state"]["old_dirs"])

    @property
    def current_H_diag(self) -> float:
        """Current initial Hessian diagonal scaling."""
        return self.state["global_state"]["H_diag"]
