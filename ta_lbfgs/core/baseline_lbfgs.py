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
        H_diag = state.get("H_diag")

        num_old = len(old_dirs)

        if "rho" not in state:
            state["rho"] = [None] * history_size
            state["alpha"] = [None] * history_size
        rho = state["rho"]
        alpha = state["alpha"]

        # Compute rho_i = 1 / (y_i^T s_i)
        for i in range(num_old):
            rho[i] = 1.0 / old_stps[i].dot(old_dirs[i])

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

        # Compute s_k and y_k
        flat_grad_old = state.get("flat_grad")
        if flat_grad_old is None:
            state["flat_grad"] = flat_grad.clone()
            return {"skipped": True, "ys": 0.0}

        y = flat_grad.sub(flat_grad_old)  # y_k = g_{k+1} - g_k
        s = state.get("d").mul(state.get("t"))  # s_k = alpha_k * d_k

        ys = y.dot(s)  # y_k^T s_k (secant condition)
        result = {"skipped": False, "ys": ys.item()}

        if damping:
            # Powell damping for robustness
            Bs = self.two_loop_recursion(s)  # NOT the inverse — need B*s
            # Simplified: use y directly with damping correction
            sBs = s.dot(Bs) if Bs is not None else s.dot(s)

            if ys < eps * sBs:
                if debug:
                    print(f"Damping applied: ys={ys:.6f}, sBs={sBs:.6f}")
                theta = (1 - eps) * sBs / (sBs - ys)
                y_damped = theta * y + (1 - theta) * Bs
                y = y_damped
                ys = y.dot(s)

        if ys > 1e-10:
            # Accept curvature pair
            if len(old_dirs) == history_size:
                old_dirs.pop(0)
                old_stps.pop(0)
            old_dirs.append(y)
            old_stps.append(s)

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

        if not is_legal(flat_grad):
            state["fail"] = True
            return loss

        # Compute search direction via two-loop recursion
        if len(state["old_dirs"]) == 0:
            d = -flat_grad  # Steepest descent for first iteration
        else:
            d = -self.two_loop_recursion(flat_grad)

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
        self.curvature_update(flat_grad_new, damping=group.get("damping", False))

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

        # Reset rho/alpha caches
        state["rho"] = [None] * new_size
        state["alpha"] = [None] * new_size

    @property
    def history_count(self) -> int:
        """Current number of stored curvature pairs."""
        return len(self.state["global_state"]["old_dirs"])

    @property
    def current_H_diag(self) -> float:
        """Current initial Hessian diagonal scaling."""
        return self.state["global_state"]["H_diag"]
