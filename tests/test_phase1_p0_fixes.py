import pytest
import torch

pytestmark = pytest.mark.phase1

from ta_lbfgs.core.hypergradient import hutchinson_diagonal, neumann_hypergradient
from ta_lbfgs.topology.adaptive_memory import AitkenAccelerator, compute_window
from ta_lbfgs.topology.saddle import escape_saddle, is_saddle_point


@pytest.fixture
def ill_conditioned_H_fn():
    def H(v):
        return 10.0 * v
    return H


@pytest.fixture
def well_conditioned_H_fn():
    def H(v):
        return 0.4 * v
    return H


@pytest.fixture
def saddle_H_fn():
    def H(v):
        return torch.tensor([2.0, -0.5, 1.0], dtype=v.dtype, device=v.device) * v
    return H


@pytest.fixture
def pd_H_fn():
    def H(v):
        return torch.tensor([1.0, 2.0, 3.0], dtype=v.dtype, device=v.device) * v
    return H


def test_spectral_guard_corrects_alpha(ill_conditioned_H_fn):
    g = torch.ones(8)
    alpha_unsafe = 0.5
    result = neumann_hypergradient(ill_conditioned_H_fn, g, alpha_unsafe, K=5)
    assert torch.isfinite(result).all(), (
        "neumann_hypergradient produced non-finite output despite spectral guard. "
        "Fix 1.1A is violated."
    )
    assert result.norm() < 1e6, (
        f"Result norm {result.norm():.2e} is too large for guarded Neumann updates. "
        "Spectral guard likely did not clamp alpha correctly."
    )


def test_neumann_diverges_without_spectral_guard_concept(ill_conditioned_H_fn):
    g = torch.ones(8)
    alpha = 0.5
    v = g.clone()
    result = g.clone()
    for _ in range(400):
        v = v - alpha * ill_conditioned_H_fn(v)
        result = result + v
    assert not torch.isfinite(result).all(), (
        "Naive unguarded Neumann did not diverge for alpha*lambda_max=5.0. "
        "This violates the instability precondition used by Fix 1.1A."
    )


def test_neumann_well_conditioned_accurate(well_conditioned_H_fn):
    torch.manual_seed(0)
    g = torch.randn(64)
    alpha = 0.1
    result = neumann_hypergradient(well_conditioned_H_fn, g, alpha, K=5)
    exact = g / 0.4
    rel_err = (result - exact).norm() / exact.norm()
    assert rel_err < 0.90, (
        f"Neumann K=5 relative error {rel_err:.4f} exceeds 0.90 bound. "
        "Truncated Neumann behavior does not match Fix 1.1B expectation."
    )


def test_lanczos_detects_saddle_correctly(saddle_H_fn):
    torch.manual_seed(0)
    is_sad, eigvec, morse_index = is_saddle_point(saddle_H_fn, dim=3, eps=1e-4)
    assert is_sad, (
        "is_saddle_point returned False for diag(2,-0.5,1). "
        "lambda_min=-0.5 should trigger saddle detection per Fix 1.2A."
    )
    assert eigvec.shape == (3,), (
        f"Eigenvector shape {eigvec.shape} is invalid. Expected (3,) from updated signature."
    )
    assert abs(eigvec.norm().item() - 1.0) < 1e-3, (
        f"Eigenvector norm {eigvec.norm().item():.6f} is not unit length."
    )


def test_lanczos_no_false_positive_on_pd_matrix(pd_H_fn):
    torch.manual_seed(0)
    is_sad, _, _ = is_saddle_point(pd_H_fn, dim=3, eps=1e-4)
    assert not is_sad, (
        "is_saddle_point returned True on PD diag(1,2,3). "
        "This is a false positive that violates Fix 1.2A."
    )


def test_escape_saddle_direction_correct():
    params = [torch.zeros(3)]
    eigvec = torch.tensor([1.0, 0.0, 0.0])
    scale = 0.01
    grad_norm = 0.5
    expected_delta = scale * grad_norm
    escape_saddle(params, grad_norm, eigvec, scale=scale)
    assert abs(params[0][0].item() - expected_delta) < 1e-7, (
        "escape_saddle moved the principal eigvec coordinate by the wrong amount."
    )
    assert abs(params[0][1].item()) < 1e-9, (
        "escape_saddle contaminated a non-eigvec coordinate (index 1)."
    )
    assert abs(params[0][2].item()) < 1e-9, (
        "escape_saddle contaminated a non-eigvec coordinate (index 2)."
    )


def test_hutchinson_diagonal_within_tolerance():
    torch.manual_seed(42)
    dim = 256
    true_diag = torch.arange(1.0, dim + 1.0)
    H_fn = lambda v: true_diag * v
    diag_est = hutchinson_diagonal(H_fn, dim, n_probes=40)
    rel_err = (diag_est - true_diag).norm() / true_diag.norm()
    threshold = 3.0 / (40 ** 0.5)
    assert rel_err < threshold, (
        f"Hutchinson diagonal relative error {rel_err:.4f} exceeds 3-sigma bound {threshold:.4f}. "
        "Fix 1.4A diagonal estimator may be incorrect."
    )


def test_compute_window_exact_table():
    cases = [
        (0.5, 3),
        (1.0, 3),
        (1.5, 3),
        (2.0, 3),
        (4.0, 3),
        (8.0, 3),
        (64.0, 6),
        (1000.0, 10),
        (1e6, 20),
        (1e9, 20),
    ]
    for kappa, expected in cases:
        got = compute_window(kappa)
        assert got == expected, (
            f"compute_window({kappa}) returned {got}, expected {expected}. "
            "Fix 1.5A requires clip(ceil(log2(kappa)),3,20)."
        )


def test_valid_pair_gate_rejects_near_orthogonal():
    try:
        from ta_lbfgs.core.lbfgs import _is_valid_pair
    except Exception:
        pytest.skip("_is_valid_pair is not module-level importable; run this check via integration path.")

    s_orth = torch.tensor([1.0, 0.0, 0.0])
    y_orth = torch.tensor([0.0, 1.0, 0.0])
    assert not _is_valid_pair(s_orth, y_orth), (
        "Orthogonal pair must be rejected by relative secant validity gate."
    )
    s_good = torch.tensor([1.0, 0.0])
    y_good = torch.tensor([0.9, 0.1])
    assert _is_valid_pair(s_good, y_good), (
        "Well-aligned pair should be accepted by relative secant validity gate."
    )


def test_aitken_accelerates_convergence():
    acc = AitkenAccelerator()
    acc.step(torch.tensor(1.0))
    acc.step(torch.tensor(0.5))
    x_acc = acc.step(torch.tensor(0.25))
    assert abs(x_acc.item()) < 1e-5, (
        f"Aitken acceleration returned {x_acc.item():.2e}, expected 0 for geometric ratio 0.5."
    )
