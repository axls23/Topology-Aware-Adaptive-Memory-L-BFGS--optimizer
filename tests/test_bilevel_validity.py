import pytest
import torch
import torch.nn as nn

from ta_lbfgs.config import TaLBFGSConfig
from ta_lbfgs.training.bilevel import BilevelOptimizer, BilevelValidationError


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.3], dtype=torch.float32))


def toy_train_fn(model, data, hyperparams, params_override=None):
    w = params_override["w"] if params_override is not None else model.w
    wd = hyperparams.get_layer_wd(0)
    return ((w - 1.0) ** 2).mean() + wd * (w ** 2).mean()


def toy_val_fn(model, data, hyperparams, params_override=None):
    w = params_override["w"] if params_override is not None else model.w
    lr = hyperparams.get_layer_lr(0)
    return ((w + 0.5) ** 2).mean() + 1.0 * lr


def test_bilevel_optimizer_computes_nontrivial_hypergradients():
    cfg = TaLBFGSConfig(
        n_layers=1,
        outer_steps=2,
        inner_steps=3,
        initial_lr=0.05,
        initial_weight_decay=0.01,
        hypergradient_method="CG",
        cg_max_iter=20,
        cg_tol=1e-8,
        device="cpu",
        dtype="float32",
    )

    model = TinyModel()
    optimizer = BilevelOptimizer(cfg)

    result = optimizer.optimize(
        model,
        toy_train_fn,
        toy_val_fn,
        train_data=None,
        val_data=None,
        use_dashboard=False,
        run_validity_checks=True,
    )

    assert len(result["loss_history"]) == cfg.outer_steps
    assert len(result["inner_loss_history"]) == cfg.outer_steps
    assert len(result["grad_magnitude_history"]) == cfg.outer_steps
    assert len(result["validity_checks"]) == 1
    assert len(result["inner_sensitivity_debug"]) == cfg.outer_steps

    check = result["validity_checks"][0]
    assert check["sign_match"] == 1.0
    assert abs(check["fd_slope"]) > 1e-6
    assert abs(check["hypergrad"]) > 1e-8

    sens_debug = result["inner_sensitivity_debug"][0]
    assert "first_suspected_disconnect_step" in sens_debug
    assert len(sens_debug["steps"]) == cfg.inner_steps
    first_step = sens_debug["steps"][0]
    assert "predicted_sensitivity_norm" in first_step
    assert "actual_sensitivity_norm" in first_step
    assert "relative_mismatch" in first_step
    assert torch.isfinite(torch.tensor(first_step["actual_sensitivity_norm"]))


def test_bilevel_optimizer_requires_params_override_signature():
    cfg = TaLBFGSConfig(n_layers=1, outer_steps=1, inner_steps=1, device="cpu")
    model = TinyModel()
    optimizer = BilevelOptimizer(cfg)

    def bad_train_fn(model, data, hyperparams):
        return (model.w ** 2).mean()

    with pytest.raises(BilevelValidationError):
        optimizer.optimize(
            model,
            bad_train_fn,
            toy_val_fn,
            train_data=None,
            val_data=None,
            use_dashboard=False,
        )
