import ast
import pathlib

import pytest
import torch

pytestmark = pytest.mark.phase3


@pytest.mark.parametrize(
    "name,expected_group",
    [
        ("model.layers.0.self_attn.rotary_emb.inv_freq", "rope"),
        ("rope_theta", "rope"),
        ("rotary_pos_emb.frequencies", "rope"),
        ("model.embed_tokens.weight", "embedding"),
        ("lm_head.weight", "embedding"),
        ("transformer.wte.weight", "embedding"),
        ("model.layers.0.input_layernorm.weight", "layernorm"),
        ("model.norm.weight", "layernorm"),
        ("ln_f.bias", "layernorm"),
        ("base_model.model.q_proj.lora_A.weight", "lora"),
        ("base_model.model.v_proj.lora_B.weight", "lora"),
        ("model.layers.0.self_attn.q_proj.weight", "standard"),
        ("model.layers.0.mlp.gate_proj.weight", "standard"),
    ],
)
def test_classify_param_group(name, expected_group):
    from ta_lbfgs.core.lbfgs import classify_param_group

    got = classify_param_group(name, None, None)
    assert got == expected_group, (
        f"classify_param_group({name}) returned {got}, expected {expected_group}. "
        "Parameter routing table does not match Phase 3 dispatch contract."
    )


def test_should_freeze_in_inner_loop_table():
    from ta_lbfgs.core.lbfgs import should_freeze_in_inner_loop

    assert should_freeze_in_inner_loop("rope"), "Rope parameters must be frozen in inner loop."
    assert should_freeze_in_inner_loop("embedding"), "Embedding parameters must be frozen in inner loop."
    assert not should_freeze_in_inner_loop("standard"), "Standard parameters must remain trainable in inner loop."


def test_adam_diag_preconditioner_shape_and_finite():
    from ta_lbfgs.core.lbfgs import AdamDiagPreconditioner

    pre = AdamDiagPreconditioner(eps=1e-8, beta2=0.9)
    g = torch.tensor([1.0, -2.0, 0.5])
    out = pre.step(g)
    assert out.shape == g.shape, "AdamDiagPreconditioner changed gradient shape."
    assert torch.isfinite(out).all(), "AdamDiagPreconditioner returned non-finite values."


def test_adam_diag_preconditioner_stabilizes_repeated_gradient():
    from ta_lbfgs.core.lbfgs import AdamDiagPreconditioner

    pre = AdamDiagPreconditioner(eps=1e-8, beta2=0.999)
    g = torch.ones(16)
    out1 = pre.step(g)
    out2 = pre.step(g)
    assert out2.norm() <= out1.norm() + 1e-12, (
        "AdamDiagPreconditioner did not damp repeated-gradient magnitude as expected for EMA second moment updates."
    )


def test_no_item_call_in_inner_training_path():
    src = pathlib.Path("ta_lbfgs/training/inner_loop.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "item" and isinstance(node.value, ast.Name):
            violations.append(node.lineno)
    assert not violations, (
        f".item() calls found in inner loop at lines {violations}. "
        "This risks graph breaks and violates the inner-loop autograd constraint."
    )


def test_inner_loop_freeze_reexport_alignment():
    from ta_lbfgs.training.inner_loop import should_freeze_in_inner_loop as inner_freeze
    from ta_lbfgs.core.lbfgs import should_freeze_in_inner_loop as core_freeze

    for name in ["rope", "embedding", "layernorm", "standard"]:
        assert inner_freeze(name) == core_freeze(name), (
            f"Freeze guard mismatch between training.inner_loop and core.lbfgs for group {name}."
        )


def test_dispatch_categories_complete():
    from ta_lbfgs.core.lbfgs import PARAM_GROUP_TYPES

    expected = {"rope", "embedding", "layernorm", "lora", "standard"}
    got = set(PARAM_GROUP_TYPES)
    assert got == expected, (
        f"PARAM_GROUP_TYPES mismatch. Got {got}, expected {expected}. "
        "Phase 3 dispatch categories are incomplete or inconsistent."
    )
