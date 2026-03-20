import math

import pytest

pytestmark = pytest.mark.phase4


def _controller():
    try:
        from ta_lbfgs.topology.chain_topo import ChainTopologyController
    except Exception as exc:
        pytest.fail(f"ChainTopologyController import failed. Phase 4 chain topology is missing: {exc}")
    return ChainTopologyController(pivot_sigma=3.0)


def test_chain_default_state():
    c = _controller()
    assert c.current_segment == "reasoning", "Default chain segment must start at reasoning."
    assert c.topology_valid is False, "Topology validity must initialize to False before any observations."


def test_window_scale_mapping():
    c = _controller()
    c.current_segment = "reasoning"
    assert c.window_scale() == 0.5, "Reasoning segment must scale memory window by 0.5."
    c.current_segment = "answer"
    assert c.window_scale() == 1.0, "Answer segment must scale memory window by 1.0."
    c.current_segment = "verify"
    assert c.window_scale() == 0.7, "Verify segment must scale memory window by 0.7."


def test_pivot_detection_fires_at_3sigma():
    c = _controller()
    noisy_history = [1.0 + 0.05 * i for i in range(-5, 5)]
    for g in noisy_history:
        c.on_outer_step(float(g), prm_score=None)

    mu = sum(noisy_history) / len(noisy_history)
    var = sum((x - mu) ** 2 for x in noisy_history) / len(noisy_history)
    sigma = math.sqrt(var)
    spike = mu + 4.0 * sigma
    assert spike > mu + 3.0 * sigma, "Spike precondition invalid: expected spike above 3-sigma pivot threshold."

    c.on_outer_step(spike, prm_score=None)
    assert c.topology_valid, (
        "Chain topology did not mark pivot validity after a guaranteed 4-sigma spike."
    )


def test_chain_no_false_pivot_on_stable_history():
    c = _controller()
    stable_history = [1.0] * 10
    for g in stable_history:
        c.on_outer_step(g, prm_score=None)
    c.on_outer_step(1.0, prm_score=None)
    assert not c.topology_valid, (
        "Chain topology raised a pivot flag on zero-variance stable history."
    )


def test_chain_prm_segment_transition():
    c = _controller()
    for _ in range(8):
        c.on_outer_step(1.0, prm_score=0.2)
    c.on_outer_step(1.0, prm_score=0.9)
    assert c.current_segment in {"reasoning", "answer", "verify"}, (
        "Chain controller produced an invalid segment label outside the contract set."
    )
