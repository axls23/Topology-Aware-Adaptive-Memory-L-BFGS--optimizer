import pytest
import torch

pytestmark = pytest.mark.phase2


@pytest.fixture
def banded_attn_matrix():
    T, w = 16, 4
    M = torch.zeros(T, T)
    for i in range(T):
        for j in range(max(0, i - w), min(T, i + w + 1)):
            M[i, j] = 1.0
    M /= M.sum(-1, keepdim=True)
    return M


@pytest.fixture
def sink_attn_matrix():
    T = 16
    M = torch.full((T, T), 0.3 / (T - 1))
    M[:, 0] = 0.7
    M /= M.sum(-1, keepdim=True)
    return M


@pytest.fixture
def attention_builder():
    try:
        from ta_lbfgs.topology.attention_topo import AttentionTopologyBuilder
    except Exception as exc:
        pytest.fail(
            f"AttentionTopologyBuilder import failed. Phase 2 topology integration missing: {exc}"
        )
    return AttentionTopologyBuilder(model=None, window_size=4, warmup_steps=5)


@pytest.fixture
def moe_builder():
    try:
        from ta_lbfgs.topology.moe_topo import MoETopologyBuilder
    except Exception as exc:
        pytest.fail(
            f"MoETopologyBuilder import failed. Phase 2 topology integration missing: {exc}"
        )
    return MoETopologyBuilder(n_experts=8, top_k=2, m_max=20, ttl_expire=5)


def test_attention_topo_api_complete(attention_builder):
    required = {"accumulate_secant", "derive_mask", "classify_head", "hessian_strategy", "should_rederive"}
    missing = required - set(dir(attention_builder))
    assert not missing, (
        f"AttentionTopologyBuilder missing required methods: {missing}. "
        "Phase 2 API contract is incomplete."
    )


def test_classify_head_local_correct(attention_builder, banded_attn_matrix):
    ht = attention_builder.classify_head(0, 0, banded_attn_matrix)
    assert ht == "local", (
        f"classify_head returned {ht} for banded attention. Expected local head classification."
    )


def test_classify_head_sink_correct(attention_builder, sink_attn_matrix):
    ht = attention_builder.classify_head(0, 1, sink_attn_matrix)
    assert ht == "sink", (
        f"classify_head returned {ht} for BOS sink matrix. Expected sink classification."
    )


def test_secant_accumulation_nonnegative(attention_builder):
    torch.manual_seed(7)
    for _ in range(5):
        s, y = torch.randn(16), torch.randn(16)
        attention_builder.accumulate_secant(0, 0, "q", s, y)
    C = attention_builder.secant_accum.get((0, 0, "q"))
    assert C is not None, "secant_accum key (0,0,q) missing after accumulation updates."
    assert (C >= 0).all(), "secant accumulation matrix contains negative values, expected absolute outer products."
    assert C.max() > 0, "secant accumulation matrix is all zeros after random updates."


def test_hessian_strategy_routing(attention_builder, banded_attn_matrix, sink_attn_matrix):
    attention_builder.classify_head(0, 0, banded_attn_matrix)
    attention_builder.classify_head(0, 1, sink_attn_matrix)
    assert attention_builder.hessian_strategy(0, 0) == "banded", "Local head must route to banded strategy."
    assert attention_builder.hessian_strategy(0, 1) == "kfac", "Non-local head must route to kfac strategy."


def test_moe_load_proportional_window(moe_builder):
    moe_builder.load_freq[0] = torch.tensor(0.9)
    moe_builder.load_freq[1] = torch.tensor(0.05)
    m_heavy = moe_builder.window_for_expert(0)
    m_light = moe_builder.window_for_expert(1)
    assert m_heavy >= 15, "Heavy expert should have large buffer window from load-proportional mapping."
    assert m_light <= 5, "Light expert should have near-minimum buffer window from load-proportional mapping."


def test_moe_ttl_expiry_clears_buffer(moe_builder):
    moe_builder.on_forward([0])
    moe_builder.add_pair(0, torch.randn(8), torch.randn(8))
    assert len(moe_builder.get_buffer(0)) == 1, "Expected one curvature pair before TTL expiry."
    moe_builder.ttl[0] = moe_builder.ttl_expire + 1
    cleared = moe_builder.expire_stale()
    assert len(moe_builder.get_buffer(0)) == 0, "Buffer not cleared despite TTL above expiry threshold."
    assert cleared >= 1, "expire_stale should return count of purged expert buffers."


def test_moe_inactive_add_pair_noop(moe_builder):
    moe_builder.ttl[3] = 2
    moe_builder.add_pair(3, torch.randn(8), torch.randn(8))
    assert len(moe_builder.get_buffer(3)) == 0, (
        "add_pair should be no-op for inactive experts to prevent stale curvature injection."
    )


def test_kfac_embedding_precondition_shape_and_scale():
    try:
        from ta_lbfgs.utils.kfac import KFACEmbedding
    except Exception as exc:
        pytest.fail(f"KFACEmbedding import failed. Phase 2 utility integration missing: {exc}")

    torch.manual_seed(13)
    V, d = 100, 32
    kfac = KFACEmbedding(vocab_size=V, embed_dim=d)
    for _ in range(10):
        freqs = torch.rand(d).abs() + 0.1
        grad_out = torch.randn(d, d)
        kfac.update(freqs, grad_out)

    raw_grad = torch.randn(d, d)
    precond = kfac.inverse_precondition(raw_grad)
    assert precond.shape == raw_grad.shape, "KFAC preconditioned gradient shape mismatch."
    assert torch.isfinite(precond).all(), "KFAC inverse_precondition produced non-finite values."
    assert precond.norm() < raw_grad.norm() * 100, "KFAC preconditioner amplified gradient beyond stability bound."
