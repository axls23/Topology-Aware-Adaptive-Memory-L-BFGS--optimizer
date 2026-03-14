import numpy as np

from ta_lbfgs.dashboard.online_rsvd import LayerwiseOnlineRSVD


def test_layerwise_online_rsvd_emits_projection_for_monitored_layer():
    projector = LayerwiseOnlineRSVD(
        monitored_layer="block.1",
        n_components=3,
        sketch_dim=16,
        forgetting_factor=0.95,
        seed=7,
    )

    # Non-monitored layer should not emit summary payload.
    out = projector.update("block.0", np.random.randn(64))
    assert out is None

    out = projector.update("block.1", np.random.randn(64))
    assert out is not None
    assert out["layer"] == "block.1"
    assert out["coords"].shape == (3,)
    assert 0.0 <= out["explained_variance_ratio"] <= 1.0


def test_layerwise_online_rsvd_tracks_dominant_subspace_variance():
    rng = np.random.default_rng(123)
    projector = LayerwiseOnlineRSVD(
        monitored_layer="block.0",
        n_components=2,
        sketch_dim=12,
        forgetting_factor=0.97,
        warning_threshold=0.75,
        seed=11,
    )

    # Build a mostly rank-2 stream in a 128D space.
    u1 = rng.standard_normal(128)
    u2 = rng.standard_normal(128)
    u1 /= np.linalg.norm(u1)
    u2 /= np.linalg.norm(u2)

    result = None
    for _ in range(240):
        a, b = rng.normal(0.0, 3.0), rng.normal(0.0, 1.5)
        noise = 0.03 * rng.standard_normal(128)
        x = a * u1 + b * u2 + noise
        result = projector.update("block.0", x)

    assert result is not None
    assert result["explained_variance_ratio"] > 0.75
    assert result["status"] == "Reliable"
