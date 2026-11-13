import numpy as np
import pytest
import torch
from ai_vector_shield import VectorShield


@pytest.fixture
def clustered_corpus():
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=0.0, scale=0.05, size=(40, 32)) + np.array([1.0] + [0.0] * 31)
    c2 = rng.normal(loc=0.0, scale=0.05, size=(40, 32)) + np.array([0.0, 1.0] + [0.0] * 30)
    c3 = rng.normal(loc=0.0, scale=0.05, size=(40, 32)) + np.array([0.0, 0.0, 1.0] + [0.0] * 29)
    X_train = np.vstack([c1, c2, c3]).astype(np.float32)

    inliers = np.vstack([
        np.array([1.0, 0.02] + [0.0] * 30),
        np.array([0.01, 1.0] + [0.0] * 30),
    ]).astype(np.float32)

    outliers = np.vstack([
        np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 28),
        rng.normal(loc=0.0, scale=1.0, size=(3, 32)),
    ]).astype(np.float32)

    return X_train, inliers, outliers


def test_unfitted_raises():
    shield = VectorShield()
    with pytest.raises(RuntimeError, match="not fitted"):
        shield.verify(np.zeros((2, 16), dtype=np.float32))
    with pytest.raises(RuntimeError, match="not fitted"):
        shield.inspect(np.zeros((2, 16), dtype=np.float32))


def test_fit_verify_inliers_and_outliers(clustered_corpus):
    X_train, inliers, outliers = clustered_corpus
    shield = VectorShield(latent_rank=4, n_clusters=3, quantile=0.99).fit(X_train)

    inlier_preds = shield.verify(inliers)
    outlier_preds = shield.verify(outliers)

    assert inlier_preds.dtype == bool
    assert np.all(inlier_preds)
    assert not np.any(outlier_preds)


def test_inspect_telemetry_schema(clustered_corpus):
    X_train, inliers, outliers = clustered_corpus
    shield = VectorShield(latent_rank=4, n_clusters=3).fit(X_train)

    results = shield.inspect(np.vstack([inliers[:1], outliers[:1]]))
    assert len(results) == 2

    in_res, out_res = results
    for item in results:
        assert set(item.keys()) == {"allowed", "residual", "risk_ratio", "chart_id"}
        assert isinstance(item["allowed"], bool)
        assert isinstance(item["chart_id"], int)

    assert in_res["allowed"] is True
    assert in_res["risk_ratio"] <= 1.0
    assert out_res["allowed"] is False
    assert out_res["risk_ratio"] > 1.0


def test_single_1d_vector_and_torch_tensor_input(clustered_corpus):
    X_train, inliers, _ = clustered_corpus
    shield = VectorShield(latent_rank=4, n_clusters=3).fit(torch.from_numpy(X_train))

    pred_1d = shield.verify(inliers[0])
    assert pred_1d.shape == (1,)
    assert bool(pred_1d[0]) is True


def test_dimension_mismatch_raises(clustered_corpus):
    X_train, _, _ = clustered_corpus
    shield = VectorShield(latent_rank=4, n_clusters=3).fit(X_train)

    wrong_dim = np.zeros((3, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="dimension mismatch"):
        shield.verify(wrong_dim)


def test_weights_only_persistence_roundtrip(tmp_path, clustered_corpus):
    X_train, inliers, outliers = clustered_corpus
    shield = VectorShield(latent_rank=4, n_clusters=3).fit(X_train)

    ckpt = tmp_path / "shield_state.pt"
    shield.save(str(ckpt))
    reloaded = VectorShield.load(str(ckpt))

    test_batch = np.vstack([inliers, outliers])
    np.testing.assert_array_equal(shield.verify(test_batch), reloaded.verify(test_batch))
