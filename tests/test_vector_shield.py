import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from ai_vector_shield import VectorShield


class TestVectorShield(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        c1 = rng.normal(loc=0.0, scale=0.05, size=(40, 32)) + np.array([1.0] + [0.0] * 31)
        c2 = rng.normal(loc=0.0, scale=0.05, size=(40, 32)) + np.array([0.0, 1.0] + [0.0] * 30)
        c3 = rng.normal(loc=0.0, scale=0.05, size=(40, 32)) + np.array([0.0, 0.0, 1.0] + [0.0] * 29)
        self.X_train = np.vstack([c1, c2, c3]).astype(np.float32)

        self.inliers = np.vstack([
            np.array([1.0, 0.02] + [0.0] * 30),
            np.array([0.01, 1.0] + [0.0] * 30),
        ]).astype(np.float32)

        self.outliers = np.vstack([
            np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 28),
            rng.normal(loc=0.0, scale=1.0, size=(3, 32)),
        ]).astype(np.float32)

    def test_unfitted_raises(self):
        shield = VectorShield()
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            shield.verify(np.zeros((2, 16), dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            shield.inspect(np.zeros((2, 16), dtype=np.float32))

    def test_fit_verify_inliers_and_outliers(self):
        shield = VectorShield(latent_rank=4, n_clusters=3, quantile=0.99).fit(self.X_train)
        inlier_preds = shield.verify(self.inliers)
        outlier_preds = shield.verify(self.outliers)
        self.assertEqual(inlier_preds.dtype, bool)
        self.assertTrue(np.all(inlier_preds))
        self.assertFalse(np.any(outlier_preds))

    def test_inspect_telemetry_schema_and_shadow_mode(self):
        shield = VectorShield(latent_rank=4, n_clusters=3, mode="shadow").fit(self.X_train)
        results = shield.inspect(np.vstack([self.inliers[:1], self.outliers[:1]]))
        self.assertEqual(len(results), 2)

        in_res, out_res = results
        expected_keys = {"allowed", "in_manifold", "action", "residual", "risk_ratio", "chart_id", "latency_ms"}
        for item in results:
            self.assertEqual(set(item.keys()), expected_keys)

        self.assertTrue(in_res["allowed"])
        self.assertEqual(in_res["action"], "allow")
        # In shadow mode, outlier is allowed through but flagged as shadow_flagged
        self.assertTrue(out_res["allowed"])
        self.assertFalse(out_res["in_manifold"])
        self.assertEqual(out_res["action"], "shadow_flagged")
        self.assertGreater(out_res["risk_ratio"], 1.0)

    def test_single_1d_vector_and_torch_tensor_input(self):
        shield = VectorShield(latent_rank=4, n_clusters=3).fit(torch.from_numpy(self.X_train))
        pred_1d = shield.verify(self.inliers[0])
        self.assertEqual(pred_1d.shape, (1,))
        self.assertTrue(bool(pred_1d[0]))

    def test_dimension_mismatch_raises(self):
        shield = VectorShield(latent_rank=4, n_clusters=3).fit(self.X_train)
        wrong_dim = np.zeros((3, 64), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            shield.verify(wrong_dim)

    def test_weights_only_persistence_roundtrip(self):
        shield = VectorShield(latent_rank=4, n_clusters=3).fit(self.X_train)
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt = Path(tmp_dir) / "shield_state.pt"
            shield.save(str(ckpt))
            reloaded = VectorShield.load(str(ckpt))
        test_batch = np.vstack([self.inliers, self.outliers])
        np.testing.assert_array_equal(shield.verify(test_batch), reloaded.verify(test_batch))


if __name__ == "__main__":
    unittest.main()
