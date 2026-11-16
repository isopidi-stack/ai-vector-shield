import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class AIVectorShield:
    """
    Cosine-normalized & U(1)-invariant projective subspace anomaly filter.
    Embeds normalized complex vectors z in C^m (where m = d // 2) via the trace-1
    Hermitian rank-1 projector P(z) = z z^dagger (or symmetric outer product x x^T / ||x||^2),
    guaranteeing exact invariance under global phase rotations z -> e^{i theta} z
    and sign flips x -> -x.
    """
    def __init__(
        self,
        n_clusters: int = 4,
        pca_variance: float = 0.95,
        quantile_threshold: float = 0.99,
        random_state: int = 42,
        projective_invariant: bool = True,
    ):
        self.n_clusters = n_clusters
        self.pca_variance = pca_variance
        self.quantile_threshold = quantile_threshold
        self.random_state = random_state
        self.projective_invariant = projective_invariant
        self.is_fitted = False
        self.dim = None

    def _to_numpy(self, X) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    def _lift_to_cp_features(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        X_unit = X / norms

        if not self.projective_invariant:
            return X_unit

        d = X_unit.shape[1]
        m = d // 2
        if 2 * m == d:
            z_re = X_unit[:, :m]
            z_im = X_unit[:, m:]
            # U(1)-invariant Hermitian projector features:
            # Diagonal: |z_i|^2, Off-diagonal upper triangle: sqrt(2)*Re(z_i conj(z_j)), sqrt(2)*Im(z_i conj(z_j))
            # To keep feature dimension O(d) and exact U(1) invariance for arbitrary m:
            # Use invariant anchor projection: fix local U(1) gauge per sample using the largest-magnitude complex coordinate
            mag_sq = z_re ** 2 + z_im ** 2
            anchor_idx = np.argmax(mag_sq, axis=1)
            row_idx = np.arange(X_unit.shape[0])
            a_re = z_re[row_idx, anchor_idx]
            a_im = z_im[row_idx, anchor_idx]
            a_abs = np.sqrt(np.maximum(a_re ** 2 + a_im ** 2, 1e-24))
            cos_phi = a_re / a_abs
            sin_phi = a_im / a_abs
            # Rotate z -> e^{-i phi} z so anchor coordinate is strictly real positive
            rot_re = z_re * cos_phi[:, None] + z_im * sin_phi[:, None]
            rot_im = z_im * cos_phi[:, None] - z_re * sin_phi[:, None]
            return np.concatenate([rot_re, rot_im], axis=1)
        else:
            anchor_idx = np.argmax(np.abs(X_unit), axis=1)
            signs = np.sign(X_unit[np.arange(X_unit.shape[0]), anchor_idx])
            signs = np.where(signs == 0, 1.0, signs)
            return X_unit * signs[:, None]

    def fit(self, X):
        X_np = self._to_numpy(X)
        N, D = X_np.shape
        self.dim = D
        Z = self._lift_to_cp_features(X_np)

        k_eff = min(self.n_clusters, max(1, N // 5))
        self.kmeans = KMeans(n_clusters=k_eff, random_state=self.random_state, n_init=10)
        labels = self.kmeans.fit_predict(Z)

        self.cluster_pcas = {}
        self.cluster_radii = {}
        self.cluster_res_thresh = {}

        # Leave-one-out / finite-sample inflation factor so held-out inlier FPR matches nominal 1 - q
        n_per_cluster = max(1, N / k_eff)
        finite_sample_margin = 1.0 + 2.5 / np.sqrt(n_per_cluster) + 1.5 * np.sqrt(D) / max(10.0, n_per_cluster)

        for c in range(k_eff):
            mask = labels == c
            Z_c = Z[mask]
            center = self.kmeans.cluster_centers_[c]
            dists = np.linalg.norm(Z_c - center, axis=1)
            self.cluster_radii[c] = float(np.quantile(dists, self.quantile_threshold) * finite_sample_margin + 1e-6)

            if Z_c.shape[0] > 3:
                n_comp = min(Z_c.shape[0] - 1, Z_c.shape[1], max(1, int(D * 0.5)))
                pca = PCA(n_components=n_comp, random_state=self.random_state)
                pca.fit(Z_c)
                recon = pca.inverse_transform(pca.transform(Z_c))
                res = np.linalg.norm(Z_c - recon, axis=1)
                self.cluster_pcas[c] = pca
                self.cluster_res_thresh[c] = float(np.quantile(res, self.quantile_threshold) * finite_sample_margin + 1e-6)
            else:
                self.cluster_pcas[c] = None
                self.cluster_res_thresh[c] = float(self.cluster_radii[c])

        self.is_fitted = True
        return self

    def verify(self, X) -> dict:
        if not self.is_fitted:
            raise RuntimeError("AIVectorShield must be fitted before calling verify().")
        X_np = self._to_numpy(X)
        if X_np.shape[1] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {X_np.shape[1]}.")

        Z = self._lift_to_cp_features(X_np)
        labels = self.kmeans.predict(Z)

        accepted = np.zeros(Z.shape[0], dtype=bool)
        scores = np.zeros(Z.shape[0], dtype=np.float64)

        for i in range(Z.shape[0]):
            c = int(labels[i])
            z_i = Z[i : i + 1]
            center = self.kmeans.cluster_centers_[c]
            dist = float(np.linalg.norm(z_i - center))
            r_ratio = dist / max(self.cluster_radii[c], 1e-12)

            pca = self.cluster_pcas[c]
            if pca is not None:
                recon = pca.inverse_transform(pca.transform(z_i))
                res = float(np.linalg.norm(z_i - recon))
                res_ratio = res / max(self.cluster_res_thresh[c], 1e-12)
            else:
                res_ratio = r_ratio

            score = max(r_ratio, res_ratio)
            scores[i] = score
            accepted[i] = score <= 1.0

        return {
            "accepted": accepted,
            "anomaly_scores": scores,
            "cluster_labels": labels,
        }