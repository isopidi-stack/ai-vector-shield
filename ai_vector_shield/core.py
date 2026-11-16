import numpy as np
import torch
from sklearn.cluster import KMeans

class AIVectorShield:
    """
    Projective C P^{d-1} / S^{d-1} Subspace Anomaly Shield with exact U(1) phase
    and sign-flip gauge canonicalization and per-cluster tangent SVD projection.
    """
    def __init__(self, dim: int, latent_rank: int = 8, n_clusters: int = 4, quantile: float = 0.99, random_state: int = 42):
        self.dim = dim
        self.latent_rank = latent_rank
        self.n_clusters = n_clusters
        self.quantile = quantile
        self.random_state = random_state
        self.is_fitted = False
        self.centroids = None
        self. projectors = None
        self.radii = None
        self.res_thresholds = None

    @staticmethod
    def _canonicalize_u1_gauge(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        U = X / norms
        d = U.shape[1]
        m = d // 2
        if 2 * m == d:
            z_re = U[:, :m]
            z_im = U[:, m:]
            mag_sq = z_re ** 2 + z_im ** 2
            anchor = np.argmax(mag_sq, axis=1)
            rows = np.arange(U.shape[0])
            a_re = z_re[rows, anchor]
            a_im = z_im[rows, anchor]
            a_abs = np.sqrt(np.maximum(a_re ** 2 + a_im ** 2, 1e-24))
            cos_p = a_re / a_abs
            sin_p = a_im / a_abs
            rot_re = z_re * cos_p[:, None] + z_im * sin_p[:, None]
            rot_im = z_im * cos_p[:, None] - z_re * sin_p[:, None]
            return np.concatenate([rot_re, rot_im], axis=1)
        else:
            anchor = np.argmax(np.abs(U), axis=1)
            signs = np.sign(U[np.arange(U.shape[0]), anchor])
            signs = np.where(signs == 0, 1.0, signs)
            return U * signs[:, None]

    def fit(self, X: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {X.shape[1]}")

        Z = self._canonicalize_u1_gauge(X.astype(np.float64))
        N, D = Z.shape
        k_eff = min(self.n_clusters, max(1, N // 4))

        km = KMeans(n_clusters=k_eff, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(Z)
        self.centroids = km.cluster_centers_.astype(np.float64)

        self.projectors = np.zeros((k_eff, D, self.latent_rank), dtype=np.float64)
        self.radii = np.zeros(k_eff, dtype=np.float64)
        self.res_thresholds = np.zeros(k_eff, dtype=np.float64)

        n_per_cluster = max(1.0, float(N) / k_eff)
        margin = 1.0 + 2.2 / np.sqrt(n_per_cluster) + 1.2 * np.sqrt(D) / max(10.0, n_per_cluster)

        for c in range(k_eff):
            mask = labels == c
            Z_c = Z[mask]
            center = self.centroids[c]
            diff = Z_c - center[None, :]
            dists = np.linalg.norm(diff, axis=1)
            self.radii[c] = float(np.quantile(dists, self.quantile) * margin + 1e-6)

            if Z_c.shape[0] > 1:
                _, _, Vt = np.linalg.svd(diff, full_matrices=False)
                r_actual = min(self.latent_rank, Vt.shape[0])
                basis = Vt[:r_actual].T
                self.projectors[c, :, :r_actual] = basis
                proj = (diff @ basis) @ basis.T
                res = np.linalg.norm(diff - proj, axis=1)
                self.res_thresholds[c] = float(np.quantile(res, self.quantile) * margin + 1e-6)
            else:
                self.res_thresholds[c] = self.radii[c]

        self.is_fitted = True
        return self

    def verify(self, X: np.ndarray, mode: str = "enforce") -> dict:
        if not self.is_fitted:
            raise RuntimeError("Shield is not fitted.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {X.shape[1]}")

        Z = self._canonicalize_u1_gauge(X.astype(np.float64))
        N = Z.shape[0]

        dists_all = np.linalg.norm(Z[:, None, :] - self.centroids[None, :, :], axis=2)
        labels = np.argmin(dists_all, axis=1)

        radial_ratios = np.zeros(N, dtype=np.float64)
        residual_ratios = np.zeros(N, dtype=np.float64)

        for i in range(N):
            c = int(labels[i])
            diff = Z[i] - self.centroids[c]
            dist = float(np.linalg.norm(diff))
            radial_ratios[i] = dist / max(self.radii[c], 1e-12)

            basis = self.projectors[c]
            proj = (diff @ basis) @ basis.T
            res = float(np.linalg.norm(diff - proj))
            residual_ratios[i] = res / max(self.res_thresholds[c], 1e-12)

        scores = np.maximum(radial_ratios, residual_ratios)
        raw_accept = scores <= 1.0
        accepted = np.ones(N, dtype=bool) if mode == "shadow" else raw_accept

        return {
            "accepted": accepted,
            "would_quarantine": ~raw_accept,
            "anomaly_scores": scores,
            "cluster_ids": labels,
            "telemetry": {
                "radial_ratios": radial_ratios,
                "residual_ratios": residual_ratios,
                "max_score": float(np.max(scores)),
            },
        }

    def state_dict(self) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Cannot export state_dict of unfitted shield.")
        return {
            "dim": torch.tensor(self.dim, dtype=torch.int64),
            "latent_rank": torch.tensor(self.latent_rank, dtype=torch.int64),
            "n_clusters": torch.tensor(self.n_clusters, dtype=torch.int64),
            "quantile": torch.tensor(self.quantile, dtype=torch.float64),
            "centroids": torch.from_numpy(self.centroids),
            "projectors": torch.from_numpy(self.projectors),
            "radii": torch.from_numpy(self.radii),
            "res_thresholds": torch.from_numpy(self.res_thresholds),
        }

    def load_state_dict(self, state: dict):
        self.dim = int(state["dim"].item())
        self.latent_rank = int(state["latent_rank"].item())
        self.n_clusters = int(state["n_clusters"].item())
        self.quantile = float(state["quantile"].item())
        self.centroids = state["centroids"].detach().cpu().numpy().astype(np.float64)
        self.projectors = state["projectors"].detach().cpu().numpy().astype(np.float64)
        self.radii = state["radii"].detach().cpu().numpy().astype(np.float64)
        self.res_thresholds = state["res_thresholds"].detach().cpu().numpy().astype(np.float64)
        self.is_fitted = True
        return self