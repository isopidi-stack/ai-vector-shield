import torch
import numpy as np
from .core import AIVectorShield

class VectorShield:
    """
    Enterprise Vector Firewall for RAG pipelines powered by piecewise-local
    complex projective manifold verification.
    """
    def __init__(self, latent_rank: int = 8, n_clusters: int = 3, quantile: float = 0.99):
        self.latent_rank = latent_rank
        self.n_clusters = n_clusters
        self.quantile = quantile
        self._shield = None
        self._is_fitted = False

    def fit(self, embeddings):
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)

        N, D = embeddings.shape
        if self.latent_rank is None:
            self.latent_rank = max(4, min(16, D // 24))
        if self.n_clusters is None:
            self.n_clusters = max(2, min(6, N // 100))

        self._shield = AIVectorShield(
            dim=D,
            latent_rank=self.latent_rank,
            n_clusters=self.n_clusters,
            quantile=self.quantile,
        )
        self._shield.fit(embeddings)
        self._is_fitted = True
        return self

    def verify(self, query_embeddings):
        if not self._is_fitted:
            raise RuntimeError("Shield is not fitted. Call fit() or load() first.")
        predictions = self._shield.verify(query_embeddings)
        return predictions.cpu().numpy().astype(bool)

    def inspect(self, query_embeddings):
        """Returns structured per-sample security diagnostics (allowed, residual, risk_ratio, chart_id)."""
        if not self._is_fitted:
            raise RuntimeError("Shield is not fitted. Call fit() or load() first.")
        allowed, residuals, ratios, charts = self._shield.score_samples(query_embeddings)
        return [
            {
                "allowed": bool(a),
                "residual": float(r),
                "risk_ratio": float(rat),
                "chart_id": int(c),
            }
            for a, r, rat, c in zip(
                allowed.cpu().tolist(),
                residuals.cpu().tolist(),
                ratios.cpu().tolist(),
                charts.cpu().tolist(),
            )
        ]

    def save(self, filepath: str):
        """Serializes fitted manifold charts using plain tensors (safe for weights_only=True)."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted shield.")
        state = {
            "dim": int(self._shield.dim),
            "latent_rank": int(self.latent_rank),
            "n_clusters": int(self.n_clusters),
            "quantile": float(self.quantile),
            "charts": self._shield.charts,
            "global_threshold": float(self._shield.global_threshold),
        }
        torch.save(state, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Loads a serialized VectorShield model safely with weights_only=True."""
        state = torch.load(filepath, map_location="cpu", weights_only=True)
        instance = cls(
            latent_rank=state["latent_rank"],
            n_clusters=state["n_clusters"],
            quantile=state["quantile"],
        )
        shield_core = AIVectorShield(
            dim=state["dim"],
            latent_rank=state["latent_rank"],
            n_clusters=state["n_clusters"],
            quantile=state["quantile"],
        )
        shield_core.charts = state["charts"]
        shield_core.global_threshold = state["global_threshold"]
        instance._shield = shield_core
        instance._is_fitted = True
        return instance
