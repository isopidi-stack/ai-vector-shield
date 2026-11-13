import torch
import numpy as np
from .core import AIVectorShield

class VectorShield:
    """
    User-friendly, auto-tuning Vector Firewall for RAG pipelines.
    Wraps complex projective manifold verification into a 3-line API.
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

        print(f"[*] VectorShield Auto-Tuning -> Dim: {D}, Latent Rank: {self.latent_rank}, Clusters: {self.n_clusters}")

        self._shield = AIVectorShield(dim=D, latent_rank=self.latent_rank, n_clusters=self.n_clusters, quantile=self.quantile)
        self._shield.fit(embeddings)
        self._is_fitted = True
        return self

    def verify(self, query_embeddings):
        if not self._is_fitted:
            raise RuntimeError("Shield is not fitted. Call fit() or load() first.")
        
        if isinstance(query_embeddings, np.ndarray):
            query_embeddings = torch.tensor(query_embeddings, dtype=torch.float32)

        predictions = self._shield.verify(query_embeddings)
        return predictions.cpu().numpy().astype(bool) if isinstance(query_embeddings, torch.Tensor) else predictions

    def save(self, filepath: str):
        """Serializes the fitted manifold charts and configuration to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted shield.")
        state = {
            "dim": self._shield.dim,
            "latent_rank": self.latent_rank,
            "n_clusters": self.n_clusters,
            "quantile": self.quantile,
            "charts": self._shield.charts,
            "global_threshold": self._shield.global_threshold
        }
        torch.save(state, filepath)
        print(f"[*] VectorShield model successfully saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Loads a serialized VectorShield model from disk."""
        state = torch.load(filepath, map_location="cpu")
        instance = cls(
            latent_rank=state["latent_rank"],
            n_clusters=state["n_clusters"],
            quantile=state["quantile"]
        )
        shield_core = AIVectorShield(
            dim=state["dim"],
            latent_rank=state["latent_rank"],
            n_clusters=state["n_clusters"],
            quantile=state["quantile"]
        )
        shield_core.charts = state["charts"]
        shield_core.global_threshold = state["global_threshold"]
        instance._shield = shield_core
        instance._is_fitted = True
        print(f"[*] VectorShield model successfully loaded from {filepath}")
        return instance