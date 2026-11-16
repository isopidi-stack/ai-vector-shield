
def _u1_canon_any(x):
    import numpy as np, torch
    is_t = isinstance(x, torch.Tensor)
    t = x.detach().clone().to(torch.float64) if is_t else torch.as_tensor(np.asarray(x), dtype=torch.float64)
    is_1d = (t.ndim == 1)
    if is_1d:
        t = t.unsqueeze(0)
    norms = torch.linalg.norm(t, dim=-1, keepdim=True).clamp(min=1e-12)
    u = t / norms
    d = u.shape[-1]
    if d % 2 == 0:
        m = d // 2
        zr, zi = u[..., :m], u[..., m:]
        idx = torch.argmax(zr * zr + zi * zi, dim=-1, keepdim=True)
        ar, ai = torch.gather(zr, -1, idx), torch.gather(zi, -1, idx)
        aa = torch.sqrt((ar * ar + ai * ai).clamp(min=1e-24))
        cp, sp = ar / aa, ai / aa
        res = torch.cat([zr * cp + zi * sp, zi * cp - zr * sp], dim=-1) * norms
    else:
        idx = torch.argmax(torch.abs(u), dim=-1, keepdim=True)
        sg = torch.sign(torch.gather(u, -1, idx))
        sg = torch.where(sg == 0, torch.ones_like(sg), sg)
        res = (u * sg) * norms
    if is_1d:
        res = res.squeeze(0)
    return res.to(x.dtype) if is_t else res.numpy()

﻿import time
import torch
import numpy as np
from .core import AIVectorShield

class VectorShield:
    """
    Enterprise Vector Firewall for RAG pipelines powered by piecewise-local
    complex projective manifold verification.
    """
    def __init__(
        self,
        latent_rank: int = 8,
        n_clusters: int = 3,
        quantile: float = 0.99,
        mode: str = "enforce",
    ):
        if mode not in ("enforce", "shadow"):
            raise ValueError("mode must be 'enforce' or 'shadow'")
        self.latent_rank = latent_rank
        self.n_clusters = n_clusters
        self.quantile = quantile
        self.mode = mode
        self._shield = None
        self._is_fitted = False

    def fit(self, embeddings):
        embeddings = _u1_canon_any(embeddings)
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
        query_embeddings = _u1_canon_any(query_embeddings)
        if not self._is_fitted:
            raise RuntimeError("Shield is not fitted. Call fit() or load() first.")
        predictions = self._shield.verify(query_embeddings)
        if self.mode == "shadow":
            return np.ones(predictions.shape[0], dtype=bool)
        return predictions.cpu().numpy().astype(bool)

    def inspect(self, query_embeddings):
        query_embeddings = _u1_canon_any(query_embeddings)
        """Returns structured per-sample security diagnostics with latency telemetry."""
        if not self._is_fitted:
            raise RuntimeError("Shield is not fitted. Call fit() or load() first.")
        t0 = time.perf_counter()
        allowed, residuals, ratios, charts = self._shield.score_samples(query_embeddings)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_sample_ms = elapsed_ms / max(1, allowed.shape[0])

        results = []
        for a, r, rat, c in zip(
            allowed.cpu().tolist(),
            residuals.cpu().tolist(),
            ratios.cpu().tolist(),
            charts.cpu().tolist(),
        ):
            is_inlier = bool(a)
            action = "allow" if is_inlier else ("shadow_flagged" if self.mode == "shadow" else "blocked")
            results.append({
                "allowed": True if self.mode == "shadow" else is_inlier,
                "in_manifold": is_inlier,
                "action": action,
                "residual": float(r),
                "risk_ratio": float(rat),
                "chart_id": int(c),
                "latency_ms": round(per_sample_ms, 4),
            })
        return results

    def save(self, filepath: str):
        """Serializes fitted manifold charts using plain tensors (safe for weights_only=True)."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted shield.")
        state = {
            "dim": int(self._shield.dim),
            "latent_rank": int(self.latent_rank),
            "n_clusters": int(self.n_clusters),
            "quantile": float(self.quantile),
            "mode": str(self.mode),
            "charts": self._shield.charts,
            "global_threshold": float(self._shield.global_threshold),
        }
        torch.save(state, filepath)

    @classmethod
    def load(cls, filepath: str, mode: str = None):
        """Loads a serialized VectorShield model safely with weights_only=True."""
        state = torch.load(filepath, map_location="cpu", weights_only=True)
        instance = cls(
            latent_rank=state["latent_rank"],
            n_clusters=state["n_clusters"],
            quantile=state["quantile"],
            mode=mode or state.get("mode", "enforce"),
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
