import torch
import numpy as np
from sklearn.cluster import KMeans

class AIVectorShield:
    r"""
    Core engine for piecewise-local complex projective manifolds (\mathbb{CP}^d)
    providing vector-space security and out-of-distribution detection for RAG.
    """
    def __init__(self, dim: int, latent_rank: int = 8, n_clusters: int = 3, quantile: float = 0.99):
        self.dim = dim
        self.latent_rank = latent_rank
        self.n_clusters = n_clusters
        self.quantile = quantile
        self.charts = []
        self.global_threshold = 0.0

    def _validate_input(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dim}, got {x.shape[-1]}.")
        return x

    def _to_complex_projective(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.complex(x_norm, torch.zeros_like(x_norm))

    def fit(self, embeddings: torch.Tensor):
        embeddings = self._validate_input(embeddings)
        N = embeddings.shape[0]
        effective_rank = min(self.latent_rank, max(2, N // (3 * self.n_clusters)))

        z = self._to_complex_projective(embeddings)
        z_np = torch.cat([z.real, z.imag], dim=-1).cpu().numpy()

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42).fit(z_np)
        labels = kmeans.labels_

        self.charts = []
        residuals = []

        for k in range(self.n_clusters):
            cluster_mask = (labels == k)
            if cluster_mask.sum() < effective_rank:
                cluster_mask = np.ones_like(labels, dtype=bool)

            cluster_z = z[cluster_mask]
            mean = cluster_z.mean(dim=0, keepdim=True)
            centered = cluster_z - mean

            centered_matrix = torch.cat([centered.real, centered.imag], dim=-1)
            _, _, V = torch.linalg.svd(centered_matrix, full_matrices=False)

            rank = min(effective_rank, V.shape[0])
            basis = V[:rank]

            res = self._chart_residual(cluster_z, mean, basis)
            thresh = torch.quantile(res, self.quantile).item()
            self.charts.append({
                "mean_real": mean.real.contiguous(),
                "mean_imag": mean.imag.contiguous(),
                "basis": basis.contiguous(),
                "threshold": float(thresh),
            })
            residuals.append(res)

        all_res = torch.cat(residuals)
        self.global_threshold = torch.quantile(all_res, self.quantile).item()
        return self

    def _chart_residual(self, z: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        centered = z - mean
        mat = torch.cat([centered.real, centered.imag], dim=-1)
        projected = torch.matmul(mat, basis.T)
        reconstructed = torch.matmul(projected, basis)
        return torch.norm(mat - reconstructed, dim=-1)

    def score_samples(self, x: torch.Tensor):
        x = self._validate_input(x)
        z = self._to_complex_projective(x)

        all_residuals = []
        all_ratios = []
        for chart in self.charts:
            mean = torch.complex(chart["mean_real"].to(x.device), chart["mean_imag"].to(x.device))
            basis = chart["basis"].to(x.device)
            thresh = chart["threshold"]
            r = self._chart_residual(z, mean, basis)
            all_residuals.append(r)
            all_ratios.append(r / (thresh + 1e-8))

        ratios_stack = torch.stack(all_ratios, dim=-1)       # [N, K]
        residuals_stack = torch.stack(all_residuals, dim=-1) # [N, K]
        min_ratios, best_charts = torch.min(ratios_stack, dim=-1)
        best_residuals = residuals_stack.gather(1, best_charts.unsqueeze(-1)).squeeze(-1)
        allowed = min_ratios <= 1.0
        return allowed, best_residuals, min_ratios, best_charts

    def verify(self, x: torch.Tensor) -> torch.Tensor:
        allowed, _, _, _ = self.score_samples(x)
        return allowed
