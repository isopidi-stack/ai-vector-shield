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

    def _to_complex_projective(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.complex(x_norm, torch.zeros_like(x_norm))

    def fit(self, embeddings: torch.Tensor):
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        N = embeddings.shape[0]
        # Prevent overfitting by ensuring latent rank leaves plenty of degrees of freedom
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
            U, S, V = torch.linalg.svd(centered_matrix, full_matrices=False)
                
            rank = min(effective_rank, V.shape[0])
            basis = V[:rank]

            res = self._chart_residual(cluster_z, mean, basis)
            thresh = torch.quantile(res, self.quantile)
            self.charts.append((mean, basis, thresh))
            residuals.append(res)

        all_res = torch.cat(residuals)
        self.global_threshold = torch.quantile(all_res, self.quantile).item()
        return self

    def _chart_residual(self, z: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        centered = z - mean
        mat = torch.cat([centered.real, centered.imag], dim=-1)
        projected = torch.matmul(mat, basis.T)
        reconstructed = torch.matmul(projected, basis)
        residual = torch.norm(mat - reconstructed, dim=-1)
        return residual

    def verify(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        z = self._to_complex_projective(x)
        
        allowed = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        for mean, basis, thresh in self.charts:
            r = self._chart_residual(z, mean, basis)
            allowed |= (r <= thresh)
            
        return allowed