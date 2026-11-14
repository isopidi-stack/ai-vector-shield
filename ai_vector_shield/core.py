import math
import torch
import numpy as np
from sklearn.cluster import KMeans


class AIVectorShield:
    r"""
    Core engine for piecewise-local complex projective manifolds (\mathbb{CP}^d)
    providing vector-space security and out-of-distribution detection for RAG.

    Implements native complex64 spinor pairing, Hermitian Fubini-Study tangent
    space projection (z_perp = z - <mu | z> mu), and complex unitary SVD
    subspace verification (V^H V).
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
        if torch.is_complex(x):
            return (x / (torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-8)).to(torch.complex64)
        x_f = x.to(torch.float32)
        x_norm = x_f / (torch.linalg.norm(x_f, dim=-1, keepdim=True) + 1e-8)
        if x_norm.shape[-1] % 2 == 0:
            return torch.complex(x_norm[..., 0::2], x_norm[..., 1::2])
        return torch.complex(x_norm, torch.zeros_like(x_norm))

    def fit(self, embeddings: torch.Tensor):
        embeddings = self._validate_input(embeddings)
        N = embeddings.shape[0]
        effective_rank = min(self.latent_rank, max(2, N // (3 * self.n_clusters)))

        z = self._to_complex_projective(embeddings)
        complex_dim = z.shape[-1]
        z_np = torch.cat([z.real, z.imag], dim=-1).cpu().numpy()

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42).fit(z_np)
        labels = kmeans.labels_

        self.charts = []
        residuals = []

        for k in range(self.n_clusters):
            cluster_mask = (labels == k)
            if cluster_mask.sum() < effective_rank + 2:
                cluster_mask = np.ones_like(labels, dtype=bool)

            cluster_z = z[cluster_mask]
            n_k = int(cluster_z.shape[0])

            # Projective ray centroid normalized on the complex unit sphere
            raw_mean = cluster_z.mean(dim=0, keepdim=True)
            mu = raw_mean / (torch.linalg.norm(raw_mean, dim=-1, keepdim=True) + 1e-8)

            # Hermitian Fubini-Study orthogonal tangent projection: z_perp = z - <mu | z> mu
            inner = torch.sum(mu.conj() * cluster_z, dim=-1, keepdim=True)
            z_perp = cluster_z - inner * mu

            # Native complex64 unitary SVD on the local tangent chart
            _, _, Vh = torch.linalg.svd(z_perp, full_matrices=False)
            rank = min(effective_rank, Vh.shape[0], max(1, complex_dim - 1))
            basis = Vh[:rank]  # [rank, complex_dim] complex64 unitary rows

            res = self._chart_residual(cluster_z, mu, basis)
            raw_q = torch.quantile(res, self.quantile).item()

            # Real degrees of freedom in complex dimension m is 2m - 1 (with 2*rank absorbed)
            eff_real_dim = float(2 * complex_dim - 1)
            eff_subspace_dof = float(2 * rank)
            dof_dim = math.sqrt(max(1.0, eff_real_dim / max(1.0, eff_real_dim - eff_subspace_dof)))
            dof_sample = math.sqrt((n_k + 1.0) / max(1.0, n_k - rank - 1.0))
            thresh = raw_q * dof_dim * dof_sample

            self.charts.append({
                "mean_real": mu.real.contiguous(),
                "mean_imag": mu.imag.contiguous(),
                "basis_real": basis.real.contiguous(),
                "basis_imag": basis.imag.contiguous(),
                "threshold": float(thresh),
            })
            residuals.append(res * (dof_dim * dof_sample))

        all_res = torch.cat(residuals)
        self.global_threshold = torch.quantile(all_res, self.quantile).item()
        return self

    def _chart_residual(self, z: torch.Tensor, mu: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        # 1. Hermitian inner product <mu | z>
        inner = torch.sum(mu.conj() * z, dim=-1, keepdim=True)

        # 2. Orthogonal complex tangent vector z_perp = z - <mu | z> mu
        z_perp = z - inner * mu

        # 3. Project onto unitary complex basis via Hermitian conjugate transpose (.mH)
        projected = torch.matmul(z_perp, basis.mH)
        reconstructed = torch.matmul(projected, basis)
        transverse_sq = torch.sum(torch.abs(z_perp - reconstructed) ** 2, dim=-1)

        # 4. Exact Pythagorean longitudinal ray drift |1 - <mu | z>|^2
        longitudinal_sq = (torch.abs(1.0 - inner.squeeze(-1)) ** 2)
        return torch.sqrt(transverse_sq + longitudinal_sq + 1e-12)

    def score_samples(self, x: torch.Tensor):
        x = self._validate_input(x)
        z = self._to_complex_projective(x)

        all_residuals = []
        all_ratios = []
        for chart in self.charts:
            mu = torch.complex(chart["mean_real"].to(x.device), chart["mean_imag"].to(x.device))
            if "basis_real" in chart:
                basis = torch.complex(chart["basis_real"].to(x.device), chart["basis_imag"].to(x.device))
            else:
                basis = chart["basis"].to(x.device)
            thresh = chart["threshold"]
            r = self._chart_residual(z, mu, basis)
            all_residuals.append(r)
            all_ratios.append(r / (thresh + 1e-8))

        ratios_stack = torch.stack(all_ratios, dim=-1)        # [N, K]
        residuals_stack = torch.stack(all_residuals, dim=-1)  # [N, K]
        min_ratios, best_charts = torch.min(ratios_stack, dim=-1)
        best_residuals = residuals_stack.gather(1, best_charts.unsqueeze(-1)).squeeze(-1)
        allowed = min_ratios <= 1.0
        return allowed, best_residuals, min_ratios, best_charts

    def verify(self, x: torch.Tensor) -> torch.Tensor:
        allowed, _, _, _ = self.score_samples(x)
        return allowed