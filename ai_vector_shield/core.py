import math
import torch
import numpy as np
from sklearn.cluster import KMeans


class AIVectorShield:
    r"""
    Core engine for piecewise-local complex projective manifolds (\mathbb{CP}^d)
    providing vector-space security and out-of-distribution detection for RAG.

    Enforces a two-gate geometric invariant per local chart:
      1. Transverse Subspace Incidence (off-manifold tangent residual orthogonal to mu)
      2. Longitudinal Geodesic Drift (chord radius from chart ray mu)
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
        z_real = torch.cat([z.real, z.imag], dim=-1)
        D_eff = z_real.shape[-1]

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42).fit(z_real.cpu().numpy())
        labels = kmeans.labels_

        self.charts = []
        residuals = []

        for k in range(self.n_clusters):
            cluster_mask = (labels == k)
            if cluster_mask.sum() < effective_rank + 2:
                cluster_mask = np.ones_like(labels, dtype=bool)

            cluster_z = z[cluster_mask]
            n_k = int(cluster_z.shape[0])

            # Projective ray centroid normalized on the unit sphere
            raw_mean = cluster_z.mean(dim=0, keepdim=True)
            mu = raw_mean / (torch.linalg.norm(raw_mean, dim=-1, keepdim=True) + 1e-8)

            # True tangent projection orthogonal to mu (removes sphere radial curvature 1 - cos(theta))
            mat_z = torch.cat([cluster_z.real, cluster_z.imag], dim=-1)
            mat_mu = torch.cat([mu.real, mu.imag], dim=-1)
            dot_mu = torch.sum(mat_z * mat_mu, dim=-1, keepdim=True)
            tan_mat = mat_z - dot_mu * mat_mu

            _, _, Vh = torch.linalg.svd(tan_mat, full_matrices=False)
            rank = min(effective_rank, Vh.shape[0], max(1, D_eff - 2))
            basis = Vh[:rank]  # [rank, D_eff] orthonormal tangent basis orthogonal to mu

            res_perp, res_long = self._chart_components(cluster_z, mu, basis)
            raw_q_perp = torch.quantile(res_perp, self.quantile).item()
            raw_q_long = torch.quantile(res_long, self.quantile).item()

            # Finite-sample & subspace degrees-of-freedom correction for unseen queries
            dof_dim = math.sqrt(max(1.0, (D_eff - 1.0) / max(1.0, D_eff - 1.0 - rank)))
            dof_sample = math.sqrt((n_k + 1.0) / max(1.0, n_k - rank - 1.0))
            thresh_perp = max(raw_q_perp * dof_dim * dof_sample, 1e-6)
            thresh_long = max(raw_q_long * math.sqrt((n_k + 1.0) / max(1.0, n_k - 2.0)), 1e-6)

            self.charts.append({
                "mean_real": mu.real.contiguous(),
                "mean_imag": mu.imag.contiguous(),
                "basis": basis.contiguous(),
                "threshold": float(thresh_perp),
                "threshold_long": float(thresh_long),
            })
            residuals.append(res_perp * (dof_dim * dof_sample))

        all_res = torch.cat(residuals)
        self.global_threshold = torch.quantile(all_res, self.quantile).item()
        return self

    def _chart_components(self, z: torch.Tensor, mu: torch.Tensor, basis: torch.Tensor):
        mat_z = torch.cat([z.real, z.imag], dim=-1)
        mat_mu = torch.cat([mu.real, mu.imag], dim=-1)

        # Remove radial sphere curvature along mu so tangent sheet is curvature-free
        dot_mu = torch.sum(mat_z * mat_mu, dim=-1, keepdim=True)
        tan_mat = mat_z - dot_mu * mat_mu

        # Gate 1: Transverse Subspace Incidence Residual (off the tangent sheet)
        projected = torch.matmul(tan_mat, basis.mT)
        reconstructed = torch.matmul(projected, basis)
        res_perp = torch.linalg.norm(tan_mat - reconstructed, dim=-1)

        # Gate 2: Longitudinal Geodesic Chord Drift from centroid
        res_long = torch.linalg.norm(mat_z - mat_mu, dim=-1)
        return res_perp, res_long

    def score_samples(self, x: torch.Tensor):
        x = self._validate_input(x)
        z = self._to_complex_projective(x)

        all_residuals = []
        all_ratios = []
        for chart in self.charts:
            mu = torch.complex(chart["mean_real"].to(x.device), chart["mean_imag"].to(x.device))
            basis = chart["basis"].to(x.device)
            thresh_perp = chart["threshold"]
            thresh_long = chart.get("threshold_long", thresh_perp * 4.0)

            res_perp, res_long = self._chart_components(z, mu, basis)
            ratio_perp = res_perp / (thresh_perp + 1e-8)
            ratio_long = res_long / (thresh_long + 1e-8)

            combined_ratio = torch.maximum(ratio_perp, ratio_long)
            all_residuals.append(res_perp)
            all_ratios.append(combined_ratio)

        ratios_stack = torch.stack(all_ratios, dim=-1)        # [N, K]
        residuals_stack = torch.stack(all_residuals, dim=-1)  # [N, K]
        min_ratios, best_charts = torch.min(ratios_stack, dim=-1)
        best_residuals = residuals_stack.gather(1, best_charts.unsqueeze(-1)).squeeze(-1)
        allowed = min_ratios <= 1.0
        return allowed, best_residuals, min_ratios, best_charts

    def verify(self, x: torch.Tensor) -> torch.Tensor:
        allowed, _, _, _ = self.score_samples(x)
        return allowed