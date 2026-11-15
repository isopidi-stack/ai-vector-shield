import math
import torch

class PenroseTwistorHarmonicEngine:
    """
    Genuine L^2(CP^1) Spherical Harmonic & Conformal Casimir Rayleigh-Quotient Engine.
    Supports both single 3D point clouds (N, 3) and batched K-fiber bundles (T, K, 3).
    """
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")

    @staticmethod
    def points_to_celestial_spinors(pts_3d: torch.Tensor, camera_center: torch.Tensor = None) -> torch.Tensor:
        rel = pts_3d - camera_center.unsqueeze(0) if camera_center is not None else pts_3d
        r = torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-12)
        n = rel / r
        nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]

        c_dtype = torch.complex128 if pts_3d.dtype == torch.float64 else torch.complex64
        denom = torch.clamp(1.0 + nz, min=1e-12)
        p0 = torch.sqrt(0.5 * (1.0 + nz).clamp(min=0.0)).to(c_dtype)
        p1 = torch.complex(nx, ny).to(c_dtype) / torch.sqrt(2.0 * denom).to(c_dtype)

        south_mask = (1.0 + nz) < 1e-10
        p0 = torch.where(south_mask, torch.zeros_like(p0), p0)
        p1 = torch.where(south_mask, torch.ones_like(p1), p1)
        return torch.stack([p0, p1], dim=-1)

    @staticmethod
    def spinor_to_unit_ray(pi: torch.Tensor) -> torch.Tensor:
        p0, p1 = pi[..., 0], pi[..., 1]
        norm_sq = (torch.abs(p0) ** 2 + torch.abs(p1) ** 2).clamp(min=1e-12)
        nx = 2.0 * torch.real(torch.conj(p0) * p1) / norm_sq
        ny = 2.0 * torch.imag(torch.conj(p0) * p1) / norm_sq
        nz = (torch.abs(p0) ** 2 - torch.abs(p1) ** 2) / norm_sq
        return torch.stack([nx, ny, nz], dim=-1)

    @staticmethod
    def evaluate_complex_eigenmodes(pi: torch.Tensor) -> dict:
        n = PenroseTwistorHarmonicEngine.spinor_to_unit_ray(pi)
        x, y, z = n[..., 0], n[..., 1], n[..., 2]
        c_dtype = pi.dtype

        Y00  = torch.full_like(x, 0.5 * math.sqrt(1.0 / math.pi)).to(c_dtype)
        c1   = 0.5 * math.sqrt(1.5 / math.pi)
        Y1m1 =  c1 * torch.complex(x, -y).to(c_dtype)
        Y10  =  0.5 * math.sqrt(3.0 / math.pi) * z.to(c_dtype)
        Y1p1 = -c1 * torch.complex(x,  y).to(c_dtype)

        c2_2 = 0.25 * math.sqrt(7.5 / math.pi)
        c2_1 = 0.5  * math.sqrt(7.5 / math.pi)
        c2_0 = 0.25 * math.sqrt(5.0 / math.pi)
        xy_p = torch.complex(x,  y).to(c_dtype)
        xy_m = torch.complex(x, -y).to(c_dtype)

        Y2m2 =  c2_2 * (xy_m ** 2)
        Y2m1 =  c2_1 * z.to(c_dtype) * xy_m
        Y20  =  c2_0 * (3.0 * (z ** 2) - 1.0).to(c_dtype)
        Y2p1 = -c2_1 * z.to(c_dtype) * xy_p
        Y2p2 =  c2_2 * (xy_p ** 2)

        return {
            (0,  0): Y00,
            (1, -1): Y1m1, (1, 0): Y10, (1, 1): Y1p1,
            (2, -2): Y2m2, (2, -1): Y2m1, (2, 0): Y20, (2, 1): Y2p1, (2, 2): Y2p2,
        }

    @staticmethod
    def evaluate_real_basis(pi: torch.Tensor) -> torch.Tensor:
        n = PenroseTwistorHarmonicEngine.spinor_to_unit_ray(pi)
        x, y, z = n[..., 0], n[..., 1], n[..., 2]

        b00  = torch.full_like(x, 0.5 * math.sqrt(1.0 / math.pi))
        c1   = 0.5 * math.sqrt(3.0 / math.pi)
        b1m1 = c1 * y
        b10  = c1 * z
        b1p1 = c1 * x

        c2   = 0.5 * math.sqrt(15.0 / math.pi)
        b2m2 = c2 * x * y
        b2m1 = c2 * y * z
        b20  = 0.25 * math.sqrt(5.0 / math.pi) * (3.0 * z * z - 1.0)
        b2p1 = c2 * x * z
        b2p2 = 0.5 * c2 * (x * x - y * y)

        return torch.stack([b00, b1m1, b10, b1p1, b2m2, b2m1, b20, b2p1, b2p2], dim=-1)

    def compute_conformal_energy_functional(
        self,
        pts_3d: torch.Tensor,
        camera_center: torch.Tensor = None,
        ridge_lambda: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects the empirical radial field f(pi_k) = ||r_k|| / mean(||r||) onto the 9-mode
        real orthonormal spherical harmonic basis B (shape (..., K, 9)) via ridge-regularized
        normal equations and evaluates the Laplace-Beltrami / Casimir Rayleigh quotient:
            E_Casimir = (2 * ||c_{l=1}||^2 + 6 * ||c_{l=2}||^2) / (||c||^2 + 1e-12)
        Supports both (N, 3) single clouds and (T, K, 3) batched fiber bundles.
        """
        rel = pts_3d - camera_center.unsqueeze(0) if camera_center is not None else pts_3d
        radii = torch.norm(rel, dim=-1).clamp(min=1e-12)  # (..., K)
        f_field = radii / (torch.mean(radii, dim=-1, keepdim=True) + 1e-12)  # Scale-invariant profile

        pi = self.points_to_celestial_spinors(rel)  # (..., K, 2)
        B = self.evaluate_real_basis(pi)            # (..., K, 9)

        K_pts = B.shape[-2]
        BtB = torch.einsum("...ki,...kj->...ij", B, B) / float(K_pts)  # (..., 9, 9)
        Btf = torch.einsum("...ki,...k->...i", B, f_field) / float(K_pts)  # (..., 9)

        I9 = torch.eye(9, dtype=B.dtype, device=B.device)
        coeffs = torch.linalg.solve(BtB + ridge_lambda * I9, Btf.unsqueeze(-1)).squeeze(-1)  # (..., 9)

        eigenvalues = torch.tensor([0.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0], dtype=B.dtype, device=B.device)
        numerator = torch.sum(eigenvalues * (coeffs ** 2), dim=-1)
        denominator = torch.sum(coeffs ** 2, dim=-1) + 1e-12
        rayleigh_energy = numerator / denominator
        return rayleigh_energy, coeffs, pi