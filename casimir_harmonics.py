import math
import torch

class SpinorHarmonicCasimirEngine:
    """
    Evaluates the l <= 2 spherical harmonic basis on S^2 / CP^1 and computes the
    regularized Rayleigh quotient for the Laplace-Beltrami operator -Delta_{S^2}
    with eigenvalues lambda_l = l(l+1) in {0, 2, 6}.

    Unsold's Addition Theorem Identity:
      sum_{l=0}^2 sum_{m=-l}^l |Y_{l,m}(n)|^2 = (1 + 3 + 5) / (4 * pi) = 9 / (4 * pi) approx 0.7161972439.
    """
    UNSOLD_CONSTANT_L2 = 9.0 / (4.0 * math.pi)

    @staticmethod
    def evaluate_real_spherical_harmonics_l2(dirs: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the 9 real orthonormal spherical harmonics Y_{l,m} (l <= 2) on unit vectors dirs (..., 3).
        """
        u = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp(min=1e-12)
        x, y, z = u[..., 0], u[..., 1], u[..., 2]

        c0 = math.sqrt(1.0 / (4.0 * math.pi))
        Y00 = torch.full_like(x, c0)

        c1 = math.sqrt(3.0 / (4.0 * math.pi))
        Y1_m1 = c1 * y
        Y1_0  = c1 * z
        Y1_p1 = c1 * x

        c2_xy = 0.5 * math.sqrt(15.0 / math.pi)
        c2_z2 = 0.25 * math.sqrt(5.0 / math.pi)
        Y2_m2 = c2_xy * x * y
        Y2_m1 = c2_xy * y * z
        Y2_0  = c2_z2 * (3.0 * z * z - 1.0)
        Y2_p1 = c2_xy * x * z
        Y2_p2 = 0.5 * c2_xy * (x * x - y * y)

        return torch.stack([Y00, Y1_m1, Y1_0, Y1_p1, Y2_m2, Y2_m1, Y2_0, Y2_p1, Y2_p2], dim=-1)

    @staticmethod
    def verify_unsold_theorem(dirs: torch.Tensor) -> float:
        Y = SpinorHarmonicCasimirEngine.evaluate_real_spherical_harmonics_l2(dirs)
        norm_sq = torch.sum(Y ** 2, dim=-1)
        return float(torch.max(torch.abs(norm_sq - SpinorHarmonicCasimirEngine.UNSOLD_CONSTANT_L2)).item())

    @staticmethod
    def compute_casimir_rayleigh_quotient(
        dirs: torch.Tensor,
        field_vals: torch.Tensor,
        ridge_lambda: float = 1e-3,
        return_diagnostics: bool = False,
    ):
        """
        Fits the 9-mode spherical harmonic expansion to field_vals (T, K) across K fibers
        and evaluates the Laplacian Rayleigh quotient:
            E = (2 ||c_1||^2 + 6 ||c_2||^2) / (||c_0||^2 + ||c_1||^2 + ||c_2||^2 + eps).
        """
        Y = SpinorHarmonicCasimirEngine.evaluate_real_spherical_harmonics_l2(dirs)  # (T, K, 9)
        Yt = Y.transpose(-1, -2)  # (T, 9, K)
        G = torch.matmul(Yt, Y)   # (T, 9, 9)
        I9 = torch.eye(9, dtype=G.dtype, device=G.device).unsqueeze(0)

        rhs = torch.matmul(Yt, field_vals.unsqueeze(-1)).squeeze(-1)  # (T, 9)
        coeffs = torch.linalg.solve(G + ridge_lambda * I9, rhs)

        c0_sq = coeffs[:, 0] ** 2
        c1_sq = torch.sum(coeffs[:, 1:4] ** 2, dim=-1)
        c2_sq = torch.sum(coeffs[:, 4:9] ** 2, dim=-1)

        num = 2.0 * c1_sq + 6.0 * c2_sq
        den = c0_sq + c1_sq + c2_sq + 1e-12
        casimir_e = num / den

        if return_diagnostics:
            cond_G = torch.linalg.cond(G + ridge_lambda * I9)
            return casimir_e, {"coeffs": coeffs, "cond_G_max": float(torch.max(cond_G).item())}
        return casimir_e