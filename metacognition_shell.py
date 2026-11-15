import math
import torch
from casimir_harmonics import PenroseTwistorHarmonicEngine

class CP3MetacognitionShell:
    """
    Phase 4 Closed-Loop Activation Trajectory Smoothing Shell:
      - Evaluates the genuine field-level L^2(CP^1) Conformal Casimir Rayleigh quotient
        via PenroseTwistorHarmonicEngine.compute_conformal_energy_functional over a sliding
        window of celestial spinors (replacing the local finite-difference proxy).
      - Evaluates the non-vanishing null-cone incidence residual ||omega_t - i X_t pi_t||_2
        (replacing the identically zero Re(i pi^dagger X pi) expression).
    """
    def __init__(self, d_model: int = 63, window_size: int = 9, device=None):
        if device is not None:
            self.device = device
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = torch.device("xpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.d_model = d_model
        self.num_fibers = d_model // 3
        self.window_size = window_size
        self.harmonics = PenroseTwistorHarmonicEngine(device=self.device)

        gen = torch.Generator(device="cpu").manual_seed(42)
        Q, _ = torch.linalg.qr(torch.randn((d_model, d_model), generator=gen, dtype=torch.float64))
        self.Q = Q.to(device=self.device)

    def evaluate_trajectory_metrics(self, H: torch.Tensor) -> dict:
        H_work = H.to(device=self.device, dtype=torch.float64)
        T = H_work.shape[0]
        K = self.num_fibers
        R_fibers = (H_work @ self.Q[:, :3 * K]).view(T, K, 3)

        # 1. Field-level L^2(CP^1) Casimir Rayleigh quotient across the K-fiber cloud at each step t
        casimir_strain, _, _ = self.harmonics.compute_conformal_energy_functional(R_fibers, ridge_lambda=1e-4)  # (T,)

        # 2. Non-vanishing Twistor Incidence Residual ||omega_t - i X_t pi_{t-1}||_2 across fibers
        pi_fibers = self.harmonics.points_to_celestial_spinors(R_fibers)  # (T, K, 2)
        rx, ry, rz = R_fibers[..., 0], R_fibers[..., 1], R_fibers[..., 2]
        rt = torch.norm(R_fibers, dim=-1)
        X00 = torch.complex(rt + rz, torch.zeros_like(rt))
        X01 = torch.complex(rx, -ry)
        X10 = torch.complex(rx,  ry)
        X11 = torch.complex(rt - rz, torch.zeros_like(rt))
        X_herm = torch.stack([torch.stack([X00, X01], dim=-1), torch.stack([X10, X11], dim=-1)], dim=-2)

        # Compare exact omega_t = i X_t pi_t against parallel-transported omega_trans = i X_t pi_{t-1}
        omega_exact = 1j * torch.einsum("tkab,tkb->tka", X_herm[1:], pi_fibers[1:])
        omega_trans = 1j * torch.einsum("tkab,tkb->tka", X_herm[1:], pi_fibers[:-1])
        incidence_residual = torch.sqrt(torch.sum(torch.abs(omega_exact - omega_trans) ** 2, dim=(-2, -1)))
        incidence_residual = torch.cat([torch.zeros(1, dtype=torch.float64, device=self.device), incidence_residual], dim=0)

        return {
            "casimir_strain": casimir_strain,
            "incidence_residual": incidence_residual,
        }