import torch
from casimir_harmonics import SpinorHarmonicCasimirEngine

class MetacognitiveTwistorShell:
    """
    Phase 4 Metacognitive Twistor Shell:
      1. Computes strictly U(1)-gauge invariant Fubini-Study projective incidence residual:
             R_inc(t) = 2 |r_t| * sqrt(1 - |<pi_t | pi_{t-1}>|^2)
         which is invariant under arbitrary local phase transformations pi_t -> e^{i alpha_t} pi_t.
      2. Implements an active Closed-Loop Variational Trajectory Stabilizer that damps
         anomalous curvature/incidence excursions via screened Poisson / Tikhonov smoothing.
    """
    @staticmethod
    def vec_to_unit_spinor(v: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=1e-12)
        u = v / r
        x, y, z = u[..., 0], u[..., 1], u[..., 2]
        theta = torch.acos(torch.clamp(z, -1.0 + 1e-12, 1.0 - 1e-12))
        phi = torch.atan2(y, x)
        c_dtype = torch.complex128 if v.dtype == torch.float64 else torch.complex64
        s0 = torch.cos(0.5 * theta).to(c_dtype)
        s1 = (torch.sin(0.5 * theta) * torch.exp(1j * phi)).to(c_dtype)
        return torch.stack([s0, s1], dim=-1)

    @staticmethod
    def evaluate_trajectory_diagnostics(H: torch.Tensor, phase_angles: torch.Tensor = None) -> dict:
        T, D = H.shape
        K = D // 3
        fibers = H[:, :3 * K].reshape(T, K, 3)
        radii = torch.linalg.norm(fibers, dim=-1)  # (T, K)
        dirs = fibers / radii.unsqueeze(-1).clamp(min=1e-12)

        pi = MetacognitiveTwistorShell.vec_to_unit_spinor(fibers)  # (T, K, 2)
        if phase_angles is not None:
            # Apply local U(1) gauge transformation pi -> e^{i alpha} pi to verify gauge invariance
            phase_factor = torch.exp(1j * phase_angles.to(pi.real.dtype)).unsqueeze(-1)
            pi = pi * phase_factor

        # U(1)-gauge invariant Fubini-Study sine distance: sqrt(1 - |<pi_t | pi_{t-1}>|^2)
        inner = torch.sum(torch.conj(pi[1:]) * pi[:-1], dim=-1)  # (T-1, K)
        abs_inner_sq = torch.clamp(torch.abs(inner) ** 2, max=1.0)
        fs_chord = torch.sqrt(torch.clamp(1.0 - abs_inner_sq, min=0.0))

        inc_res_step = 2.0 * radii[1:] * fs_chord
        inc_res_total = float(torch.sum(inc_res_step).item())

        casimir_t = SpinorHarmonicCasimirEngine.compute_casimir_rayleigh_quotient(dirs, radii)
        casimir_mean = float(torch.mean(casimir_t).item())

        return {
            "incidence_residual": inc_res_total,
            "incidence_step_profile": inc_res_step,
            "casimir_mean": casimir_mean,
            "casimir_profile": casimir_t,
        }

    @staticmethod
    def closed_loop_stabilize(H: torch.Tensor, lam: float = 2.0, iterations: int = 3) -> tuple[torch.Tensor, dict]:
        """
        Active closed-loop trajectory feedback operator:
        Identifies high-incidence-residual steps and applies adaptive variational
        Laplacian damping (I + w_t * lam * L) H_stab = H while preserving clean regions.
        """
        H_curr = H.clone()
        diag_before = MetacognitiveTwistorShell.evaluate_trajectory_diagnostics(H_curr)

        for _ in range(iterations):
            d2 = H_curr[2:] - 2.0 * H_curr[1:-1] + H_curr[:-2]
            curv = torch.linalg.norm(d2, dim=-1)
            med_c = torch.median(curv)
            mad_c = torch.median(torch.abs(curv - med_c)).clamp(min=1e-6)
            z_c = (curv - med_c) / (1.4826 * mad_c)
            alpha = torch.sigmoid((z_c - 3.0) * 2.0).unsqueeze(-1)  # adaptive feedback gain in [0, 1]

            correction = 0.5 * (H_curr[2:] + H_curr[:-2]) - H_curr[1:-1]
            H_next = H_curr.clone()
            H_next[1:-1] = H_curr[1:-1] + (lam / (1.0 + 2.0 * lam)) * alpha * correction
            H_curr = H_next

        diag_after = MetacognitiveTwistorShell.evaluate_trajectory_diagnostics(H_curr)
        return H_curr, {
            "residual_before": diag_before["incidence_residual"],
            "residual_after": diag_after["incidence_residual"],
            "suppression_ratio": 1.0 - (diag_after["incidence_residual"] / max(diag_before["incidence_residual"], 1e-12)),
        }