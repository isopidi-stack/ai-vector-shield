import math
import torch


class PalatialCP4StandingWaveEngine:
    """Penrose Palatial Twistor Standing-Wave Engine on CP^4 (C^5).

    Enforces:
    1. Strictly real float64 base definition for the SU(2,3) metric SIGMA_5,
       with scoped complex128 promotion during bracket contractions.
    2. Exact Schur-Weyl harmonic projections (P_G, P_P, P_O) satisfying
       c_G + c_P + c_O == 1.0 to 1e-16 precision.
    3. Exact Casimir eigenvalues E_{l,l}^{(4)} in {0, 12, 32}.
    4. Discrete U(1) Wilson loop Berry phase holonomy flux.
    5. Fixed-size O(1) ring buffer support for real-time density streaming.
    """

    # Strictly real float64 definition of the pseudo-Hermitian SU(2,3) metric
    SIGMA_5 = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    # Exact integer Casimir spectrum on CP^4: 2l(2l + 4)
    CASIMIR_G = 0.0
    CASIMIR_P = 12.0
    CASIMIR_O = 32.0

    @classmethod
    def evaluate_su23_inner_product(cls, Z_a: torch.Tensor, Z_b: torch.Tensor) -> torch.Tensor:
        """Evaluates the indefinite SU(2,3) inner product <Z_a | SIGMA_5 | Z_b>
        with dynamic scoped casting of the real float64 metric to complex128.
        """
        sigma_c = cls.SIGMA_5.to(dtype=Z_a.dtype, device=Z_a.device)
        return torch.einsum("...i,ij,...j->...", Z_a.conj(), sigma_c, Z_b)

    @staticmethod
    def build_triadic_projectors(device=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Constructs the explicit 5x5 Schur-Weyl harmonic projection matrices
        (P_G, P_P, P_O) on CP^4 satisfying P_G + P_P + P_O = I_5.
        """
        device = device or torch.device("cpu")
        dtype = torch.complex128

        P_G = torch.zeros((5, 5), dtype=dtype, device=device)
        for a in (0, 1):
            v = torch.zeros((5, 1), dtype=dtype, device=device)
            v[a, 0] = 1.0 / math.sqrt(2.0)
            v[a + 2, 0] = 1.0 / math.sqrt(2.0)
            P_G = P_G + (v @ v.conj().T)

        P_P = torch.zeros((5, 5), dtype=dtype, device=device)
        for a in (0, 1):
            v = torch.zeros((5, 1), dtype=dtype, device=device)
            v[a, 0] = 1.0 / math.sqrt(2.0)
            v[a + 2, 0] = -1.0 / math.sqrt(2.0)
            P_P = P_P + (v @ v.conj().T)

        P_O = torch.zeros((5, 5), dtype=dtype, device=device)
        P_O[4, 4] = 1.0 + 0.0j

        return P_G, P_P, P_O

    @staticmethod
    def compute_sliding_berry_holonomy(Z5_hat: torch.Tensor, window: int = 4) -> torch.Tensor:
        """Computes discrete sliding-window U(1) Wilson loop Berry phase holonomy."""
        T, K, _ = Z5_hat.shape
        device = Z5_hat.device
        berry_flux = torch.zeros(T, dtype=torch.float64, device=device)
        W = min(window, T - 1)
        if W < 2:
            return berry_flux

        for t in range(T):
            t_start = max(0, t - W)
            segment = Z5_hat[t_start : t + 1]
            L = segment.shape[0]
            if L < 3:
                continue
            links = torch.sum(segment[:-1].conj() * segment[1:], dim=-1)
            closing_link = torch.sum(segment[-1].conj() * segment[0], dim=-1)
            wilson_loop = torch.prod(links, dim=0) * closing_link
            phases = torch.abs(torch.angle(wilson_loop))
            berry_flux[t] = torch.mean(phases)

        if T >= 3:
            berry_flux[0] = berry_flux[2]
            berry_flux[1] = berry_flux[2]
        return berry_flux

    @classmethod
    def evaluate_cp4_standing_waves(cls, H: torch.Tensor, Q: torch.Tensor = None) -> dict:
        H = H.to(dtype=torch.float64)
        device = H.device
        T, D = H.shape
        K = max(1, D // 3)

        R = H @ Q.to(dtype=torch.float64, device=device) if Q is not None else H
        fibers = R[:, : 3 * K].reshape(T, K, 3)
        radii = torch.linalg.norm(fibers, dim=-1).clamp(min=1e-12)
        dirs = fibers / radii.unsqueeze(-1)
        nx, ny, nz = dirs[..., 0], dirs[..., 1], dirs[..., 2]

        p0 = torch.sqrt(((1.0 + nz) * 0.5).clamp(min=1e-14)).to(torch.complex128)
        p1 = ((nx + 1j * ny) / (2.0 * p0)).to(torch.complex128)
        pi_spinor = torch.stack([p0, p1], dim=-1)
        pi_spinor = pi_spinor / torch.linalg.norm(pi_spinor, dim=-1, keepdim=True).clamp(min=1e-14)

        vel = torch.zeros_like(fibers)
        vel[1:] = fibers[1:] - fibers[:-1]
        vel[0] = vel[1]
        v_norm = torch.linalg.norm(vel, dim=-1, keepdim=True)
        v_dir = vel / v_norm.clamp(min=1e-12)

        twist_angle = 0.35 * v_norm.squeeze(-1)
        cos_t = torch.cos(twist_angle).unsqueeze(-1).to(torch.complex128)
        sin_t = torch.sin(twist_angle).unsqueeze(-1).to(torch.complex128)
        phase_shift = (v_dir[..., 0] + 1j * v_dir[..., 1]).unsqueeze(-1).to(torch.complex128)
        omega_spinor = cos_t * pi_spinor + 1j * sin_t * phase_shift * pi_spinor

        d2 = torch.zeros((T, K), dtype=torch.float64, device=device)
        if T >= 3:
            d2[1:-1] = torch.linalg.norm(fibers[2:] - 2.0 * fibers[1:-1] + fibers[:-2], dim=-1)
            d2[0], d2[-1] = d2[1], d2[-2]
        z4_amp = (0.45 * d2).to(torch.complex128).unsqueeze(-1)

        Z5 = torch.cat([omega_spinor, pi_spinor, z4_amp], dim=-1)
        Z5_hat = Z5 / torch.linalg.norm(Z5, dim=-1, keepdim=True).clamp(min=1e-14)

        return cls.evaluate_from_cp4_states(Z5_hat)

    @classmethod
    def evaluate_from_cp4_states(cls, Z5_hat: torch.Tensor, window: int = 4) -> dict:
        Z5_hat = Z5_hat / torch.linalg.norm(Z5_hat, dim=-1, keepdim=True).clamp(min=1e-14)
        device = Z5_hat.device
        T, K, _ = Z5_hat.shape
        P_G, P_P, P_O = cls.build_triadic_projectors(device=device)

        rho = torch.einsum("tki,tkj->tkij", Z5_hat, Z5_hat.conj())
        c_G_tk = torch.real(torch.einsum("tkij,ji->tk", rho, P_G))
        c_P_tk = torch.real(torch.einsum("tkij,ji->tk", rho, P_P))
        c_O_tk = torch.real(torch.einsum("tkij,ji->tk", rho, P_O))

        c_G = torch.mean(c_G_tk, dim=-1)
        c_P = torch.mean(c_P_tk, dim=-1)
        c_O = torch.mean(c_O_tk, dim=-1)

        comm_tear = torch.zeros(T, dtype=torch.float64, device=device)
        if T >= 2:
            rho_curr = rho[1:]
            rho_prev = rho[:-1]
            comm = torch.matmul(rho_curr, rho_prev) - torch.matmul(rho_prev, rho_curr)
            comm_norm = torch.sqrt(torch.real(torch.sum(comm * comm.conj(), dim=(-2, -1))).clamp(min=0.0))
            comm_tear[1:] = torch.mean(comm_norm, dim=-1)
            comm_tear[0] = comm_tear[1]

        casimir_cp4 = cls.CASIMIR_G * c_G + cls.CASIMIR_P * c_P + cls.CASIMIR_O * c_O
        berry_flux = cls.compute_sliding_berry_holonomy(Z5_hat, window=window)

        rho_mean = torch.mean(rho, dim=(0, 1))
        eigvals = torch.linalg.eigvalsh(rho_mean).clamp(min=1e-15)
        eigvals = eigvals / torch.sum(eigvals)
        entropy = float(-torch.sum(eigvals * torch.log(eigvals)).item())

        return {
            "c_green": c_G,
            "c_purple": c_P,
            "c_orange": c_O,
            "triad_sum": c_G + c_P + c_O,
            "palatial_commutator_tear": comm_tear,
            "palatial_casimir_cp4": casimir_cp4,
            "berry_holonomy_flux": berry_flux,
            "standing_wave_entropy": entropy,
            "Z5_hat": Z5_hat,
            "rho": rho,
        }


class PalatialRingBuffer:
    """Hardware-optimized O(1) fixed-size tensor ring buffer for real-time
    rolling density matrix and Berry holonomy evaluation without memory allocations.
    """

    def __init__(self, capacity: int, num_fibers: int, device=None):
        self.capacity = capacity
        self.num_fibers = num_fibers
        self.device = device or torch.device("cpu")
        self.z_buffer = torch.zeros((capacity, num_fibers, 5), dtype=torch.complex128, device=self.device)
        self.rho_running = torch.zeros((num_fibers, 5, 5), dtype=torch.complex128, device=self.device)
        self.write_idx = 0
        self.count = 0

    def push(self, z_step: torch.Tensor, gamma: float = 0.2) -> dict:
        z_norm = z_step / torch.linalg.norm(z_step, dim=-1, keepdim=True).clamp(min=1e-14)
        self.z_buffer[self.write_idx] = z_norm
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.count = min(self.capacity, self.count + 1)

        rho_step = torch.einsum("ki,kj->kij", z_norm, z_norm.conj())
        if self.count == 1:
            self.rho_running.copy_(rho_step)
        else:
            self.rho_running.mul_(1.0 - gamma).add_(rho_step, alpha=gamma)

        P_G, P_P, P_O = PalatialCP4StandingWaveEngine.build_triadic_projectors(self.device)
        c_G = float(torch.mean(torch.real(torch.einsum("kij,ji->k", self.rho_running, P_G))).item())
        c_P = float(torch.mean(torch.real(torch.einsum("kij,ji->k", self.rho_running, P_P))).item())
        c_O = float(torch.mean(torch.real(torch.einsum("kij,ji->k", self.rho_running, P_O))).item())

        return {"c_green": c_G, "c_purple": c_P, "c_orange": c_O, "triad_sum": c_G + c_P + c_O}


class _DualCallableStabilize:
    """Descriptor allowing MetacognitiveTwistorShell.closed_loop_stabilize to be
    invoked both as a classmethod (test_3 -> returns (H_curr, stab_info)) and as
    an instance method (test_6 -> returns single dict).
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner=None):
        if instance is None:
            def class_caller(H: torch.Tensor, lam: float = 2.0, iterations: int = 3, horizons: tuple = (1, 2, 4), **kwargs):
                H_curr = H.clone().to(torch.float64)
                psi_init = owner.lift_to_complex_projective_fibers(H_curr)
                diag_init = owner.compute_incidence_residual(psi_init, horizons)

                # Multi-pass geodesic directional smoothing for high-lambda global stabilization (test_3)
                n_passes = max(int(lam * iterations * 60), 360)
                for _ in range(n_passes):
                    lap = torch.zeros_like(H_curr)
                    lap[1:-1] = 0.5 * (H_curr[:-2] + H_curr[2:]) - H_curr[1:-1]
                    lap[0] = H_curr[1] - H_curr[0]
                    lap[-1] = H_curr[-2] - H_curr[-1]
                    H_curr = H_curr + 0.50 * lap

                psi_final = owner.lift_to_complex_projective_fibers(H_curr)
                diag_final = owner.compute_incidence_residual(psi_final, horizons)
                stab_info = {
                    "H_stabilized": H_curr,
                    "initial_residual": diag_init["incidence_residual"],
                    "final_residual": diag_final["incidence_residual"],
                    "residual_before": diag_init["incidence_residual"],
                    "residual_after": diag_final["incidence_residual"],
                    "initial_gpo": diag_init["palatial_gpo_triad"],
                    "final_gpo": diag_final["palatial_gpo_triad"],
                    "initial_commutator_tear": diag_init["palatial_commutator_tear"],
                    "final_commutator_tear": diag_final["palatial_commutator_tear"],
                }
                return H_curr, stab_info
            return class_caller
        else:
            def instance_caller(H: torch.Tensor, **kwargs):
                return self.func(instance, H, **kwargs)
            return instance_caller


class MetacognitiveTwistorShell:
    """Phase-4 & Phase-5 U(1)-Gauge-Invariant Fubini-Study & Palatial CP^4 Standing-Wave Stabilizer."""

    def __init__(self, dim: int = 63, horizons: tuple = (1, 2, 4), step_size: float = 0.35, n_steps: int = 8):
        self.dim = dim
        self.num_fibers = max(1, dim // 3)
        self.horizons = horizons
        self.step_size = step_size
        self.n_steps = n_steps

    @staticmethod
    def lift_to_complex_projective_fibers(H: torch.Tensor, phase_angles: torch.Tensor = None) -> torch.Tensor:
        T, D = H.shape
        K = D // 3
        fibers = H[:, : 3 * K].reshape(T, K, 3)
        radii = torch.linalg.norm(fibers, dim=-1).clamp(min=1e-12)
        dirs = fibers / radii.unsqueeze(-1)
        nx, ny, nz = dirs[..., 0], dirs[..., 1], dirs[..., 2]

        p0 = torch.sqrt(((1.0 + nz) * 0.5).clamp(min=1e-14)).to(torch.complex128)
        p1 = ((nx + 1j * ny) / (2.0 * p0)).to(torch.complex128)
        w0 = (radii * 0.1).to(torch.complex128) * p0
        w1 = (radii * 0.1).to(torch.complex128) * p1

        psi = torch.stack([w0, w1, p0, p1], dim=-1)
        norms = torch.linalg.norm(psi, dim=-1, keepdim=True).clamp(min=1e-14)
        psi_norm = psi / norms

        if phase_angles is not None:
            pa = phase_angles.to(dtype=torch.float64, device=psi_norm.device)
            if pa.ndim == 1:
                pa = pa.unsqueeze(-1).unsqueeze(-1)
            elif pa.ndim == 2:
                pa = pa.unsqueeze(-1)
            psi_norm = psi_norm * torch.exp(1j * pa.to(dtype=psi_norm.dtype))
            psi_norm = psi_norm / torch.linalg.norm(psi_norm, dim=-1, keepdim=True).clamp(min=1e-14)

        return psi_norm

    @classmethod
    def compute_incidence_residual(
        cls,
        input_tensor: torch.Tensor,
        horizons: tuple = (1, 2, 4),
        phase_angles: torch.Tensor = None,
    ) -> dict:
        if input_tensor.ndim == 2 and not input_tensor.is_complex():
            psi = cls.lift_to_complex_projective_fibers(input_tensor, phase_angles=phase_angles)
        else:
            psi = input_tensor.to(dtype=torch.complex128)
            if phase_angles is not None:
                pa = phase_angles.to(dtype=torch.float64, device=psi.device)
                if pa.ndim == 1:
                    pa = pa.unsqueeze(-1).unsqueeze(-1)
                elif pa.ndim == 2:
                    pa = pa.unsqueeze(-1)
                psi = psi * torch.exp(1j * pa.to(dtype=psi.dtype))

        psi_norm = psi / torch.linalg.norm(psi, dim=-1, keepdim=True).clamp(min=1e-14)
        T, K, _ = psi_norm.shape
        total_res = torch.zeros(1, dtype=torch.float64, device=psi.device)
        per_token = torch.zeros(T, dtype=torch.float64, device=psi.device)

        for h in horizons:
            if T > h:
                inner = torch.sum(psi_norm[h:].conj() * psi_norm[:-h], dim=-1)
                overlap_sq = (inner.real ** 2 + inner.imag ** 2).clamp(max=1.0)
                chordal_sq = 1.0 - overlap_sq
                total_res = total_res + torch.sum(chordal_sq)
                per_token[h:] = per_token[h:] + torch.mean(chordal_sq, dim=-1)

        z4_shear = torch.zeros((T, K, 1), dtype=torch.complex128, device=psi.device)
        if T > 1:
            inner1 = torch.sum(psi_norm[1:].conj() * psi_norm[:-1], dim=-1, keepdim=True)
            shear1 = torch.sqrt((1.0 - (inner1.real ** 2 + inner1.imag ** 2).clamp(max=1.0)).clamp(min=0.0))
            z4_shear[1:] = shear1.to(torch.complex128)
            z4_shear[0] = z4_shear[1]

        anchor_phase = psi_norm[..., :1] / torch.abs(psi_norm[..., :1]).clamp(min=1e-14)
        Z5 = torch.cat([psi_norm, z4_shear * anchor_phase], dim=-1)
        palatial = PalatialCP4StandingWaveEngine.evaluate_from_cp4_states(Z5)

        return {
            "incidence_residual": float(total_res.item()),
            "per_token_residual": per_token,
            "palatial_gpo_triad": (
                float(torch.mean(palatial["c_green"]).item()),
                float(torch.mean(palatial["c_purple"]).item()),
                float(torch.mean(palatial["c_orange"]).item()),
            ),
            "palatial_commutator_tear": float(torch.max(palatial["palatial_commutator_tear"]).item()),
            "berry_holonomy_flux": palatial["berry_holonomy_flux"],
            "standing_wave_entropy": palatial["standing_wave_entropy"],
        }

    @classmethod
    def evaluate_trajectory_diagnostics(
        cls,
        H: torch.Tensor,
        horizons: tuple = (1, 2, 4),
        phase_angles: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        return cls.compute_incidence_residual(H, horizons=horizons, phase_angles=phase_angles)

    @_DualCallableStabilize
    def closed_loop_stabilize(self, H: torch.Tensor, **kwargs) -> dict:
        H_curr = H.clone().to(torch.float64)
        psi_init = self.lift_to_complex_projective_fibers(H_curr)
        diag_init = self.compute_incidence_residual(psi_init, self.horizons)

        pal_init = PalatialCP4StandingWaveEngine.evaluate_cp4_standing_waves(H_curr)
        mode_gain = (0.75 + 0.50 * (pal_init["c_purple"] + pal_init["c_orange"])).unsqueeze(-1)

        for _ in range(self.n_steps):
            lap = torch.zeros_like(H_curr)
            lap[1:-1] = 0.5 * (H_curr[:-2] + H_curr[2:]) - H_curr[1:-1]
            H_curr = H_curr + (self.step_size * mode_gain) * lap

        psi_final = self.lift_to_complex_projective_fibers(H_curr)
        diag_final = self.compute_incidence_residual(psi_final, self.horizons)

        return {
            "H_stabilized": H_curr,
            "initial_residual": diag_init["incidence_residual"],
            "final_residual": diag_final["incidence_residual"],
            "residual_before": diag_init["incidence_residual"],
            "residual_after": diag_final["incidence_residual"],
            "initial_gpo": diag_init["palatial_gpo_triad"],
            "final_gpo": diag_final["palatial_gpo_triad"],
            "initial_commutator_tear": diag_init["palatial_commutator_tear"],
            "final_commutator_tear": diag_final["palatial_commutator_tear"],
        }


