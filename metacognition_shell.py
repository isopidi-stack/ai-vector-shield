import math
import torch


class PalatialCP4StandingWaveEngine:
    """
    Penrose Palatial Twistor Standing-Wave Engine on CP^4 (C^5).
    Lifts token trajectories into 5D homogeneous twistor-horizon coordinates
    Z^I = (omega^0, omega^1, pi_0', pi_1', phi_horizon) in C^5, evaluates the
    non-commutative Palatial density commutator [rho_t, rho_{t-1}], and decomposes
    the standing-wave spectrum into the exact triadic Green-Purple-Orange (GPO)
    partition of unity:
        c_Green(t) + c_Purple(t) + c_Orange(t) == 1.0  (exact to 1e-16 float64).
    """

    @staticmethod
    def build_triadic_projectors(device=None) -> tuple:
        device = device or torch.device("cpu")
        dtype = torch.complex128

        # Orthonormal basis vectors spanning C^5:
        # Green (2D equatorial incidence standing-wave subspace: (omega + pi)/sqrt(2))
        P_G = torch.zeros((5, 5), dtype=dtype, device=device)
        for a in (0, 1):
            v = torch.zeros((5, 1), dtype=dtype, device=device)
            v[a, 0] = 1.0 / math.sqrt(2.0)
            v[a + 2, 0] = 1.0 / math.sqrt(2.0)
            P_G = P_G + (v @ v.conj().T)

        # Purple (2D chiral helicity / twist standing-wave subspace: (omega - pi)/sqrt(2))
        P_P = torch.zeros((5, 5), dtype=dtype, device=device)
        for a in (0, 1):
            v = torch.zeros((5, 1), dtype=dtype, device=device)
            v[a, 0] = 1.0 / math.sqrt(2.0)
            v[a + 2, 0] = -1.0 / math.sqrt(2.0)
            P_P = P_P + (v @ v.conj().T)

        # Orange (1D Palatial cosmological / curvature horizon mode Z^4)
        P_O = torch.zeros((5, 5), dtype=dtype, device=device)
        P_O[4, 4] = 1.0 + 0.0j

        return P_G, P_P, P_O

    @classmethod
    def evaluate_cp4_standing_waves(cls, H: torch.Tensor, Q: torch.Tensor = None) -> dict:
        H = H.to(dtype=torch.float64)
        device = H.device
        T, D = H.shape
        K = max(1, D // 3)

        if Q is not None:
            R = H @ Q.to(dtype=torch.float64, device=device)
        else:
            R = H

        fibers = R[:, : 3 * K].reshape(T, K, 3)
        radii = torch.linalg.norm(fibers, dim=-1).clamp(min=1e-12)
        dirs = fibers / radii.unsqueeze(-1)
        nx, ny, nz = dirs[..., 0], dirs[..., 1], dirs[..., 2]

        # Stereographic / Hopf spinor lift pi_{A'} on CP^1 (U(1)-gauge covariant)
        p0 = torch.sqrt(((1.0 + nz) * 0.5).clamp(min=1e-14)).to(torch.complex128)
        p1 = ((nx + 1j * ny) / (2.0 * p0)).to(torch.complex128)
        pi_spinor = torch.stack([p0, p1], dim=-1)
        pi_norm = torch.linalg.norm(pi_spinor, dim=-1, keepdim=True).clamp(min=1e-14)
        pi_spinor = pi_spinor / pi_norm

        # Construct Palatial incidence spinor omega^A:
        # At geodesic equilibrium, omega^A aligns with pi_{A'} (Green mode), while
        # local velocity/twist induces chiral phase displacement (Purple mode)
        vel = torch.zeros_like(fibers)
        vel[1:] = fibers[1:] - fibers[:-1]
        vel[0] = vel[1]
        v_norm = torch.linalg.norm(vel, dim=-1, keepdim=True)
        v_dir = vel / v_norm.clamp(min=1e-12)

        # Palatial chiral coupling angle from local velocity & radius modulation
        twist_angle = 0.35 * v_norm.squeeze(-1)
        cos_t = torch.cos(twist_angle).unsqueeze(-1).to(torch.complex128)
        sin_t = torch.sin(twist_angle).unsqueeze(-1).to(torch.complex128)
        phase_shift = (v_dir[..., 0] + 1j * v_dir[..., 1]).unsqueeze(-1).to(torch.complex128)
        omega_spinor = cos_t * pi_spinor + 1j * sin_t * phase_shift * pi_spinor

        # 5th Homogeneous Coordinate Z^4 (Orange Palatial curvature / action horizon)
        d2 = torch.zeros((T, K), dtype=torch.float64, device=device)
        if T >= 3:
            d2[1:-1] = torch.linalg.norm(fibers[2:] - 2.0 * fibers[1:-1] + fibers[:-2], dim=-1)
            d2[0], d2[-1] = d2[1], d2[-2]
        z4_amp = (0.45 * d2).to(torch.complex128).unsqueeze(-1)

        # Assemble 5D homogeneous state Z^I in C^5 and normalize to unit S^9 sphere (CP^4)
        Z5 = torch.cat([omega_spinor, pi_spinor, z4_amp], dim=-1)  # (T, K, 5)
        Z5_norm = torch.linalg.norm(Z5, dim=-1, keepdim=True).clamp(min=1e-14)
        Z5_hat = Z5 / Z5_norm  # (T, K, 5)

        return cls.evaluate_from_cp4_states(Z5_hat)

    @classmethod
    def evaluate_from_cp4_states(cls, Z5_hat: torch.Tensor) -> dict:
        """
        Directly evaluates standing-wave GPO triad and non-commutative Palatial
        commutator tear from unit CP^4 states Z5_hat of shape (T, K, 5).
        Strictly invariant under arbitrary local U(1) gauge transformations
        Z5_hat[t, k] -> exp(i * theta_{t,k}) * Z5_hat[t, k].
        """
        Z5_hat = Z5_hat / torch.linalg.norm(Z5_hat, dim=-1, keepdim=True).clamp(min=1e-14)
        device = Z5_hat.device
        T, K, _ = Z5_hat.shape
        P_G, P_P, P_O = cls.build_triadic_projectors(device=device)

        # Pure-state rank-1 Palatial density matrices rho_{t,k} = |Z><Z| on CP^4
        # Notice rho_{t,k} is 100% invariant under local U(1) phase Z -> e^{i theta} Z!
        rho = torch.einsum("tki,tkj->tkij", Z5_hat, Z5_hat.conj())  # (T, K, 5, 5)

        c_G_tk = torch.real(torch.einsum("tkij,ji->tk", rho, P_G))
        c_P_tk = torch.real(torch.einsum("tkij,ji->tk", rho, P_P))
        c_O_tk = torch.real(torch.einsum("tkij,ji->tk", rho, P_O))

        c_G = torch.mean(c_G_tk, dim=-1)  # (T,)
        c_P = torch.mean(c_P_tk, dim=-1)  # (T,)
        c_O = torch.mean(c_O_tk, dim=-1)  # (T,)

        # Non-commutative Palatial commutator [rho_t, rho_{t-1}] Frobenius norm
        comm_tear = torch.zeros(T, dtype=torch.float64, device=device)
        if T >= 2:
            rho_curr = rho[1:]
            rho_prev = rho[:-1]
            comm = torch.matmul(rho_curr, rho_prev) - torch.matmul(rho_prev, rho_curr)
            comm_norm = torch.sqrt(torch.real(torch.sum(comm * comm.conj(), dim=(-2, -1))).clamp(min=0.0))
            comm_tear[1:] = torch.mean(comm_norm, dim=-1)
            comm_tear[0] = comm_tear[1]

        # Palatial Standing-Wave Casimir Eigenvalue Functional (E_G=0, E_P=12, E_O=32 on CP^4)
        casimir_cp4 = 0.0 * c_G + 12.0 * c_P + 32.0 * c_O

        # Running cavity resonance density matrix (standing-wave coherence across time)
        rho_mean = torch.mean(rho, dim=(0, 1))  # (5, 5)
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
            "standing_wave_entropy": entropy,
            "Z5_hat": Z5_hat,
        }


class MetacognitiveTwistorShell:
    """
    Phase-4 & Phase-5 U(1)-Gauge-Invariant Fubini-Study & Palatial CP^4 Standing-Wave Stabilizer.
    """
    def __init__(self, dim: int = 63, horizons: tuple = (1, 2, 4), step_size: float = 0.35, n_steps: int = 8):
        self.dim = dim
        self.num_fibers = max(1, dim // 3)
        self.horizons = horizons
        self.step_size = step_size
        self.n_steps = n_steps

    @staticmethod
    def lift_to_complex_projective_fibers(H: torch.Tensor) -> torch.Tensor:
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
        return psi / norms

    @classmethod
    
        @classmethod
    
    @classmethod
    def evaluate_trajectory_diagnostics(cls, H, horizons=(1, 2, 4)):
        psi = cls.lift_to_complex_projective_fibers(H)
        return cls.compute_incidence_residual(psi, horizons)

    def compute_incidence_residual(cls, psi: torch.Tensor, horizons: tuple = (1, 2, 4)) -> dict:
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

        # Lift 4D projective twistor fibers to 5D Palatial CP^4 standing-wave states
        # using the local h=1 Fubini-Study chordal shear as the U(1)-invariant 5th coordinate!
        z4_shear = torch.zeros((T, K, 1), dtype=torch.complex128, device=psi.device)
        if T > 1:
            inner1 = torch.sum(psi_norm[1:].conj() * psi_norm[:-1], dim=-1, keepdim=True)
            shear1 = torch.sqrt((1.0 - (inner1.real ** 2 + inner1.imag ** 2).clamp(max=1.0)).clamp(min=0.0))
            z4_shear[1:] = shear1.to(torch.complex128)
            z4_shear[0] = z4_shear[1]
        # Attach U(1) phase of psi_norm[..., 0] so global/local phase rotations factor out identically:
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
            "standing_wave_entropy": palatial["standing_wave_entropy"],
        }

    def closed_loop_stabilize(self, H: torch.Tensor) -> dict:
        H_curr = H.clone().to(torch.float64)
        psi_init = self.lift_to_complex_projective_fibers(H_curr)
        diag_init = self.compute_incidence_residual(psi_init, self.horizons)

        # Mode-selective Palatial damping: Weight damping strength by local Orange/Purple anomaly excursion
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
            "initial_gpo": diag_init["palatial_gpo_triad"],
            "final_gpo": diag_final["palatial_gpo_triad"],
            "initial_commutator_tear": diag_init["palatial_commutator_tear"],
            "final_commutator_tear": diag_final["palatial_commutator_tear"],
        }
