"""
AI Vector Shield - Project Argon v6.1 Core Mathematical Module
Adversarially hardened:
  1. Krein-space SU(2,3) Pancharatnam-Berry connection with pseudo-norm normalization
     and sector-flip (timelike/spacelike/null) signature tracking.
  2. Exact O(1) FIFO sliding-window density matrix alongside EWMA mode.
  3. Remainder-safe 3D fiber extraction.
"""

import torch
from typing import Tuple, Dict

SIGMA_5 = torch.tensor([
    [0., 0., 1., 0., 0.],
    [0., 0., 0., 1., 0.],
    [1., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 1.]
], dtype=torch.complex128)

CP4_CASIMIR_SPECTRUM = (0.0, 20.0, 48.0)


def compute_berry_holonomy(
    Z_loop: torch.Tensor,
    use_indefinite_metric: bool = True,
    rectify_krein_sector: bool = True,
    eps: float = 1e-8
) -> Dict[str, torch.Tensor]:
    """
    Computes discrete closed-loop Pancharatnam-Berry geometric phase in projective space CP^4.
     invariant under general non-zero complex scaling Z_j -> c_j Z_j (c_j in C^x).
    When use_indefinite_metric=True, normalizes links by the Krein pseudo-norm
    sqrt(|<Z_j, Sigma_5 Z_j>| |<Z_{j+1}, Sigma_5 Z_{j+1}>|) and tracks Krein signature sector flips
    so timelike/spacelike sector signs are separated from geometric holonomy.
    """
    Z_c = Z_loop.to(torch.complex128)
    Z_next = torch.roll(Z_c, shifts=-1, dims=-2)

    if use_indefinite_metric:
        sigma = SIGMA_5.to(Z_c.device)
        paired_curr = torch.matmul(Z_c, sigma.T)
        paired_next = torch.matmul(Z_next, sigma.T)
    else:
        paired_curr = Z_c
        paired_next = Z_next

    # Self pseudo-norms <Z_j, M Z_j> (real for Hermitian M)
    self_norms = torch.sum(Z_c.conj() * paired_curr, dim=-1).real
    self_norms_next = torch.roll(self_norms, shifts=-1, dims=-1)
    null_state_mask = torch.any(torch.abs(self_norms) <= eps, dim=-1)

    # Raw link overlaps <Z_j, M Z_{j+1}>
    raw_overlaps = torch.sum(Z_c.conj() * paired_next, dim=-1)

    # Rectify Krein sector sign if requested so negative-norm states (-1) don't inject trivial pi phases
    if use_indefinite_metric and rectify_krein_sector:
        sector_signs = torch.where(self_norms >= 0.0, torch.ones_like(self_norms), -torch.ones_like(self_norms))
        overlaps = raw_overlaps * sector_signs
    else:
        overlaps = raw_overlaps

    # Scale-invariant Krein link magnitude: |<Z_j, M Z_{j+1}>| / sqrt(|<Z_j, M Z_j>| |<Z_{j+1}, M Z_{j+1}>|)
    denom = torch.sqrt((torch.abs(self_norms) * torch.abs(self_norms_next)).clamp(min=eps * eps))
    normalized_link_mags = torch.abs(overlaps) / denom

    mags = torch.abs(overlaps)
    valid_mask = torch.all(mags > eps, dim=-1) & (~null_state_mask)
    min_overlap = torch.min(normalized_link_mags, dim=-1).values

    unit_links = torch.where(
        mags > eps,
        overlaps / mags.clamp(min=eps),
        torch.ones_like(overlaps)
    )
    wilson_loop = torch.prod(unit_links, dim=-1)
    flux = torch.where(valid_mask, torch.abs(torch.angle(wilson_loop)), torch.zeros_like(min_overlap))

    # Count how many times the loop crosses between positive and negative Krein sectors
    sector_flips = torch.sum((self_norms * self_norms_next < 0.0).to(torch.int64), dim=-1)

    return {
        "berry_flux": flux,
        "wilson_loop": wilson_loop,
        "valid_mask": valid_mask,
        "null_state_mask": null_state_mask,
        "sector_flips": sector_flips,
        "min_overlap": min_overlap
    }


class PalatialRingBuffer:
    """
    Ring buffer supporting both:
      1. Exact O(1) sliding-window average density matrix over the last N states (mode='exact_window')
      2. Exponentially weighted moving average density matrix (mode='ewma')
    """
    def __init__(self, num_fibers: int, capacity: int, mode: str = "exact_window", gamma: float = 0.1, device: str = "cpu"):
        if mode not in ("exact_window", "ewma"):
            raise ValueError("mode must be 'exact_window' or 'ewma'")
        self.num_fibers = num_fibers
        self.capacity = capacity
        self.mode = mode
        self.gamma = gamma
        self.device = device

        self.buffer = torch.zeros((capacity, num_fibers, 5), dtype=torch.complex128, device=device)
        self.rho_accum = torch.zeros((num_fibers, 5, 5), dtype=torch.complex128, device=device)
        self.ptr = 0
        self.count = 0

    def push(self, Z5_state: torch.Tensor) -> torch.Tensor:
        Z_in = Z5_state.to(dtype=torch.complex128, device=self.device)
        Z_in = Z_in / torch.linalg.norm(Z_in, dim=-1, keepdim=True).clamp(min=1e-12)
        proj_in = torch.einsum("fi,fj->fij", Z_in, Z_in.conj())

        if self.mode == "exact_window":
            if self.count == self.capacity:
                Z_out = self.buffer[self.ptr]
                proj_out = torch.einsum("fi,fj->fij", Z_out, Z_out.conj())
                self.rho_accum = self.rho_accum - proj_out + proj_in
            else:
                self.rho_accum = self.rho_accum + proj_in
                self.count += 1
            self.buffer[self.ptr] = Z_in
            self.ptr = (self.ptr + 1) % self.capacity
            return self.rho_accum / float(self.count)
        else:
            if self.count == 0:
                self.rho_accum = proj_in
                self.count = 1
            else:
                self.rho_accum = (1.0 - self.gamma) * self.rho_accum + self.gamma * proj_in
                self.count = min(self.count + 1, self.capacity)
            self.buffer[self.ptr] = Z_in
            self.ptr = (self.ptr + 1) % self.capacity
            return self.rho_accum


def extract_3d_fibers_padded(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    d = x.shape[-1]
    rem = d % 3
    pad_len = (3 - rem) % 3
    if pad_len > 0:
        x = torch.nn.functional.pad(x, (0, pad_len), mode="constant", value=0.0)
    fibers = x.reshape(*x.shape[:-1], -1, 3)
    return fibers, pad_len


# =========================================================================
# PENROSE POSTULATES EXTENSION: 5-QUANTUM-NUMBER DUAL SPINORS & 25 FREEDOMS
# =========================================================================

def compute_sigma5_freedom_matrix(Z5: torch.Tensor) -> torch.Tensor:
    """
    Constructs the 5x5 (25-element) system freedom operator M^alpha_beta = Z^alpha (Sigma_5 Z)_beta^*
    representing all 25 coupled degrees of freedom governed by SIGMA_5 on C^5.
    Satisfies Tr(M) == Z_5^dagger Sigma_5 Z_5.
    """
    Z_c = Z5.to(torch.complex128)
    sigma = SIGMA_5.to(Z_c.device)
    Z_dual = torch.matmul(Z_c.conj(), sigma.T)
    return torch.einsum("...i,...j->...ij", Z_c, Z_dual)


def build_five_quantum_twistor_state(
    n_principal: torch.Tensor,
    l_azimuthal: torch.Tensor,
    m_magnetic: torch.Tensor,
    s_electron_omega: torch.Tensor,
    s_core_pi: torch.Tensor,
    L0_scale: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Maps the 5 Quantum Numbers (n, l, m_l, s_omega, s_pi) into the 5-component
    twistor Z_5 = (omega^0, omega^1, pi_0', pi_1', phi) in C^5.
    Pairs the electron Weyl spinor omega^A with the conjugate core Weyl spinor pi_{A'}
    across all 25 elements of SIGMA_5.
    """
    n_f = n_principal.to(torch.float64)
    l_f = l_azimuthal.to(torch.float64)
    m_f = m_magnetic.to(torch.float64)
    sw_f = s_electron_omega.to(torch.float64)
    sp_f = s_core_pi.to(torch.float64)

    l_norm = torch.sqrt((l_f * (l_f + 1.0)).clamp(min=1.0))
    cos_theta = (m_f / l_norm).clamp(-0.999, 0.999)
    half_theta = 0.5 * torch.acos(cos_theta)

    phase_omega = torch.exp(1j * sw_f * torch.pi)
    omega_0 = torch.cos(half_theta).to(torch.complex128) * phase_omega
    omega_1 = torch.sin(half_theta).to(torch.complex128) * phase_omega.conj()
    omega_A = torch.stack([omega_0, omega_1], dim=-1) * (n_f.unsqueeze(-1) / float(L0_scale))

    phase_pi = torch.exp(1j * sp_f * torch.pi)
    pi_0 = torch.cos(half_theta).to(torch.complex128) * phase_pi
    pi_1 = torch.sin(half_theta).to(torch.complex128) * phase_pi.conj()
    pi_A_prime = torch.stack([pi_0, pi_1], dim=-1)

    phi_shell = (1.0 / n_f.clamp(min=1.0)).to(torch.complex128).unsqueeze(-1)

    Z5_complete = torch.cat([omega_A, pi_A_prime, phi_shell], dim=-1)
    Z5_incomplete_4qn = torch.cat([omega_A, torch.zeros_like(pi_A_prime), phi_shell], dim=-1)

    freedom_matrix_25 = compute_sigma5_freedom_matrix(Z5_complete)
    sigma = SIGMA_5.to(Z5_complete.device)
    norm_complete = torch.einsum("...i,ij,...j->...", Z5_complete.conj(), sigma, Z5_complete).real
    norm_incomplete = torch.einsum("...i,ij,...j->...", Z5_incomplete_4qn.conj(), sigma, Z5_incomplete_4qn).real

    return {
        "Z5_complete": Z5_complete,
        "Z5_incomplete_4qn": Z5_incomplete_4qn,
        "freedom_matrix_25": freedom_matrix_25,
        "sigma5_norm_complete": norm_complete,
        "sigma5_norm_incomplete": norm_incomplete,
        "core_electron_spinor_coupling": norm_complete - norm_incomplete,
    }


def apply_googly_chiral_patch_coupling(
    Z_left: torch.Tensor,
    Z_right: torch.Tensor,
    epsilon_googly: float = 0.05
) -> Dict[str, torch.Tensor]:
    """
    Applies unitary non-commutative Heisenberg operator patch coupling between
    left-handed and right-handed twistor bundles while preserving C^5 norms.
    """
    Z_l = Z_left.to(torch.complex128)
    Z_r = Z_right.to(torch.complex128)
    P_L = torch.einsum("...i,...j->...ij", Z_l, Z_l.conj())
    P_R = torch.einsum("...i,...j->...ij", Z_r, Z_r.conj())

    comm_gen = 1j * (torch.matmul(P_L, P_R) - torch.matmul(P_R, P_L))
    U_googly = torch.matrix_exp(1j * float(epsilon_googly) * comm_gen)

    Z_l_out = torch.matmul(U_googly, Z_l.unsqueeze(-1)).squeeze(-1)
    Z_r_out = torch.matmul(U_googly.conj().transpose(-1, -2), Z_r.unsqueeze(-1)).squeeze(-1)

    return {
        "Z_left": Z_l_out,
        "Z_right": Z_r_out,
        "U_googly": U_googly,
        "freedom_matrix_25_left": compute_sigma5_freedom_matrix(Z_l_out),
    }


# =========================================================================
# CLOSED-LOOP VOLUMETRIC TELEMETRY & SELF-MONITORING PROTECTION SUITE
# =========================================================================

class MetacognitiveShieldProtector:
    """
    Couples ai-vector-shield's 25-element SIGMA_5 freedom matrix and Krein-space
    Berry holonomy with a continuous erf() volumetric manifold projection.
    Measures structural Dirichlet roughness and geometric phase shear in a closed
    feedback loop to automatically damp and rectify adversarial trajectory drift.
    """
    def __init__(self, resolution: int = 24, roughness_gain: float = 0.35, holonomy_gain: float = 0.25):
        self.resolution = int(resolution)
        self.roughness_gain = float(roughness_gain)
        self.holonomy_gain = float(holonomy_gain)

    def render_shield_manifold_field(self, freedom_matrix_25: torch.Tensor) -> torch.Tensor:
        """
        Projects the 5x5 (25-element) SIGMA_5 freedom tensor onto an analytical
        Z-buffer-free 2D density field using continuous erf() radial cell envelopes.
        """
        M_mag = torch.abs(freedom_matrix_25)
        while M_mag.dim() > 2:
            M_mag = M_mag.mean(dim=0)
        M_norm = M_mag / M_mag.max().clamp(min=1e-12)

        coords = torch.linspace(-1.0, 1.0, self.resolution, dtype=torch.float64, device=M_norm.device)
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        centers = torch.linspace(-0.75, 0.75, 5, dtype=torch.float64, device=M_norm.device)

        field = torch.zeros((self.resolution, self.resolution), dtype=torch.float64, device=M_norm.device)
        radius = 0.16
        softness = 0.05

        for r in range(5):
            for c in range(5):
                dist = torch.sqrt((gx - centers[c]) ** 2 + (gy - centers[r]) ** 2 + 1e-12)
                envelope = 0.5 * (1.0 - torch.erf((dist - radius) / softness))
                field = field + M_norm[r, c] * envelope

        return field

    def evaluate_metacognitive_shield(self, Z_loop: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Executes the closed-loop protection cycle over a sequence of C^5 twistor states:
          1. Evaluates the 25-element SIGMA_5 freedom tensor M^alpha_beta across the loop.
          2. Renders the internal 2D manifold density field via analytical erf() optics.
          3. Computes Krein-space Berry holonomy and spatial manifold roughness energy.
          4. Applies a closed-loop protective damping gate to rectify adversarial drift.
        """
        Z_c = Z_loop.to(torch.complex128)
        M_25 = compute_sigma5_freedom_matrix(Z_c)
        manifold_field = self.render_shield_manifold_field(M_25)

        # Spatial Dirichlet roughness energy on the rendered manifold
        dy = manifold_field[1:, :] - manifold_field[:-1, :]
        dx = manifold_field[:, 1:] - manifold_field[:, :-1]
        roughness_energy = (dy ** 2).mean() + (dx ** 2).mean()

        # Indefinite Krein-space Berry holonomy across the trajectory loop
        holonomy_res = compute_berry_holonomy(Z_c, use_indefinite_metric=True, rectify_krein_sector=True)
        berry_phase_mag = torch.abs(holonomy_res["berry_phase"]).mean()

        # Closed-loop attenuation gate triggered by manifold roughness + phase shear
        protection_gate = torch.exp(
            -(self.roughness_gain * roughness_energy + self.holonomy_gain * berry_phase_mag)
        )

        # Rectify the trajectory and re-normalize in C^5
        Z_rectified = Z_c * protection_gate
        Z_rectified = Z_rectified / torch.linalg.norm(Z_rectified, dim=-1, keepdim=True).clamp(min=1e-12)

        return {
            "Z_rectified": Z_rectified,
            "freedom_matrix_25": M_25,
            "rendered_manifold": manifold_field,
            "roughness_energy": roughness_energy,
            "berry_phase": holonomy_res["berry_phase"],
            "sector_flip_count": holonomy_res["sector_flip_count"],
            "protection_gate": protection_gate,
        }
