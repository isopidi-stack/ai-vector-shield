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
