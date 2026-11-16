"""
AI Vector Shield - Project Argon v6.0 Core Mathematical Module
Implements SU(2,3) indefinite-metric Berry holonomy, exact O(1) FIFO sliding-window
density matrices, remainder-safe 3D fiber extraction, and reference-tube anomaly detection.
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
    eps: float = 1e-8
) -> Dict[str, torch.Tensor]:
    """
    Computes discrete closed-loop Berry geometric phase W = prod <Z_j, Z_{j+1}> <Z_m, Z_0>.
    Supports both the SU(2,3) indefinite Hermitian pairing <Z_j, Sigma_5 Z_{j+1}> and the
    positive-definite Euclidean inner product, with explicit overlap degeneracy guards.
    """
    Z_c = Z_loop.to(torch.complex128)
    Z_next = torch.roll(Z_c, shifts=-1, dims=-2)
    
    if use_indefinite_metric:
        sigma = SIGMA_5.to(Z_c.device)
        paired_next = torch.matmul(Z_next, sigma.T)
    else:
        paired_next = Z_next
        
    overlaps = torch.sum(Z_c.conj() * paired_next, dim=-1)
    mags = torch.abs(overlaps)
    
    # Degeneracy guard: flag loops where any adjacent link approaches orthogonality / null cone
    valid_mask = torch.all(mags > eps, dim=-1)
    min_overlap = torch.min(mags, dim=-1).values
    
    unit_links = torch.where(
        mags > eps,
        overlaps / mags.clamp(min=eps),
        torch.ones_like(overlaps)
    )
    wilson_loop = torch.prod(unit_links, dim=-1)
    flux = torch.where(valid_mask, torch.abs(torch.angle(wilson_loop)), torch.zeros_like(min_overlap))
    
    return {
        "berry_flux": flux,
        "wilson_loop": wilson_loop,
        "valid_mask": valid_mask,
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
        """
        Inserts a normalized (num_fibers, 5) state in O(1) time and returns the current (num_fibers, 5, 5) density matrix.
        """
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
    """
    Reshapes arbitrary feature dimension d into 3D fibers without truncating remainder dimensions.
    Pads trailing dimensions with zeros when d % 3 != 0 and returns (fibers, pad_len).
    """
    d = x.shape[-1]
    rem = d % 3
    pad_len = (3 - rem) % 3
    if pad_len > 0:
        x = torch.nn.functional.pad(x, (0, pad_len), mode="constant", value=0.0)
    fibers = x.reshape(*x.shape[:-1], -1, 3)
    return fibers, pad_len
