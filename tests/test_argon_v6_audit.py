"""
Standalone Adversarial Audit Suite for AI Vector Shield - Project Argon v6.1
Verifies:
  1. General GL(1, C) complex scale + U(1) gauge invariance Z_j -> r_j exp(i alpha_j) Z_j
  2. Krein negative-norm sector rectification (no spurious pi phase in spacelike sector)
  3. Null-cone & orthogonal degeneracy guards
  4. Exact O(1) FIFO sliding-window density matrix
  5. Remainder-safe 3D fiber padding
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from argon_v6_core import (
    SIGMA_5,
    CP4_CASIMIR_SPECTRUM,
    compute_berry_holonomy,
    PalatialRingBuffer,
    extract_3d_fibers_padded,
)


def test_berry_holonomy_gl1c_scale_and_u1_gauge_invariance():
    """
    Verifies invariance under full projective scaling Z_j -> r_j * exp(i * alpha_j) * Z_j
    for arbitrary positive magnitudes r_j and phases alpha_j.
    """
    torch.manual_seed(7)
    Z_loop = torch.randn(8, 5, dtype=torch.complex128)
    Z_loop = Z_loop / torch.linalg.norm(Z_loop, dim=-1, keepdim=True)

    angles = torch.rand(8, 1, dtype=torch.float64) * 2.0 * torch.pi
    scales = (torch.rand(8, 1, dtype=torch.float64) * 10.0) + 0.2
    Z_scaled = Z_loop * scales * torch.exp(1j * angles)

    for use_indef in (True, False):
        res_base = compute_berry_holonomy(Z_loop, use_indefinite_metric=use_indef)
        res_scaled = compute_berry_holonomy(Z_scaled, use_indefinite_metric=use_indef)
        assert bool(res_base["valid_mask"]) is True
        assert torch.allclose(res_base["berry_flux"], res_scaled["berry_flux"], atol=1e-12)
        assert torch.allclose(res_base["min_overlap"], res_scaled["min_overlap"], atol=1e-12)


def test_krein_spacelike_sector_rectification():
    """
    Verifies that a trivial loop in the negative-norm spacelike sector of SIGMA_5
    (e.g., odd number of states near (e_0 - e_2)/sqrt(2) where Z^dagger Sigma_5 Z = -1)
    has zero Berry flux instead of a spurious pi phase from (-1)^3 = -1.
    """
    inv_s2 = 1.0 / (2.0 ** 0.5)
    v_neg = torch.tensor([inv_s2, 0.0, -inv_s2, 0.0, 0.0], dtype=torch.complex128)
    # 3 identical states in the negative-norm sector: without Krein rectification, (-1)^3 = -1 -> angle = pi!
    Z_neg_loop = torch.stack([v_neg, v_neg, v_neg], dim=0)

    res = compute_berry_holonomy(Z_neg_loop, use_indefinite_metric=True, rectify_krein_sector=True)
    assert bool(res["valid_mask"]) is True
    assert torch.allclose(res["berry_flux"], torch.tensor(0.0, dtype=torch.float64), atol=1e-12)


def test_berry_holonomy_near_orthogonal_and_null_guard():
    """Verifies null-cone states (Z^dagger Sigma_5 Z == 0) are flagged cleanly without NaN."""
    Z_deg = torch.zeros(3, 5, dtype=torch.complex128)
    Z_deg[0, 0] = 1.0  # Null vector under Sigma_5: <e_0, Sigma_5 e_0> = 0
    Z_deg[1, 0] = 1.0
    Z_deg[2, 4] = 1.0

    res = compute_berry_holonomy(Z_deg, use_indefinite_metric=True)
    assert bool(res["null_state_mask"]) is True
    assert bool(res["valid_mask"]) is False
    assert not torch.isnan(res["berry_flux"]).any()
    assert res["berry_flux"].item() == 0.0


def test_exact_sliding_window_ring_buffer_purges_old_states():
    """Verifies exact_window mode equals the exact mean of the last N projectors and purges expired states."""
    torch.manual_seed(99)
    num_fibers, capacity = 4, 5
    buf = PalatialRingBuffer(num_fibers=num_fibers, capacity=capacity, mode="exact_window")

    seq = []
    for _ in range(12):
        z = torch.randn(num_fibers, 5, dtype=torch.complex128)
        z = z / torch.linalg.norm(z, dim=-1, keepdim=True)
        seq.append(z)
        rho_stream = buf.push(z)

    last_n = torch.stack(seq[-capacity:], dim=0)
    projs = torch.einsum("nfi,nfj->nfij", last_n, last_n.conj())
    rho_exact = projs.mean(dim=0)

    assert torch.allclose(rho_stream, rho_exact, atol=1e-12)


def test_non_divisible_dimension_fiber_padding():
    """Verifies dimensions not divisible by 3 retain all features via padding."""
    for d in (63, 64, 65):
        x = torch.ones(2, 10, d, dtype=torch.float64)
        fibers, pad_len = extract_3d_fibers_padded(x)
        assert fibers.shape[-1] == 3
        assert fibers.shape[-2] * 3 == d + pad_len
        assert torch.isclose(fibers.sum(), x.sum())


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
