"""
Unit Test Suite for AI Vector Shield - Project Argon v6.0
Tests SU(2,3) Berry holonomy U(1) gauge invariance, near-orthogonal overlap guards,
exact O(1) sliding-window density matrix vs EWMA, and remainder-safe fiber extraction.
"""

import torch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from argon_v6_core import (
    SIGMA_5,
    CP4_CASIMIR_SPECTRUM,
    compute_berry_holonomy,
    PalatialRingBuffer,
    extract_3d_fibers_padded,
)


def test_berry_holonomy_u1_gauge_invariance_both_metrics():
    """Verifies local U(1) phase invariance Z_j -> exp(i alpha_j) Z_j under both Sigma_5 and Euclidean pairings."""
    torch.manual_seed(7)
    Z_loop = torch.randn(8, 5, dtype=torch.complex128)
    Z_loop = Z_loop / torch.linalg.norm(Z_loop, dim=-1, keepdim=True)

    angles = torch.rand(8, 1, dtype=torch.float64) * 2.0 * torch.pi
    Z_gauged = Z_loop * torch.exp(1j * angles)

    for use_indef in (True, False):
        res_base = compute_berry_holonomy(Z_loop, use_indefinite_metric=use_indef)
        res_gauged = compute_berry_holonomy(Z_gauged, use_indefinite_metric=use_indef)
        assert bool(res_base["valid_mask"]) is True
        assert torch.allclose(res_base["berry_flux"], res_gauged["berry_flux"], atol=1e-12)


def test_berry_holonomy_near_orthogonal_guard():
    """Verifies degenerate/orthogonal adjacent states are cleanly flagged without NaN propagation."""
    Z_deg = torch.zeros(3, 5, dtype=torch.complex128)
    Z_deg[0, 0] = 1.0  # Null vector under Sigma_5: <e_0, Sigma_5 e_0> = 0
    Z_deg[1, 0] = 1.0
    Z_deg[2, 4] = 1.0

    res = compute_berry_holonomy(Z_deg, use_indefinite_metric=True)
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

    # Exact average of the last 5 states
    last_n = torch.stack(seq[-capacity:], dim=0)  # (N, F, 5)
    projs = torch.einsum("nfi,nfj->nfij", last_n, last_n.conj())
    rho_exact = projs.mean(dim=0)

    assert torch.allclose(rho_stream, rho_exact, atol=1e-12)
    # Trace of each fiber density matrix must be 1.0
    traces = torch.einsum("fii->f", rho_stream).real
    assert torch.allclose(traces, torch.ones(num_fibers, dtype=torch.float64), atol=1e-12)


def test_non_divisible_dimension_fiber_padding():
    """Verifies dimensions not divisible by 3 (e.g., d=64, d=65) retain all features via padding."""
    for d in (63, 64, 65):
        x = torch.ones(2, 10, d, dtype=torch.float64)
        fibers, pad_len = extract_3d_fibers_padded(x)
        assert fibers.shape[-1] == 3
        assert fibers.shape[-2] * 3 == d + pad_len
        # Sum of elements is preserved without truncation
        assert torch.isclose(fibers.sum(), x.sum())

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    for name, fn in sorted( list(globals().items()) ):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")

