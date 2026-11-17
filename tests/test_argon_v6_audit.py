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
    compute_sigma5_freedom_matrix,
    build_five_quantum_twistor_state,
    apply_googly_chiral_patch_coupling,
    MetacognitiveShieldProtector,
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


from benchmark_adaptive_block_shield import run_adaptive_benchmark


def test_adaptive_block_and_bump_detection():
    torch.manual_seed(7)
    """Verifies >= 90% recall, <= 5% FPR, > 65% L2 error reduction, and 0.0 clean distortion."""
    m = run_adaptive_benchmark()
    assert m["recall"] >= 0.90
    assert m["fpr"] <= 0.05
    assert m["purified_attack_l2"] < 0.35 * m["unpurified_attack_l2"]
    assert m["clean_token_distortion"] == 0.0


def test_penrose_five_quantum_rules_and_25_sigma5_freedoms():
    """
    Verifies the 5-Quantum-Number Dual-Spinor (omega^A, pi_{A'}) & 25-Element SIGMA_5 Postulate
    and unitary Googly chiral patch coupling inside ai-vector-shield.
    """
    assert SIGMA_5.numel() == 25
    n = torch.tensor([1.0, 2.0], dtype=torch.float64)
    l = torch.tensor([0.0, 1.0], dtype=torch.float64)
    m = torch.tensor([0.0, 0.0], dtype=torch.float64)
    s_omega = torch.tensor([0.5, 0.5], dtype=torch.float64)
    s_pi = torch.tensor([0.5, -0.5], dtype=torch.float64)

    res = build_five_quantum_twistor_state(n, l, m, s_omega, s_pi)
    assert res["freedom_matrix_25"].shape == (2, 5, 5)
    assert res["freedom_matrix_25"][0].numel() == 25
    tr_25 = torch.einsum("...ii->...", res["freedom_matrix_25"]).real
    assert torch.allclose(tr_25, res["sigma5_norm_complete"], atol=1e-12)
    assert torch.all(torch.abs(res["core_electron_spinor_coupling"]) > 1.5)

    g_out = apply_googly_chiral_patch_coupling(
        res["Z5_complete"],
        res["Z5_complete"].conj(),
        epsilon_googly=0.05
    )
    I5 = torch.eye(5, dtype=torch.complex128).unsqueeze(0).expand(2, 5, 5)
    U = g_out["U_googly"]
    assert torch.allclose(torch.matmul(U, U.conj().transpose(-1, -2)), I5, atol=1e-12)


def test_metacognitive_shield_closed_loop_protection():
    """
    Verifies the closed-loop MetacognitiveShieldProtector inside ai-vector-shield:
      1. Projects 25-element SIGMA_5 states into a continuous 2D manifold field.
      2. Detects both spatial roughness energy and Krein-space Berry phase shear.
      3. Outputs a unit-normalized rectified trajectory with valid protection gating.
    """
    torch.manual_seed(42)
    Z_seq = torch.randn(12, 5, dtype=torch.complex128)
    Z_seq = Z_seq / torch.linalg.norm(Z_seq, dim=-1, keepdim=True)

    protector = MetacognitiveShieldProtector(resolution=20, roughness_gain=0.35, holonomy_gain=0.25)
    out = protector.evaluate_metacognitive_shield(Z_seq)

    assert out["freedom_matrix_25"].shape == (12, 5, 5)
    assert out["freedom_matrix_25"][0].numel() == 25
    assert out["rendered_manifold"].shape == (20, 20)
    assert out["roughness_energy"] > 0.0
    assert 0.0 < float(out["protection_gate"].item()) <= 1.0
    assert torch.allclose(
        torch.linalg.norm(out["Z_rectified"], dim=-1),
        torch.ones(12, dtype=torch.float64),
        atol=1e-12
    )


def test_projective_cp3_mandelbrot_shield_projection():
    """Verifies native trigonometry-free CP^3 Mandelbrot projection + closed-loop shield gating."""
    torch.manual_seed(23)
    enc = ProjectiveCP3MandelbrotEncoder(iterations=16)
    protector = MetacognitiveShieldProtector(resolution=16)
    tokens = torch.randn(10, 24, dtype=torch.float64)
    f_out = enc.iterate_cp3_mandelbrot_orbit(tokens)
    s_out = protector.evaluate_metacognitive_shield(f_out["Z5_cp3_mandelbrot"])
    assert s_out["freedom_matrix_25"].shape == (10, 5, 5)
    assert 0.0 < float(s_out["protection_gate"].item()) <= 1.0


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")






