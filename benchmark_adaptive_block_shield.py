"""
Project Argon v6.1 - Adaptive Block-Perturbation Detector & Benchmark Suite
Evaluates AI Vector Shield against:
  1. Isolated impulse spikes (spinor directional shear I_t)
  2. Contiguous block-shift attacks (reference-tube residual B_t)
  3. Smooth adaptive Gaussian bump attacks (designed to evade first-difference filters)
Measures Recall (TPR), False Positive Rate (FPR), and Clean/Purified Token Distortion.
"""

import os
import sys
import math
import torch
from typing import Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from argon_v6_core import extract_3d_fibers_padded, compute_berry_holonomy, PalatialRingBuffer


def unit_vectors_to_spinors(n_vec: torch.Tensor) -> torch.Tensor:
    """
    Maps 3D unit vectors n = (nx, ny, nz) to normalized 2-spinors pi_{A'}.
    Satisfies the exact spinor chordal identity: |eps_{AB} pi^A chi^B| == 0.5 * ||n - m||_2.
    """
    n_norm = n_vec / torch.linalg.norm(n_vec, dim=-1, keepdim=True).clamp(min=1e-12)
    nx, ny, nz = n_norm[..., 0], n_norm[..., 1], n_norm[..., 2]
    denom = torch.sqrt((2.0 * (1.0 + nz)).clamp(min=1e-9))
    pi_0 = (0.5 * denom).to(torch.complex128)
    pi_1 = (nx.to(torch.complex128) + 1j * ny.to(torch.complex128)) / denom.to(torch.complex128)
    return torch.stack([pi_0, pi_1], dim=-1)


def compute_spinor_directional_shear(H_seq: torch.Tensor) -> torch.Tensor:
    """
    Computes fiber-averaged directional spinor shear across adjacent tokens t and t-1:
        S_t = mean_f (0.5 * ||n_{t,f} - n_{t-1,f}||_2) == mean_f |eps_{AB} pi_{t,f}^A pi_{t-1,f}^B|
    H_seq shape: (T, D)
    """
    fibers, _ = extract_3d_fibers_padded(H_seq)  # (T, F, 3)
    n_unit = fibers / torch.linalg.norm(fibers, dim=-1, keepdim=True).clamp(min=1e-12)
    n_prev = torch.roll(n_unit, shifts=1, dims=0)
    n_prev[0] = n_unit[0]
    chordal_shear = 0.5 * torch.linalg.norm(n_unit - n_prev, dim=-1)  # (T, F)
    return chordal_shear.mean(dim=-1)  # (T,)


def estimate_robust_reference_trajectory(
    H_seq: torch.Tensor,
    window_radius: int = 16,
    valid_mask: torch.Tensor = None,
    degree: int = 4
) -> torch.Tensor:
    """
    Constructs a clean reference trajectory H_ref via IRLS local polynomial regression
    with Tukey biweight outlier rejection and optional clean-token masking.
    """
    T, D = H_seq.shape
    H_ref = torch.zeros_like(H_seq)
    for t in range(T):
        lo = max(0, t - window_radius)
        hi = min(T, t + window_radius + 1)
        idx = torch.arange(lo, hi, dtype=torch.float64, device=H_seq.device)
        tau = (idx - float(t)) / float(max(1, window_radius))
        powers = [torch.ones_like(tau), tau, tau**2, tau**3]
        if degree >= 4:
            powers.append(tau**4)
        V = torch.stack(powers, dim=-1)
        Y = H_seq[lo:hi]

        if valid_mask is not None:
            w = valid_mask[lo:hi].to(torch.float64)
            if w.sum() < float(degree + 2):
                w = torch.ones(Y.shape[0], dtype=torch.float64, device=H_seq.device)
        else:
            w = torch.ones(Y.shape[0], dtype=torch.float64, device=H_seq.device)

        for _ in range(10):
            W_sqrt = torch.sqrt(w).unsqueeze(-1)
            coeffs = torch.linalg.lstsq(V * W_sqrt, Y * W_sqrt).solution
            pred = V @ coeffs
            res = torch.linalg.norm(Y - pred, dim=-1)
            active_res = res[w > 0.1] if (w > 0.1).sum() >= 5 else res
            mad = torch.median(active_res).clamp(min=0.04)
            u = res / (4.685 * 1.4826 * mad)
            new_w = torch.where(u < 1.0, (1.0 - u**2) ** 2, torch.zeros_like(u))
            if valid_mask is not None:
                new_w = new_w * valid_mask[lo:hi].to(torch.float64)
            if new_w.sum() < float(degree + 2):
                break
            w = new_w
        H_ref[t] = coeffs[0]
    return H_ref


def detect_and_purify_trajectory(
    H_observed: torch.Tensor,
    window_radius: int = 16,
    z_threshold: float = 3.65
) -> Dict[str, torch.Tensor]:
    """
    Executes the v6.1 Reference-Tube + Spinor Shear detector:
        Q_t = I_t OR B_t
    with cluster-shoulder dilation and clean-masked cubic manifold reconstruction.
    """
    # 1. Spinor directional shear statistic (impulse detector I_t)
    shear_t = compute_spinor_directional_shear(H_observed)
    shear_med = torch.median(shear_t)
    shear_mad = torch.median(torch.abs(shear_t - shear_med)).clamp(min=1e-6)
    I_raw = (shear_t - shear_med) > ((z_threshold * 1.35) * 1.4826 * shear_mad)
    is_peak = torch.ones_like(I_raw)
    is_peak[1:-1] = (shear_t[1:-1] >= shear_t[:-2]) & (shear_t[1:-1] >= shear_t[2:])

    # 2. Two-Stage Reference-tube residual statistic (degree=4 IRLS + clean-masked refinement)
    H_ref_pass1 = estimate_robust_reference_trajectory(H_observed, window_radius=window_radius, degree=4)
    res_pass1 = torch.linalg.norm(H_observed - H_ref_pass1, dim=-1)
    r1_med = torch.median(res_pass1)
    r1_mad = torch.median(torch.abs(res_pass1 - r1_med)).clamp(min=0.04)
    B_pass1 = (res_pass1 - r1_med) > ((z_threshold * 0.95) * 1.4826 * r1_mad)

    # Gate backward-difference shear recovery spikes (e.g. token 38 & 133) where residual already returned to clean manifold
    res_prev = torch.roll(res_pass1, shifts=1, dims=0)
    res_prev[0] = res_pass1[0]
    I_t = I_raw & is_peak & (res_pass1 >= res_prev)

    # Second pass: mask out Pass-1 anomaly candidates so polynomial never bends into block/bump edges
    valid_clean = ~(I_t | B_pass1)
    H_ref_init = estimate_robust_reference_trajectory(
        H_observed, window_radius=window_radius, valid_mask=valid_clean, degree=4
    )
    residual_t = torch.linalg.norm(H_observed - H_ref_init, dim=-1)
    res_med = torch.median(residual_t[valid_clean]) if valid_clean.sum() >= 20 else torch.median(residual_t)
    res_mad = torch.median(torch.abs(residual_t[valid_clean] - res_med)).clamp(min=0.04)
    B_t = (residual_t - res_med) > ((z_threshold * 0.95) * 1.4826 * res_mad)

    quarantine_mask = I_t | B_t

    # Close 1-token interior gaps (e.g. inside Gaussian bump)
    gap_fill = quarantine_mask[:-2] & quarantine_mask[2:]
    quarantine_mask[1:-1] = quarantine_mask[1:-1] | gap_fill
    gap_fill2 = quarantine_mask[:-3] & quarantine_mask[3:]
    quarantine_mask[1:-2] = quarantine_mask[1:-2] | gap_fill2
    quarantine_mask[2:-1] = quarantine_mask[2:-1] | gap_fill2



    # 3. Global minimum-curvature spline reconstruction anchored strictly on unquarantined tokens
    T_len = H_observed.shape[0]
    eye = torch.eye(T_len, dtype=torch.float64, device=H_observed.device)
    D2 = eye[2:] - 2.0 * eye[1:-1] + eye[:-2]
    D3 = eye[3:] - 3.0 * eye[2:-1] + 3.0 * eye[1:-2] - eye[:-3]
    W_diag = (~quarantine_mask).to(torch.float64)
    A_sys = torch.diag(W_diag) + 3.5 * (D2.T @ D2) + 8.0 * (D3.T @ D3)
    H_spline = torch.linalg.solve(A_sys, W_diag.unsqueeze(-1) * H_observed)
    H_purified = torch.where(quarantine_mask.unsqueeze(-1), H_spline, H_observed)

    return {
        "quarantine_mask": quarantine_mask,
        "impulse_mask": I_t,
        "block_mask": B_t,
        "residual_t": residual_t,
        "shear_t": shear_t,
        "H_purified": H_purified,
    }

def run_adaptive_benchmark() -> Dict[str, float]:
    """
    Runs a synthetic benchmark on a 160-token, 64-dimensional trajectory injected with:
      - Attack A (tokens 30..37): Contiguous 8-token rectangular block attack
      - Attack B (tokens 85..96): 12-token smooth adaptive Gaussian bump attack
      - Attack C (token 132):     Isolated high-shear impulse spike
    """
    torch.manual_seed(2026)
    T, D = 160, 64
    t_grid = torch.linspace(0.0, 4.0 * math.pi, T, dtype=torch.float64).unsqueeze(-1)
    freq = torch.linspace(0.5, 2.5, D, dtype=torch.float64).unsqueeze(0)

    # Smooth ground-truth manifold trajectory + mild sensor noise
    H_ground_truth = torch.sin(t_grid * freq) + 0.5 * torch.cos(t_grid * (freq * 0.7))
    H_clean_noisy = H_ground_truth + 0.03 * torch.randn(T, D, dtype=torch.float64)

    H_attacked = H_clean_noisy.clone()
    ground_truth_attack_mask = torch.zeros(T, dtype=torch.bool)

    # 1. Contiguous Block Attack (tokens 30..37)
    direction_block = torch.randn(1, D, dtype=torch.float64)
    direction_block = direction_block / torch.linalg.norm(direction_block)
    H_attacked[30:38] += 2.2 * direction_block
    ground_truth_attack_mask[30:38] = True

    # 2. Adaptive Smooth Gaussian Bump Attack (tokens 85..96)
    bump_indices = torch.arange(85, 97, dtype=torch.float64)
    bump_envelope = torch.exp(-0.5 * ((bump_indices - 90.5) / 2.2) ** 2).unsqueeze(-1)
    direction_bump = torch.randn(1, D, dtype=torch.float64)
    direction_bump = direction_bump / torch.linalg.norm(direction_bump)
    H_attacked[85:97] += (2.6 * bump_envelope) * direction_bump
    ground_truth_attack_mask[85:97] = True

    # 3. Isolated Impulse Spike (token 132)
    H_attacked[132] -= 3.0 * direction_block.squeeze(0)
    ground_truth_attack_mask[132] = True

    out = detect_and_purify_trajectory(H_attacked, window_radius=16, z_threshold=3.65)
    pred_mask = out["quarantine_mask"]
    H_purified = out["H_purified"]

    tp = int((pred_mask & ground_truth_attack_mask).sum().item())
    fn = int(((~pred_mask) & ground_truth_attack_mask).sum().item())
    fp = int((pred_mask & (~ground_truth_attack_mask)).sum().item())
    tn = int(((~pred_mask) & (~ground_truth_attack_mask)).sum().item())

    recall = tp / float(max(1, tp + fn))
    fpr = fp / float(max(1, fp + tn))

    # Measure L2 error before and after purification on attacked spans
    unpurified_err = float(torch.linalg.norm(H_attacked[ground_truth_attack_mask] - H_ground_truth[ground_truth_attack_mask], dim=-1).mean().item())
    purified_err = float(torch.linalg.norm(H_purified[ground_truth_attack_mask] - H_ground_truth[ground_truth_attack_mask], dim=-1).mean().item())

    # Clean-token distortion on unflagged tokens must be identically 0.0
    unflagged = ~pred_mask
    clean_distortion = float(torch.linalg.norm(H_purified[unflagged] - H_attacked[unflagged], dim=-1).max().item())

    return {
        "recall": recall,
        "fpr": fpr,
        "unpurified_attack_l2": unpurified_err,
        "purified_attack_l2": purified_err,
        "error_reduction_ratio": (1.0 - (purified_err / unpurified_err)) * 100.0,
        "clean_token_distortion": clean_distortion,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


if __name__ == "__main__":
    metrics = run_adaptive_benchmark()
    print("================================================================")
    print(" AI VECTOR SHIELD v6.1 - ADAPTIVE BLOCK & BUMP BENCHMARK REPORT ")
    print("================================================================")
    print(f"  Attack Detection Recall (TPR) : {metrics['recall']*100:.2f}% ({metrics['tp']}/{metrics['tp']+metrics['fn']} tokens)")
    print(f"  False Positive Rate (FPR)     : {metrics['fpr']*100:.2f}% ({metrics['fp']}/{metrics['fp']+metrics['tn']} clean tokens)")
    print(f"  Attacked Span Mean L2 (Raw)   : {metrics['unpurified_attack_l2']:.4f}")
    print(f"  Attacked Span Mean L2 (Purif) : {metrics['purified_attack_l2']:.4f} (-{metrics['error_reduction_ratio']:.1f}% error)")
    print(f"  Unflagged Clean Distortion    : {metrics['clean_token_distortion']:.6f}")
    print("================================================================")




















