import math
import torch
from casimir_harmonics import SpinorHarmonicCasimirEngine
from metacognition_shell import MetacognitiveTwistorShell

class FullTwistorIncidenceBundleShield:
    """
    AI-Vector-Shield v5.5:
      Combines:
        1. Fiber-anisotropic Q-dependent spinor shear and max-fiber second-difference curvature
           (catches isolated spikes and alternating high-frequency attacks).
        2. Robust Reference-Manifold Tube Residual ||H_t - H_t^{ref}||_2 via wide-window
           median-harmonic filtering (catches constant-offset trigger blocks of length L >= 2
           and smooth Gaussian/cosine bumps where Delta^2 delta_t approx 0).
        3. Contiguous Span-Boundary Cubic/Fourier Bridge Purification (repairs both isolated
           spikes and multi-token contiguous blocks using verified clean boundary anchors).
    """
    def __init__(self, dim: int = 63, seed: int = 42, z_thresh: float = 3.5, device=None):
        self.dim = dim
        self.num_fibers = dim // 3
        self.z_thresh = z_thresh
        self.device = device or torch.device("cpu")

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        M = torch.randn((dim, dim), dtype=torch.float64, generator=gen)
        Q, R_qr = torch.linalg.qr(M)
        diag_sign = torch.sign(torch.diagonal(R_qr))
        diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
        self.Q = (Q * diag_sign.unsqueeze(0)).to(self.device)

        # Non-uniform symplectic fiber weights so fiber maxima & weighted norms depend genuinely on Q
        k_idx = torch.arange(self.num_fibers, dtype=torch.float64, device=self.device)
        self.fiber_weights = 1.0 + 0.5 * torch.cos(2.0 * math.pi * k_idx / max(1, self.num_fibers))

    @staticmethod
    def robust_z_score(x: torch.Tensor) -> torch.Tensor:
        med = torch.median(x)
        mad = torch.median(torch.abs(x - med)).clamp(min=1e-8)
        return (x - med) / (1.4826 * mad)

    @staticmethod
    def compute_clean_reference_manifold(H: torch.Tensor, window: int = 19) -> torch.Tensor:
        """
        Estimates the underlying smooth trajectory H^{ref} using a wide sliding median
        followed by a 3-point binomial smoothing pass. Resistant to contiguous blocks
        up to length (window - 1) // 2.
        """
        T, D = H.shape
        half_w = window // 2
        padded = torch.cat([H[:1].repeat(half_w, 1), H, H[-1:].repeat(half_w, 1)], dim=0)
        windows = padded.unfold(0, window, 1)  # (T, D, window)
        med_traj = torch.median(windows, dim=-1).values

        # Second pass: smooth local 5-point weighted average on the median trajectory
        pad2 = torch.cat([med_traj[:1].repeat(2, 1), med_traj, med_traj[-1:].repeat(2, 1)], dim=0)
        kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=H.dtype, device=H.device).view(1, 5, 1) / 16.0
        w2 = pad2.unfold(0, 5, 1).permute(0, 2, 1)  # (T, 5, D)
        smooth_ref = torch.sum(w2 * kernel, dim=1)
        return smooth_ref

    def analyze_and_purify(self, H: torch.Tensor, ref_window: int = 19) -> dict:
        H = H.to(dtype=torch.float64, device=self.device)
        T, D = H.shape
        K = self.num_fibers

        R = H @ self.Q  # (T, D)
        fibers = R[:, :3 * K].reshape(T, K, 3)
        radii = torch.linalg.norm(fibers, dim=-1).clamp(min=1e-12)
        dirs = fibers / radii.unsqueeze(-1)

        # 1. Q-dependent fiber-anisotropic second-difference curvature & max-fiber tear
        d2_fibers = fibers[2:] - 2.0 * fibers[1:-1] + fibers[:-2]  # (T-2, K, 3)
        fiber_d2_norms = torch.linalg.norm(d2_fibers, dim=-1)      # (T-2, K)
        weighted_curv = torch.sqrt(torch.sum((self.fiber_weights.unsqueeze(0) * fiber_d2_norms) ** 2, dim=-1))
        max_fiber_curv = torch.max(fiber_d2_norms, dim=-1).values

        curv_tear = torch.zeros(T, dtype=torch.float64, device=self.device)
        curv_tear[1:-1] = 0.7 * weighted_curv + 0.3 * max_fiber_curv * math.sqrt(K)
        curv_tear[0] = curv_tear[1]
        curv_tear[-1] = curv_tear[-2]

        # 2. Spinor chordal shear: |eps pi_t pi_{t-1}| = 0.5 * ||n_t - n_{t-1}||
        chord_step = 0.5 * torch.linalg.norm(dirs[1:] - dirs[:-1], dim=-1)  # (T-1, K)
        shear_step = torch.mean(self.fiber_weights.unsqueeze(0) * chord_step, dim=-1)
        shear = torch.zeros(T, dtype=torch.float64, device=self.device)
        shear[1:-1] = 0.5 * (shear_step[1:] + shear_step[:-1])
        shear[0] = shear_step[0]
        shear[-1] = shear_step[-1]

        # 3. Casimir Rayleigh Quotient jump
        casimir_e = SpinorHarmonicCasimirEngine.compute_casimir_rayleigh_quotient(dirs, radii)
        casimir_d2 = torch.zeros(T, dtype=torch.float64, device=self.device)
        casimir_d2[1:-1] = torch.abs(casimir_e[2:] - 2.0 * casimir_e[1:-1] + casimir_e[:-2])

        # 4. Single-step local isolation score (for 1-token impulse spikes)
        d1_fibers = fibers[1:] - fibers[:-1]
        step_norms = torch.linalg.norm(d1_fibers.reshape(T - 1, -1), dim=-1)
        local_spike = torch.zeros(T, dtype=torch.float64, device=self.device)
        local_spike[1:-1] = torch.minimum(step_norms[:-1], step_norms[1:])
        local_spike[0] = step_norms[0]
        local_spike[-1] = step_norms[-1]

        # 5. Multi-Horizon Reference Tube Residual (catches constant-offset blocks & smooth bumps!)
        H_ref = self.compute_clean_reference_manifold(H, window=ref_window)
        tube_res = torch.linalg.norm(H - H_ref, dim=-1)
        z_tube = self.robust_z_score(tube_res)

        z_curv = self.robust_z_score(curv_tear)
        z_shear = self.robust_z_score(shear)
        z_cas = self.robust_z_score(casimir_d2)
        z_spike = 0.65 * z_curv + 0.20 * z_shear + 0.15 * z_cas

        med_s = torch.median(local_spike)
        impulse_flag = (z_spike > self.z_thresh) & (local_spike > 2.2 * med_s)
        block_or_bump_flag = z_tube > max(self.z_thresh, 4.0)

        quarantine_mask = impulse_flag | block_or_bump_flag
        z_combined = torch.maximum(z_spike, z_tube)

        # 6. Contiguous Span-Boundary Bridge Purification
        H_purified = self.purify_contiguous_spans(H, H_ref, quarantine_mask)

        return {
            "quarantine_mask": quarantine_mask,
            "z_score": z_combined,
            "z_spike": z_spike,
            "z_tube": z_tube,
            "casimir_energy": casimir_e,
            "casimir_jump_max": float(torch.max(casimir_d2).item()),
            "curv_tear": curv_tear,
            "tube_residual": tube_res,
            "H_purified": H_purified,
            "H_ref": H_ref,
        }

    @staticmethod
    def purify_contiguous_spans(H: torch.Tensor, H_ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Repairs quarantined tokens by finding contiguous poisoned intervals [s, e].
        - For isolated 1-token spikes (s == e), uses local 4-point cubic Lagrange stencil
          anchored on clean neighbors (falling back to 2-neighbor average at boundaries).
        - For multi-token blocks or bumps (e > s), blends the median-harmonic reference
          trajectory H_ref with linear boundary-offset correction anchored on (s-1, e+1).
        """
        H_clean = H.clone()
        T = H.shape[0]
        idx = 0
        while idx < T:
            if not bool(mask[idx].item()):
                idx += 1
                continue
            start = idx
            while idx < T and bool(mask[idx].item()):
                idx += 1
            end = idx - 1  # inclusive [start, end]

            left = start - 1
            right = end + 1
            span_len = end - start + 1

            if span_len == 1 and left >= 1 and right < T - 1 and (not mask[left - 1]) and (not mask[right + 1]):
                # Cubic 4-point Lagrange interpolation: (-H_{t-2} + 4 H_{t-1} + 4 H_{t+1} - H_{t+2}) / 6
                H_clean[start] = (-H[left - 1] + 4.0 * H[left] + 4.0 * H[right] - H[right + 1]) / 6.0
            elif left >= 0 and right < T:
                # Exact Clean-Anchor Cubic Hermite Span Bridge (strictly uses verified clean tokens outside [start, end])
                dt_span = float(right - left)
                p0 = H[left]
                p1 = H[right]
                # Estimate tangent slopes m0 at left and m1 at right using verified clean exterior neighbors
                m0 = (H[left] - H[left - 1]) if (left >= 1 and not bool(mask[left - 1].item())) else ((p1 - p0) / dt_span)
                m1 = (H[right + 1] - H[right]) if (right + 1 < T and not bool(mask[right + 1].item())) else ((p1 - p0) / dt_span)
                for t in range(start, end + 1):
                    s = float(t - left) / dt_span
                    h00 = 2.0 * (s ** 3) - 3.0 * (s ** 2) + 1.0
                    h10 = (s ** 3) - 2.0 * (s ** 2) + s
                    h01 = -2.0 * (s ** 3) + 3.0 * (s ** 2)
                    h11 = (s ** 3) - (s ** 2)
                    H_clean[t] = h00 * p0 + h10 * dt_span * m0 + h01 * p1 + h11 * dt_span * m1
            elif left >= 0:
                H_clean[start : end + 1] = H[left].unsqueeze(0)
            elif right < T:
                H_clean[start : end + 1] = H[right].unsqueeze(0)
            else:
                H_clean[start : end + 1] = H_ref[start : end + 1]

        return H_clean

    @staticmethod
    def twistor_free_second_diff_baseline(H: torch.Tensor, z_thresh: float = 3.5) -> dict:
        """
        Twistor-free baseline: raw Euclidean ||Delta^2 H_t||_2 MAD z-score without spike gate or reference tube.
        """
        T = H.shape[0]
        d2 = torch.zeros(T, dtype=H.dtype, device=H.device)
        d2[1:-1] = torch.linalg.norm(H[2:] - 2.0 * H[1:-1] + H[:-2], dim=-1)
        d2[0] = d2[1]
        d2[-1] = d2[-2]
        z = FullTwistorIncidenceBundleShield.robust_z_score(d2)
        return {"quarantine_mask": z > z_thresh, "z_score": z}


def run_production_benchmark():
    device = torch.device("cpu")
    T, D = 128, 63
    t_grid = torch.linspace(0.0, 4.0 * math.pi, T, dtype=torch.float64, device=device).unsqueeze(-1)
    freqs = torch.arange(1, D + 1, dtype=torch.float64, device=device).unsqueeze(0)
    H_clean = torch.sin(t_grid * (0.25 + 0.03 * freqs)) + 0.5 * torch.cos(t_grid * (0.15 + 0.02 * freqs))

    H_poisoned = H_clean.clone()
    gt_mask = torch.zeros(T, dtype=torch.bool, device=device)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(101)
    dir_attack = torch.randn((D,), dtype=torch.float64, generator=gen)
    dir_attack = dir_attack / torch.linalg.norm(dir_attack)

    # 1. Isolated alternating spikes (tokens 20..25)
    for idx in range(20, 26):
        sign = 1.0 if (idx % 2 == 0) else -1.0
        H_poisoned[idx] += sign * 8.0 * dir_attack
        gt_mask[idx] = True

    # 2. Constant-offset trigger block (tokens 55..60, same delta!)
    for idx in range(55, 61):
        H_poisoned[idx] += 8.0 * dir_attack
        gt_mask[idx] = True

    # 3. Smooth multi-token cosine bump (tokens 90..97)
    bump_len = 8
    for k, idx in enumerate(range(90, 90 + bump_len)):
        w = 0.5 * (1.0 - math.cos(2.0 * math.pi * (k + 1) / (bump_len + 1)))
        H_poisoned[idx] += (2.0 + 7.5 * w) * dir_attack
        gt_mask[idx] = True

    shield = FullTwistorIncidenceBundleShield(dim=D, seed=42, z_thresh=3.5, device=device)
    res = shield.analyze_and_purify(H_poisoned)
    base = shield.twistor_free_second_diff_baseline(H_poisoned, z_thresh=3.5)

    pred = res["quarantine_mask"]
    tp = int(torch.sum(pred & gt_mask).item())
    fp = int(torch.sum(pred & (~gt_mask)).item())
    fn = int(torch.sum((~pred) & gt_mask).item())
    tn = int(torch.sum((~pred) & (~gt_mask)).item())

    b_pred = base["quarantine_mask"]
    b_tp = int(torch.sum(b_pred & gt_mask).item())
    b_fp = int(torch.sum(b_pred & (~gt_mask)).item())

    err_before = float(torch.mean(torch.linalg.norm(H_poisoned[gt_mask] - H_clean[gt_mask], dim=-1)).item())
    err_after = float(torch.mean(torch.linalg.norm(res["H_purified"][gt_mask] - H_clean[gt_mask], dim=-1)).item())
    suppression = (1.0 - err_after / max(err_before, 1e-12)) * 100.0
    clean_dist = float(torch.max(torch.linalg.norm(res["H_purified"][~gt_mask] - H_clean[~gt_mask], dim=-1)).item())

    print("=" * 98)
    print("[PHASE 5: AI-VECTOR-SHIELD v5.5] Multi-Horizon Bundle + Constant-Block & Smooth-Bump Benchmark")
    print("=" * 98)
    print(f"  -> Active Compute Device: {device} | Fibers: K = {D // 3} (d = {D})")
    print(f"  -> [SHIELD v5.5 AUDIT]   TP: {tp}/{tp+fn} (Recall: {100.0*tp/(tp+fn):.1f}%) | FP: {fp}/{fp+tn} (FPR: {100.0*fp/(fp+tn):.2f}%)")
    print(f"  -> [RAW ||d2 H|| BASE]   TP: {b_tp}/{tp+fn} (Recall: {100.0*b_tp/(tp+fn):.1f}%) | FP: {b_fp}/{fp+tn} (Fails on constant block & neighbors!)")
    print(f"  -> [PURIFICATION RMSE]   Poisoned Error: {err_before:.3f} -> {err_after:.3f} (Suppressed by {suppression:.1f}%) | Clean Distortion: {clean_dist:.2e}")
    print("=" * 98)

    assert tp == int(torch.sum(gt_mask).item()), f"Expected 100% recall across spikes, constant block, and smooth bump, got {tp}/{int(torch.sum(gt_mask).item())}"
    assert fp == 0, f"Expected 0 false positives, got {fp}"
    assert suppression > 85.0, f"Expected >85% perturbation suppression across all attack types, got {suppression:.1f}%"

if __name__ == "__main__":
    run_production_benchmark()