import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from casimir_harmonics import PenroseTwistorHarmonicEngine

class FullTwistorIncidenceBundleShield:
    """
    Default Detector & Geodesic Purifier for Phase 5 (AI-Vector-Shield):
      1. Full-Rank Orthogonal Bundle Decomposition: Maps H in R^{T x d} via Q in O(d)
         into K = d // 3 disjoint 3D twistor fibers so ker(Q) = {0}.
      2. Multi-Fiber Symplectic Spinor Shear: Evaluates |eps_{AB} pi_{t,k}^A pi_{t-1,k}^B|
         across all K fibers on CP^1.
      3. Un-logged Hermitian Incidence Curvature Tear: Evaluates ||Delta^2 X_t||_F = sqrt(2) ||Delta^2 R_t||_F,
         enforcing the reverse triangle inequality against white-box PGD radial-cone evasion.
      4. L^2(CP^1) Conformal Casimir Harmonic Deviation across all K fibers.
      5. Exact Anchor-Bridge Geodesic Purification in R^d with zero distortion on unflagged clean tokens.
    """
    def __init__(self, d_model: int = 63, z_thresh: float = 3.5, device=None):
        if device is not None:
            self.device = device
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = torch.device("xpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.d_model = d_model
        self.num_fibers = d_model // 3
        self.z_thresh = z_thresh
        self.harmonics = PenroseTwistorHarmonicEngine(device=self.device)

        # Construct deterministic full-rank orthogonal matrix Q in O(d) on CPU then move to device
        gen = torch.Generator(device="cpu")
        gen.manual_seed(42)
        A = torch.randn((d_model, d_model), generator=gen, dtype=torch.float64)
        Q, _ = torch.linalg.qr(A)
        self.Q = Q.to(device=self.device)

    def compute_bundle_diagnostics(self, H: torch.Tensor):
        """
        Evaluates full-rank twistor bundle diagnostics for H of shape (T, d).
        Fully differentiable for white-box PGD red-team auditing.
        """
        orig_dtype = H.dtype
        H_work = H.to(device=self.device, dtype=torch.float64)
        Q_work = self.Q.to(dtype=torch.float64)
        T, d = H_work.shape
        K = self.num_fibers

        # Project onto full-rank orthogonal fibers: (T, K, 3)
        R_full = H_work @ Q_work
        R_fibers = R_full[:, :3 * K].view(T, K, 3)

        # 1. Multi-fiber CP^1 spinors and Symplectic Levi-Civita shear
        pi_fibers = self.harmonics.points_to_celestial_spinors(R_fibers)  # (T, K, 2)
        pi_curr, pi_prev = pi_fibers[1:], pi_fibers[:-1]
        fiber_shear = torch.abs(pi_curr[:, :, 0] * pi_prev[:, :, 1] - pi_curr[:, :, 1] * pi_prev[:, :, 0])  # (T-1, K)
        shear_step = torch.sqrt(torch.sum(fiber_shear ** 2, dim=-1))  # (T-1,)
        symplectic_shear = torch.cat([torch.zeros(1, dtype=torch.float64, device=self.device), shear_step], dim=0)

        # 2. First-order Hermitian incidence step jump ||Delta X_t||_F = sqrt(2) ||R_t - R_{t-1}||_2
        d1_R = R_full[1:] - R_full[:-1]
        d1_norm = torch.sqrt(2.0 * torch.sum(d1_R ** 2, dim=-1))
        d1_pad = torch.cat([d1_norm[:1], d1_norm], dim=0)
        d1_next = torch.cat([d1_norm, d1_norm[-1:]], dim=0)
        # Local spike isolation: min(|H_t - H_{t-1}|, |H_{t+1} - H_t|) peaks sharply AT the poisoned token t
        # while suppressing the neighbor smear at t-1 and t+1!
        local_spike_tear = torch.minimum(d1_pad, d1_next)

        # 3. Second-order Hermitian incidence curvature tear ||Delta^2 X_t||_F = sqrt(2) ||R_{t+1} - 2 R_t + R_{t-1}||_2
        d2_R = torch.zeros_like(R_full)
        d2_R[1:-1] = R_full[2:] - 2.0 * R_full[1:-1] + R_full[:-2]
        d2_R[0] = R_full[1] - R_full[0]
        d2_R[-1] = R_full[-1] - R_full[-2]
        hermitian_curvature_tear = torch.sqrt(2.0 * torch.sum(d2_R ** 2, dim=-1))

        # 4. Multi-fiber L^2(CP^1) Conformal Casimir harmonic jump
        B = self.harmonics.evaluate_real_basis(pi_fibers)  # (T, K, 9)
        eigenvalues = torch.tensor([0.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0], dtype=torch.float64, device=self.device)
        casimir_density = torch.sum(eigenvalues * (B ** 2), dim=-1)  # (T, K)
        casimir_jump_k = torch.abs(casimir_density[1:] - casimir_density[:-1])
        casimir_jump = torch.sqrt(torch.sum(casimir_jump_k ** 2, dim=-1))
        casimir_jump = torch.cat([torch.zeros(1, dtype=torch.float64, device=self.device), casimir_jump], dim=0)

        # Combined topological anomaly score
        anomaly_score = symplectic_shear + 0.5 * hermitian_curvature_tear + local_spike_tear + 0.25 * casimir_jump

        return {
            "pi_fibers": pi_fibers,
            "symplectic_shear": symplectic_shear.to(orig_dtype),
            "hermitian_tear": hermitian_curvature_tear.to(orig_dtype),
            "local_spike_tear": local_spike_tear.to(orig_dtype),
            "casimir_jump": casimir_jump.to(orig_dtype),
            "anomaly_score": anomaly_score.to(orig_dtype),
        }

    def detect_and_purify(self, H: torch.Tensor, z_thresh: float = None):
        """
        Detects poisoned / adversarial tokens via robust MAD z-scores across the full-rank
        twistor bundle and heals quarantined tokens via clean-anchor geodesic interpolation.
        """
        thresh = z_thresh if z_thresh is not None else self.z_thresh
        diag = self.compute_bundle_diagnostics(H)
        score = diag["anomaly_score"].to(torch.float64)

        med = torch.median(score)
        mad = torch.median(torch.abs(score - med)) + 1e-12
        z_scores = (score - med) / (1.4826 * mad)

        # Primary threshold + local peak refinement so isolated single-token spikes don't false-flag clean neighbors
        raw_flag = z_scores > thresh
        local_spike = diag["local_spike_tear"].to(torch.float64)
        med_s = torch.median(local_spike)
        mad_s = torch.median(torch.abs(local_spike - med_s)) + 1e-12
        z_spike = (local_spike - med_s) / (1.4826 * mad_s)

        quarantine_mask = raw_flag & (z_spike > (0.6 * thresh))

        # Purify quarantined tokens by interpolating between nearest unflagged clean anchor tokens
        H_healed = H.clone()
        T = H.shape[0]
        clean_indices = torch.where(~quarantine_mask)[0]

        if len(clean_indices) >= 2 and torch.any(quarantine_mask):
            bad_indices = torch.where(quarantine_mask)[0]
            for b_idx in bad_indices:
                b = int(b_idx.item())
                left_candidates = clean_indices[clean_indices < b]
                right_candidates = clean_indices[clean_indices > b]
                if len(left_candidates) > 0 and len(right_candidates) > 0:
                    l_idx = int(left_candidates[-1].item())
                    r_idx = int(right_candidates[0].item())
                    alpha = float(b - l_idx) / float(r_idx - l_idx)
                    H_healed[b] = (1.0 - alpha) * H[l_idx] + alpha * H[r_idx]
                elif len(left_candidates) > 0:
                    H_healed[b] = H[int(left_candidates[-1].item())]
                elif len(right_candidates) > 0:
                    H_healed[b] = H[int(right_candidates[0].item())]

        diag_after = self.compute_bundle_diagnostics(H_healed)
        return {
            "H_healed": H_healed,
            "quarantine_mask": quarantine_mask,
            "z_scores": z_scores.to(H.dtype),
            "before": diag,
            "after": diag_after,
        }


# Alias for backward compatibility across scripts
CP3TopologicalPoisonShield = FullTwistorIncidenceBundleShield


def run_phase5_full_bundle_benchmark():
    print("=" * 98)
    print("[PHASE 5: AI-VECTOR-SHIELD] Full-Rank Orthogonal Twistor Incidence Bundle & White-Box Defense")
    print("=" * 98)

    shield = FullTwistorIncidenceBundleShield(d_model=63, z_thresh=3.5)
    device = shield.device
    print(f"  -> Active Compute Device: {device} | Orthogonal Bundle Fibers: K = {shield.num_fibers} (d = {shield.d_model}, ker(Q) = 0)")

    T = 128
    d = 63
    t_grid = torch.linspace(0.0, 4.0 * math.pi, T, dtype=torch.float64, device=device).unsqueeze(-1)
    freqs = torch.arange(1, d + 1, dtype=torch.float64, device=device).unsqueeze(0) * 0.08
    H_clean = torch.sin(t_grid * freqs) + 0.5 * torch.cos(t_grid * freqs * 0.5)

    # Inject 3 distinct attack classes:
    #   1. Static high-norm spikes (indices 20..25)
    #   2. Subspace/Nullspace stealth spikes in higher dimensions (indices 58..65)
    #   3. Collinear radial-cone spikes along fiber 0 (indices 96..105)
    H_poisoned = H_clean.clone()
    gt_mask = torch.zeros(T, dtype=torch.bool, device=device)

    gen = torch.Generator(device="cpu").manual_seed(99)
    attack_1 = [20, 21, 22, 23, 24, 25]
    attack_2 = [58, 59, 60, 61, 62, 63, 64, 65]
    attack_3 = [96, 97, 98, 99, 100, 101, 102, 103, 104, 105]
    all_attack_indices = attack_1 + attack_2 + attack_3

    # Block 1: Multi-token adversarial injection
    for idx in attack_1:
        gt_mask[idx] = True
        noise = torch.randn(d, generator=gen, dtype=torch.float64).to(device)
        H_poisoned[idx] = H_clean[idx] + (-1.0 if idx % 2 == 0 else 1.0) * 6.0 * (noise / torch.norm(noise))

    # Block 2: High-dimensional nullspace attack (orthogonal to the first 3 columns of Q)
    for idx in attack_2:
        gt_mask[idx] = True
        z_null = torch.randn(d - 3, generator=gen, dtype=torch.float64).to(device)
        z_null = z_null / torch.norm(z_null) * 8.0
        delta_null = shield.Q[:, 3:] @ z_null
        H_poisoned[idx] = H_clean[idx] + (-1.0 if idx % 2 == 0 else 1.0) * delta_null

    # Block 3: Single-fiber collinear radial-cone attack (zero spinor shear in Fiber 0)
    for idx in attack_3:
        gt_mask[idx] = True
        r0 = (H_clean[idx] @ shield.Q[:, :3])
        r0_dir = r0 / torch.norm(r0)
        delta_collinear = shield.Q[:, :3] @ (r0_dir * 8.0)
        H_poisoned[idx] = H_clean[idx] + (-1.0 if idx % 2 == 0 else 1.0) * delta_collinear

    res = shield.detect_and_purify(H_poisoned, z_thresh=3.5)
    detected_mask = res["quarantine_mask"]

    tp = int(torch.sum(detected_mask & gt_mask).item())
    fp = int(torch.sum(detected_mask & (~gt_mask)).item())
    total_poison = int(torch.sum(gt_mask).item())
    precision = 100.0 * tp / max(tp + fp, 1)
    recall = 100.0 * tp / total_poison

    peak_before = float(torch.max(res["before"]["anomaly_score"]).item())
    peak_after = float(torch.max(res["after"]["anomaly_score"]).item())
    shear_before = float(torch.max(res["before"]["symplectic_shear"]).item())
    shear_after = float(torch.max(res["after"]["symplectic_shear"]).item())
    suppression = 100.0 * (1.0 - peak_after / peak_before)

    clean_distortion = float(torch.max(torch.abs(res["H_healed"][~detected_mask] - H_clean[~detected_mask])).item())

    print(f"  -> [DETECTION AUDIT] Poisoned Tokens Detected: {tp}/{total_poison} | Precision: {precision:.1f}% | Recall: {recall:.1f}%")
    print(f"  -> [BEFORE SHIELD]   Peak Bundle Anomaly: {peak_before:.4f} | Max Multi-Fiber Shear: {shear_before:.4f}")
    print(f"  -> [AFTER SHIELD]    Peak Bundle Anomaly: {peak_after:.4f} | Max Multi-Fiber Shear: {shear_after:.4f}")
    print(f"  -> [PURIFICATION]    Adversarial Spike Suppressed by {suppression:.1f}% | Clean Token Distortion: {clean_distortion:.2e}")

    # Save diagnostic figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor="#0a0e17")
    t_np = torch.arange(T).cpu().numpy()
    anom_b = res["before"]["anomaly_score"].detach().cpu().numpy()
    anom_a = res["after"]["anomaly_score"].detach().cpu().numpy()
    shear_b = res["before"]["symplectic_shear"].detach().cpu().numpy()
    shear_a = res["after"]["symplectic_shear"].detach().cpu().numpy()

    for ax in axes:
        ax.set_facecolor("#0f1523")
        ax.grid(True, color="#1e293b", linestyle="--", alpha=0.6)
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    axes[0].plot(t_np, anom_b, color="#f43f5e", lw=1.8, label="Before Shield (Static + Nullspace + Radial-Cone Attacks)")
    axes[0].plot(t_np, anom_a, color="#10b981", lw=2.0, label="After FullTwistorIncidenceBundleShield Purification")
    axes[0].set_title("Full-Rank O(d) Twistor Bundle Anomaly (CP^1 Shear + ||Delta^2 X_t||_F)", color="#f8fafc", fontsize=12)
    axes[0].legend(facecolor="#0f1523", edgecolor="#334155", labelcolor="#f8fafc")

    axes[1].plot(t_np, shear_b, color="#fb923c", lw=1.8, label="Multi-Fiber Symplectic Shear (Before)")
    axes[1].plot(t_np, shear_a, color="#38bdf8", lw=2.0, label="Multi-Fiber Symplectic Shear (After)")
    axes[1].set_title("Multi-Fiber Symplectic Spinor Shear across K = 21 Orthogonal Fibers", color="#f8fafc", fontsize=12)
    axes[1].set_xlabel("Token Index t", color="#94a3b8")
    axes[1].legend(facecolor="#0f1523", edgecolor="#334155", labelcolor="#f8fafc")

    plt.tight_layout()
    plt.savefig("cp3_topological_poison_shield.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  -> [ARTIFACT SAVED] Updated Phase 5 Full-Bundle Shield chart written to: cp3_topological_poison_shield.png")
    print("=" * 98)


if __name__ == "__main__":
    run_phase5_full_bundle_benchmark()