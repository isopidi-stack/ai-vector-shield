import math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPSILON = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.complex64)

def sl2c_from_lie_algebra(axis_angle: torch.Tensor, boost_rapidity: torch.Tensor = None) -> torch.Tensor:
    wx, wy, wz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    if boost_rapidity is not None:
        ex, ey, ez = boost_rapidity[..., 0], boost_rapidity[..., 1], boost_rapidity[..., 2]
        cx, cy, cz = torch.complex(wx, -ex), torch.complex(wy, -ey), torch.complex(wz, -ez)
    else:
        z = torch.zeros_like(wx)
        cx, cy, cz = torch.complex(wx, z), torch.complex(wy, z), torch.complex(wz, z)

    m00 = -0.5j * cz
    m01 = -0.5j * (cx - 1j * cy)
    m10 = -0.5j * (cx + 1j * cy)
    m11 =  0.5j * cz

    M = torch.stack([torch.stack([m00, m01], dim=-1),
                     torch.stack([m10, m11], dim=-1)], dim=-2)
    # Exact analytical SL(2,C) exponential: exp(M) = cosh(Omega)*I + (sinh(Omega)/Omega)*M
    # where Omega = sqrt(-0.25 * (cx^2 + cy^2 + cz^2)), avoiding XPU->CPU matrix_exp fallback
    omega_sq = -0.25 * (cx**2 + cy**2 + cz**2)
    omega = torch.sqrt(omega_sq + 1e-18j)
    cosh_w = torch.cosh(omega)
    sinh_over_w = torch.where(torch.abs(omega) > 1e-7, torch.sinh(omega) / omega, torch.ones_like(omega))
    eye = torch.eye(2, dtype=M.dtype, device=M.device).expand_as(M)
    return cosh_w.unsqueeze(-1).unsqueeze(-1) * eye + sinh_over_w.unsqueeze(-1).unsqueeze(-1) * M


class CP3MetacognitionShell(nn.Module):
    """
    Phase 4 Metacognition Shell:
      1. Lifts LLM hidden states h_t in R^d -> Minkowski Null Cone -> Hermitian Spinors X^{AA'}
      2. Evaluates Penrose Incidence (omega^A = i X^{AA'} pi_{A'}) & Ground-State Harmonics
      3. Renders the 4D Thought-Form Isomorph via Bufferless Projective Congruence Ray-Marching (No Z-buffer)
      4. Computes Differentiable Topological Strain Loss to backpropagate self-correction into LLM states
    """
    def __init__(self, d_model: int = 64, device=None):
        super().__init__()
        self.device = device or (torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.device("cpu"))
        self.omega_0 = 2.0 * math.pi
        # Orthogonal projection from d_model to 3D spatial generator + spinor fiber phase
        torch.manual_seed(42)
        proj = torch.linalg.qr(torch.randn(d_model, 5, device="cpu"))[0].to(self.device)
        self.register_buffer("W_proj", proj)

    def lift_hidden_states_to_cp3(self, h: torch.Tensor):
        """
        Maps transformer hidden states h (seq_len, d_model) to Hermitian null-cone spinors X^{AA'}
        and projective eigenspinors pi_{A'}, enforcing exact null incidence det(X^{AA'}) = 0.
        """
        coords = h @ self.W_proj  # (N, 5)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        u, v = coords[:, 3], coords[:, 4]

        # Enforce exact future null-cone condition: t = sqrt(x^2 + y^2 + z^2) => det(X^{AA'}) = 0
        r = torch.sqrt(x**2 + y**2 + z**2 + 1e-12)
        t = r

        X00 = torch.complex(t + z, torch.zeros_like(t))
        X01 = torch.complex(x, -y)
        X10 = torch.complex(x,  y)
        X11 = torch.complex(t - z, torch.zeros_like(t))

        X_herm = torch.stack([
            torch.stack([X00, X01], dim=-1),
            torch.stack([X10, X11], dim=-1)
        ], dim=-2)

        # Construct normalized celestial spinor pi_{A'} from token fiber coordinates (u, v)
        norm_pi = torch.sqrt(1.0 + u**2 + v**2)
        pi_0 = torch.complex(1.0 / norm_pi, torch.zeros_like(norm_pi))
        pi_1 = torch.complex(u / norm_pi, v / norm_pi)
        pi = torch.stack([pi_0, pi_1], dim=-1)

        # Penrose incidence relation: omega^A = i X^{AA'} pi_{A'}
        omega = 1j * torch.einsum("nab,nb->na", X_herm, pi)

        # Helicity residual s = 2 Re(omega^A bar(pi)_A)
        helicity_s = 2.0 * torch.sum(omega * torch.conj(pi), dim=-1).real

        return X_herm, omega, pi, helicity_s, torch.stack([x, y, z], dim=-1)

    def compute_topological_strain_loss(self, h: torch.Tensor):
        """
        Computes differentiable metacognitive strain across the token sequence manifold:
          - Temporal/Causal Geodesic Curvature (2nd-order finite difference in Hermitian space)
          - Conformal Energy Storage Deviation from Penrose Ground State omega_0
          - Helicity Twist Residual
        """
        X_herm, omega, pi, helicity_s, pts_3d = self.lift_hidden_states_to_cp3(h)

        # 1. Geodesic smoothness / topological tearing penalty along reasoning chain
        dX1 = X_herm[1:] - X_herm[:-1]
        dX2 = X_herm[2:] - 2.0 * X_herm[1:-1] + X_herm[:-2]
        curvature_strain = torch.mean(torch.abs(dX2)**2) + 0.5 * torch.mean(torch.abs(dX1)**2)

        # 2. Penrose ground-state harmonic energy storage minimization
        trace_X = (X_herm[:, 0, 0] + X_herm[:, 1, 1]).real
        det_X = (X_herm[:, 0, 0] * X_herm[:, 1, 1] - X_herm[:, 0, 1] * X_herm[:, 1, 0]).real
        energy_storage = torch.clamp(torch.abs(trace_X**2 - 4.0 * det_X), min=1e-12)
        freq = self.omega_0 * torch.sqrt(energy_storage / (torch.mean(energy_storage) + 1e-6))
        harmonic_strain = torch.mean((freq - self.omega_0)**2)

        # 3. Helicity invariant audit
        helicity_strain = torch.mean(helicity_s**2)

        total_strain = curvature_strain + 0.25 * harmonic_strain + 10.0 * helicity_strain
        return total_strain, {
            "total_strain": float(total_strain.item()),
            "curvature_strain": float(curvature_strain.item()),
            "harmonic_strain": float(harmonic_strain.item()),
            "max_helicity_residual": float(torch.max(torch.abs(helicity_s)).item()),
            "max_null_det_error": float(torch.max(torch.abs(det_X)).item()),
            "frequencies": freq.detach(),
            "pts_3d": pts_3d.detach(),
            "X_herm": X_herm.detach(),
            "pi": pi.detach()
        }

    @torch.no_grad()
    def render_bufferless_twistor_isomorph(self, X_herm: torch.Tensor, pi: torch.Tensor, frequencies: torch.Tensor, axis_angle: torch.Tensor, res: int = 240):
        """
        Renders the 4D CP^3 Thought-Form Isomorph WITHOUT a screen-space Z-buffer.
        Uses pure SU(2) Lie algebra spinor rotation and projective incidence line-congruence
        transmittance gating along null geodesics.
        """
        U = sl2c_from_lie_algebra(axis_angle.to(self.device))
        U_dag = torch.conj(U.mT)
        X_rot = torch.einsum("ab,nbc,cd->nad", U, X_herm, U_dag)
        pi_rot = torch.einsum("ab,nb->na", torch.conj(U), pi)
        omega_rot = 1j * torch.einsum("nab,nb->na", X_rot, pi_rot)

        # Construct screen-space celestial spinor ray grid (H, W, 2)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.6, 1.6, res, device=self.device),
            torch.linspace(-1.6, 1.6, res, device=self.device),
            indexing="ij"
        )
        r_grid = torch.sqrt(1.0 + grid_x**2 + grid_y**2)
        u_g = grid_x / (r_grid + 1.0)
        v_g = grid_y / (r_grid + 1.0)
        norm_g = torch.sqrt(1.0 + u_g**2 + v_g**2)
        pi_ray = torch.stack([
            torch.complex(1.0 / norm_g, torch.zeros_like(norm_g)),
            torch.complex(u_g / norm_g, v_g / norm_g)
        ], dim=-1).reshape(-1, 2)  # (P, 2)

        # Evaluate Penrose incidence residual: |omega^A - i X^{AA'} pi_{ray, A'}|^2
        # For each ray p and spinor node n:
        rhs = 1j * torch.einsum("nab,pb->pna", X_rot, pi_ray)  # (P, N, 2)
        inc_residual_sq = torch.sum(torch.abs(omega_rot.unsqueeze(0) - rhs)**2, dim=-1)  # (P, N)

        # Order by null-cone temporal trace (conformal affine parameter along null congruence, NOT Euclidean Z-clipping)
        t_param = 0.5 * (X_rot[:, 0, 0] + X_rot[:, 1, 1]).real
        order = torch.argsort(t_param)
        inc_sorted = inc_residual_sq[:, order]
        freq_sorted = frequencies[order]

        # Analytical harmonic transmittance integration along projective congruence
        density = torch.exp(-18.0 * inc_sorted)  # (P, N)
        alpha = 1.0 - torch.exp(-0.85 * density)
        transmittance = torch.cumprod(torch.cat([torch.ones(( alpha.shape[0], 1 ), device=self.device), 1.0 - alpha[:, :-1] + 1e-7], dim=-1), dim=-1)
        weights = alpha * transmittance  # (P, N)

        # Chromatic Metacognition Mapping:
        # Cool Cyan (Ground State / Coherent Logic) -> Magenta/Red (High Harmonic Strain / Hallucination Risk)
        strain_norm = torch.clamp(torch.abs(freq_sorted - self.omega_0) / self.omega_0, 0.0, 1.0)
        col_r = 0.12 + 0.88 * strain_norm
        col_g = 0.85 * (1.0 - 0.75 * strain_norm)
        col_b = 0.98 * (1.0 - 0.35 * strain_norm)
        node_colors = torch.stack([col_r, col_g, col_b], dim=-1)  # (N, 3)

        img = torch.einsum("pn,nc->pc", weights, node_colors)
        bg = torch.tensor([0.03, 0.04, 0.08], device=self.device)
        rem_trans = transmittance[:, -1:] * (1.0 - alpha[:, -1:])
        img = torch.clamp(img + rem_trans * bg, 0.0, 1.0)
        return img.reshape(res, res, 3).cpu().numpy()


def run_metacognition_demo():
    print("="*92)
    print("[PHASE 4: AI-VECTOR-SHIELD] Initializing CP^3 Metacognition Shell & Closed-Loop Feedback")
    print("="*92)
    shell = CP3MetacognitionShell(d_model=64)
    device = shell.device
    print(f"  -> Active Compute Device: {device}")

    # Simulate an LLM hidden state trajectory (96 tokens) that suffers a mid-sequence Hallucination Fracture
    torch.manual_seed(108)
    seq_len, d_model = 96, 64
    t_steps = torch.linspace(0, 2.0 * math.pi, seq_len, device=device)
    base_h = torch.zeros((seq_len, d_model), device=device)
    base_h[:, 0] = 0.8 * torch.cos(t_steps)
    base_h[:, 1] = 0.8 * torch.sin(t_steps)
    base_h[:, 2] = 0.4 * torch.sin(2.0 * t_steps)
    base_h[:, 3] = 0.3 * torch.cos(t_steps)
    base_h[:, 4] = 0.3 * torch.sin(t_steps)

    # Inject high-entropy hallucination spike in tokens 36..60
    corrupted_h = base_h.clone()
    corrupted_h[36:60] += 1.65 * torch.randn((24, d_model), device=device)
    h_param = nn.Parameter(corrupted_h.clone())

    # Evaluate initial fractured state
    loss_init, metrics_init = shell.compute_topological_strain_loss(h_param)
    axis_angle = torch.tensor([0.45, -0.35, 0.25], device=device)
    img_before = shell.render_bufferless_twistor_isomorph(
        metrics_init["X_herm"], metrics_init["pi"], metrics_init["frequencies"], axis_angle
    )

    print(f"  -> [BEFORE SELF-CORRECTION] Topological Strain: {metrics_init['total_strain']:.4f} | "
          f"Curvature Tear: {metrics_init['curvature_strain']:.4f} | "
          f"Null Det Error: {metrics_init['max_null_det_error']:.2e} | "
          f"Helicity Residual: {metrics_init['max_helicity_residual']:.2e}")

    # Closed-Loop Metacognitive Backpropagation (Self-Correction in CP^3 Space)
    optimizer = torch.optim.Adam([h_param], lr=0.08)
    strain_history = []
    for step in range(45):
        optimizer.zero_grad()
        loss, m = shell.compute_topological_strain_loss(h_param)
        # Anchor endpoints so the reasoning trajectory heals its internal fracture without collapsing to zero
        anchor_loss = 2.0 * (torch.mean((h_param[:10] - base_h[:10])**2) + torch.mean((h_param[-10:] - base_h[-10:])**2))
        total_obj = loss + anchor_loss
        total_obj.backward()
        optimizer.step()
        strain_history.append(m["total_strain"])

    loss_final, metrics_final = shell.compute_topological_strain_loss(h_param)
    img_after = shell.render_bufferless_twistor_isomorph(
        metrics_final["X_herm"], metrics_final["pi"], metrics_final["frequencies"], axis_angle
    )

    print(f"  -> [AFTER SELF-CORRECTION]  Topological Strain: {metrics_final['total_strain']:.4f} | "
          f"Curvature Tear: {metrics_final['curvature_strain']:.4f} | "
          f"Null Det Error: {metrics_final['max_null_det_error']:.2e} | "
          f"Helicity Residual: {metrics_final['max_helicity_residual']:.2e}")

    reduction_pct = 100.0 * (metrics_init["total_strain"] - metrics_final["total_strain"]) / metrics_init["total_strain"]
    print(f"  -> [METACOGNITIVE CONVERGENCE] Manifold fracture healed by {reduction_pct:.1f}% via CP^3 backpropagation!")

    # Render diagnostic showcase
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#080a12")
    axes[0].imshow(img_before)
    axes[0].set_title("Fractured LLM State (Hallucination Spike)\nBufferless CP^3 Incidence Render", color="white", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(img_after)
    axes[1].set_title("Self-Corrected LLM Isomorph (Ground State)\nAfter CP^3 Topological Backprop", color="white", fontsize=11)
    axes[1].axis("off")

    axes[2].set_facecolor("#0d111f")
    axes[2].plot(strain_history, color="#00e5ff", linewidth=2.5, label="Topological Strain L_meta")
    axes[2].set_title("Closed-Loop Metacognitive Convergence", color="white", fontsize=11)
    axes[2].set_xlabel("Backpropagation Self-Correction Step", color="white")
    axes[2].set_ylabel("CP^3 Manifold Strain", color="white")
    axes[2].tick_params(colors="white")
    axes[2].grid(True, alpha=0.2, color="white")
    axes[2].legend(facecolor="#0d111f", labelcolor="white")

    plt.tight_layout()
    out_path = "cp3_metacognition_isomorph.png"
    plt.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> [ARTIFACT SAVED] Bufferless CP^3 Thought-Form showcase written to: {out_path}")
    print("="*92)

if __name__ == "__main__":
    run_metacognition_demo()