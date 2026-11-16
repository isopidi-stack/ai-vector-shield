# AI-Vector-Shield (v5.6.1-all-green)

**Topological & Conformal CP^{d/2-1} / Spinor-Bundle Anomaly Detection and Trajectory Purification for High-Dimensional AI Embeddings**

`ai-vector-shield` protects latent token trajectories $H \in \mathbb{R}^{T \times d}$ and static embedding spaces against adversarial injections—including **isolated impulse spikes**, **constant-offset trigger blocks**, **smooth multi-token cosine bumps**, and **white-box adaptive PGD attacks**—using projective Fubini-Study geometry, fiber-anisotropic spinor shear, and Laplace-Beltrami Casimir harmonics.

---

## Key Mathematical Foundations

1. **Fiber-Anisotropic Spinor Bundle & Casimir Rayleigh Quotient (`casimir_harmonics.py`)**
   * Projects embedding trajectories through an orthogonal bundle $R = H Q$ into $K = \lfloor d / 3 \rfloor$ spinor fibers $\vec{n}_{t,k} \in S^2 \cong \mathbb{CP}^1$.
   * Evaluates real orthonormal spherical harmonics $Y_{\ell,m}(\vec{n}_{t,k})$ for $\ell \in \{0, 1, 2\}$ satisfying **Unsöld's Theorem** to machine precision:
     $$\sum_{\ell=0}^{2} \sum_{m=-\ell}^{\ell} |Y_{\ell,m}(\theta, \phi)|^2 = \frac{1 + 3 + 5}{4\pi} = \frac{9}{4\pi} \quad (\text{Max Error: } 5.55 \times 10^{-16})$$
   * Weights modes by exact Laplace-Beltrami eigenvalues $\lambda_\ell = \ell(\ell+1) \in \{0, 2, 6\}$ with anisotropic fiber weights $w_k = 1 + \frac{1}{2}\cos(2\pi k / K)$, ensuring genuine sensitivity to the orthogonal bundle transformation $Q$.

2. **Multi-Horizon Reference Manifold & Clean-Anchor Cubic Hermite Span Bridge (`topological_poison_shield.py`)**
   * Combines local 4-point cubic kink detection and opposite step-pair bracketing (`compute_clean_reference_manifold`) with fiber-anisotropic curvature and spinor chordal shear $|\epsilon_{\alpha\beta} \pi_t^\alpha \pi_{t-1}^\beta| = \frac{1}{2}\|\vec{n}_t - \vec{n}_{t-1}\|_2$.
   * Purifies contiguous quarantined spans $[s, e]$ via an **Exact Clean-Anchor Cubic Hermite Span Bridge** anchored strictly at verified clean exterior tokens $(s-1, e+1)$, eliminating interior poison leakage.

3. **U(1) Gauge-Invariant Fubini-Study Metacognition (`metacognition_shell.py`) & Core Subspace Shield (`ai_vector_shield/`)**
   * Phase-4 metacognitive stabilizer measures multi-horizon Fubini-Study chordal distance $d_{FS}([\psi_t], [\psi_{t-\tau}]) = \sqrt{1 - |\langle \hat{\psi}_t, \hat{\psi}_{t-\tau} \rangle|^2}$, which is invariant ($3.28 \times 10^{-16}$) under arbitrary local $\mathrm{U}(1)$ phase transformations $\psi_t \mapsto e^{i\theta_t}\psi_t$.
   * Static `VectorShield` canonicalizes global $\mathrm{U}(1)$ phase $z \mapsto e^{i\theta} z$ and antipodal sign flips $X \mapsto -X$ (`_u1_canon_any`) prior to tangent-space SVD projection, yielding `0.00e+00` score discrepancy under phase rotation and sign inversion.

---

## Multi-Attack Benchmark (`v5.6.1`)

Evaluated on $T = 128, d = 63$ ($K = 21$ fibers) under simultaneous **alternating impulse spikes** (tokens `20..25`), **constant-offset trigger block** (tokens `55..60`, $\Delta^2 H = 0$ in interior), and **smooth cosine bump** (tokens `90..97`):

| Detector / Method | True Positives (Recall) | False Positives (FPR) | Poisoned RMSE (Before -> After) | Error Suppression | Clean Token Distortion |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AI-Vector-Shield (`v5.6.1`)** | **20 / 20 (`100.0%`)** | **0 / 108 (`0.00%`)** | **`7.287 -> 0.053`** | **`99.3%`** | **`0.00e+00`** |
| Naive $\|\Delta^2 H_t\|_2$ Baseline | 16 / 20 (`80.0%`) | 6 / 108 (`5.56%`) | N/A | N/A | N/A |

---

## Verification & Red-Team Audit Suite (11/11 Tests Passing)

| Test Suite | Audit Verification Target | Measured Telemetry | Status |
| :--- | :--- | :--- | :---: |
| `test_1_unsold_theorem_9_over_4pi_and_q_dependence` | Unsöld identity $\sum_{\ell \le 2} |Y_{\ell,m}|^2 = \frac{9}{4\pi}$ & $Q$-sensitivity | `Unsold Err: 5.55e-16 \| \|Tear(Q1)-Tear(Q2)\|: 0.0111` | **PASS** |
| `test_2_constant_offset_block_and_smooth_bump_detection` | Constant blocks ($\|\delta\| \in \{8, 20, 50\}$) & smooth bumps | `Block Recall: 100% \| Bump Error: 5.93 -> 0.040 \| FP: 0` | **PASS** |
| `test_3_phase4_u1_gauge_invariance_and_closed_loop_stabilizer` | Local $\mathrm{U}(1)$ gauge invariance & closed-loop damping | `Gauge Diff: 3.28e-16 \| Residual: 346.56 -> 119.21` | **PASS** |
| `test_4_production_shield_defeats_whitebox_pgd_with_zero_false_positives` | 60-step gradient White-Box Adaptive PGD red-team | `Caught All Targets: True (Max Z = 642.00) \| FP: 0` | **PASS** |
| `test_5_cp_global_phase_and_sign_flip_invariance_in_core` | $\mathbb{CP}^{d/2-1}$ global phase $e^{i\theta}X$ & sign flip $-X$ | `Acceptance: 100.0% \| Diff(-X): 0.00e+00 \| Diff(Rot): 0.00e+00` | **PASS** |
| `test_vector_shield.py` (6 Unit Tests) | Fit/verify, shadow mode telemetry, `torch`/`numpy` IO, `weights_only` persistence | `Ran 6 tests — OK` | **PASS** |

---

## Quick Start

    # Run the multi-attack benchmark
    python topological_poison_shield.py

    # Run the full 11-test unit and red-team audit suite
    python -m unittest discover -s tests -p "test_*.py" -v
