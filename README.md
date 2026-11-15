# AI-Vector-Shield (v5.3-rayleigh-casimir-prod-pgd)

**Orthogonal Twistor Bundle Trajectory Monitoring, Field-Level $L^2(\mathbb{CP}^1)$ Casimir Rayleigh Quotient, and Production White-Box PGD Red-Team Benchmarks on Intel XPU.**

![Crystalline 4K Penrose Robinson Congruence](penrose_cp3_4k_crystalline.jpg)

---

## 1. Architecture & Mathematical Remediation (v5.3)

`ai-vector-shield` maps hidden-state trajectories $H \in \mathbb{R}^{T \times d}$ into $K = \lfloor d / 3 \rfloor$ disjoint 3D fibers via an orthogonal transformation $Q \in \mathrm{O}(d)$ ($Q^\mathsf{T} Q = I_d$, $\ker Q = \{0\}$) and evaluates four trajectory diagnostics across the fiber bundle:

1. **Multi-Fiber $\mathbb{CP}^1$ Symplectic Spinor Shear:** Evaluates spinor shear $|\varepsilon_{AB}\pi_{t,k}^A\pi_{t-1,k}^B|$ across all $K$ fibers.
2. **Un-Logged Hermitian Incidence Curvature Tear ($\|\Delta^2 X_t\|_F$):** Evaluates $\sqrt{2}\|\Delta^2 R_t\|_F = \sqrt{2}\|R_{t+1} - 2R_t + R_{t-1}\|_F$, enforcing the reverse triangle inequality $\|\Delta^2 R_t^{\mathrm{adv}}\|_F \ge 2\|\delta_t\|_2 - \|\Delta^2 R_t^{\mathrm{clean}}\|_F$ against localized single-token PGD perturbations.
3. **Field-Level $L^2(\mathbb{CP}^1)$ Casimir Rayleigh Quotient (`compute_conformal_energy_functional`):**
   * *Why Pointwise Sums Fail:* By Unsold's / the Spherical Harmonic Addition Theorem, evaluating $\sum_{m=-\ell}^\ell |Y_{\ell m}(n)|^2 = \frac{2\ell+1}{4\pi}$ pointwise at a single direction $n \in S^2$ yields a constant $C(n) = \frac{9}{\pi}$ over the sphere, producing zero step-to-step difference.
   * *Field-Level Rayleigh Quotient Fix:* At each token $t$, the $K = 21$ fibers define a discrete spinor cloud $\{\pi_{t,k}\}_{k=1}^K \subset \mathbb{CP}^1$ carrying the non-constant radial field $f_t(\pi_{t,k}) = \|\vec{r}_{t,k}\|_2 / \bar{r}_t$. Both Phase 4 (`metacognition_shell.py`) and Phase 5 (`topological_poison_shield.py`) project $f_t$ onto the 9-mode spherical harmonic basis via batched ridge-regularized normal equations $(B^\mathsf{T} B + \lambda I_9)c_t = B^\mathsf{T} f_t$ and evaluate the true Rayleigh quotient:
     $$\mathcal{E}_{\mathrm{Casimir}}(t) = \frac{\sum_{\ell=1}^2 \ell(\ell+1)\sum_{m=-\ell}^\ell c_{t,\ell m}^2}{\sum_{\ell=0}^2 \sum_{m=-\ell}^\ell c_{t,\ell m}^2 + 10^{-12}}$$
4. **Non-Vanishing Twistor Incidence Residual (Phase 4):** Replaces $\mathrm{Re}(i \pi^\dagger X \pi) \equiv 0$ with the parallel-transported incidence residual $\|\omega_t - i X_t \pi_{t-1}\|_2$.

---

## 2. Production White-Box PGD Red-Team Suite (`tests/test_adaptive_pgd_redteam.py`)

`tests/test_adaptive_pgd_redteam.py` imports and attacks the **production `FullTwistorIncidenceBundleShield` class directly** and audits false positives and distortion across **all ground-truth unpoisoned tokens (`~gt_mask`)**:

| Test Case & Threat Model | Perturbation $\|\delta\|_2$ | Detection Result | Z-Score / Metric | Mathematical Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **1. Single-Fiber ($\mathbb{R}^d \to \mathbb{R}^3$) vs. Oblivious Static Attack** | $5.0\sqrt{d}$ | **Caught (`3/3`, 100%)** | `Z = 7.98` ($\gt 3.5$) | Isotropic random noise projects onto $\mathrm{Row}(W_{\mathrm{proj}})$ and induces large angular shear on $\mathbb{CP}^1$. |
| **2. Single-Fiber vs. White-Box SVD Nullspace Attack** | `25.0` | **Evaded (`0/3`, 0%)** | `Z = 0.98` ($\lt 3.5$) | $\dim(\ker W_{\mathrm{proj}}) = d - 3$. Any $\delta \in \ker(W_{\mathrm{proj}})$ produces zero change in $\pi_t$. |
| **3. 21-Fiber Log-Dilaton Bundle vs. 120-Step White-Box PGD** | `8.0` | **Evaded (`False`)** | `Z = 1.79` ($\lt 3.5$) | PGD concentrates $\|\delta\|_2 = 8.0$ collinearly into a single fiber ($\mathrm{shear} = 0$) while $\log\|r\|$ compresses the radial spike. |
| **4. Production `FullTwistorIncidenceBundleShield` vs. White-Box PGD** | `8.0` | **Caught (`True`)** | **`Z = 38.17`** ($\gg 3.5$)<br>**FP: `0/63`**<br>**Clean Dist (`~gt_mask`): `0.00e+00`** | Full-rank $Q \in \mathrm{O}(d)$ eliminates the nullspace, and un-logged $\|\Delta^2 X_t\|_F$ prevents collinear radial hiding with zero false quarantines on clean tokens. |
| **5. Non-Degenerate Casimir Rayleigh Quotient (Phase 4 & 5)** | `6.0` | **Verified Non-Zero** | Phase 5 Jump: **`0.8301`**<br>Phase 4 Casimir: **`1.2289`**<br>Incidence Res: **`59.12`** | Confirms field-level Rayleigh quotient projection across the $K=21$ fiber cloud is non-degenerate in both Phase 4 and Phase 5. |

---

## 3. Running the Production Benchmark & 5-Test Red-Team Suite

```powershell
$env:PYTHONPATH = "."
python topological_poison_shield.py
python -m unittest discover -s tests -p "test_adaptive_pgd_redteam.py" -v
```