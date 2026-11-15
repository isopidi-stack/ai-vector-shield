import unittest
import torch
from casimir_harmonics import PenroseTwistorHarmonicEngine
from topological_poison_shield import FullTwistorIncidenceBundleShield
from metacognition_shell import CP3MetacognitionShell

class SingleFiberTwistorShield:
    """
    Historical baseline single-projection shield (R^d -> R^3 -> CP^1) kept strictly
    to demonstrate why single-fiber models fail against white-box nullspace attacks.
    """
    def __init__(self, d_model: int = 63, device=None):
        self.device = device or torch.device("cpu")
        self.harmonics = PenroseTwistorHarmonicEngine(device=self.device)
        torch.manual_seed(42)
        self.W_proj = torch.randn((d_model, 3), dtype=torch.float64, device=self.device) / (d_model ** 0.5)

    def detect_anomalies_mad(self, H: torch.Tensor, z_thresh: float = 3.5):
        r = H @ self.W_proj
        pi = self.harmonics.points_to_celestial_spinors(r)
        pi_curr, pi_prev = pi[1:], pi[:-1]
        shear = torch.abs(pi_curr[:, 0] * pi_prev[:, 1] - pi_curr[:, 1] * pi_prev[:, 0])
        score = torch.cat([torch.zeros(1, dtype=H.dtype, device=H.device), shear], dim=0)
        med = torch.median(score)
        mad = torch.median(torch.abs(score - med)) + 1e-12
        z_scores = (score - med) / (1.4826 * mad)
        return z_scores > z_thresh, z_scores, score


class LogDilatonBundleShield:
    """
    Historical intermediate 21-fiber log-dilaton bundle kept strictly to demonstrate
    the logarithmic radial-concentration exploit under white-box PGD.
    """
    def __init__(self, d_model: int = 63, device=None):
        self.device = device or torch.device("cpu")
        self.num_fibers = d_model // 3
        self.harmonics = PenroseTwistorHarmonicEngine(device=self.device)
        torch.manual_seed(42)
        Q, _ = torch.linalg.qr(torch.randn((d_model, d_model), dtype=torch.float64, device=self.device))
        self.Q = Q

    def compute_bundle_metrics(self, H: torch.Tensor):
        T = H.shape[0]
        R_fibers = (H @ self.Q).view(T, self.num_fibers, 3)
        pi_fibers = self.harmonics.points_to_celestial_spinors(R_fibers)
        pi_curr, pi_prev = pi_fibers[1:], pi_fibers[:-1]
        fiber_shear = torch.abs(pi_curr[:, :, 0] * pi_prev[:, :, 1] - pi_curr[:, :, 1] * pi_prev[:, :, 0])
        radii = torch.norm(R_fibers, dim=-1).clamp(min=1e-12)
        dilaton_jump = torch.abs(torch.log(radii[1:]) - torch.log(radii[:-1]))
        step_anomaly = torch.sqrt(torch.sum(fiber_shear**2 + dilaton_jump**2, dim=-1))
        return torch.cat([torch.zeros(1, dtype=H.dtype, device=H.device), step_anomaly], dim=0)

    def detect_anomalies_mad(self, H: torch.Tensor, z_thresh: float = 3.5):
        score = self.compute_bundle_metrics(H)
        med = torch.median(score)
        mad = torch.median(torch.abs(score - med)) + 1e-12
        z_scores = (score - med) / (1.4826 * mad)
        return z_scores > z_thresh, z_scores, score


class TestProductionPGDRedTeamAndCasimirFunctional(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(101)
        self.T = 64
        self.d = 63
        t_grid = torch.linspace(0.0, 2.0 * 3.141592653589793, self.T, dtype=torch.float64).unsqueeze(-1)
        freqs = torch.arange(1, self.d + 1, dtype=torch.float64).unsqueeze(0) * 0.1
        self.H_clean = torch.sin(t_grid * freqs) + 0.5 * torch.cos(t_grid * freqs * 0.5)
        self.attack_indices = [16, 32, 48]

    def test_1_naive_static_attack_baseline(self):
        shield = SingleFiberTwistorShield(d_model=self.d)
        H_naive = self.H_clean.clone()
        for idx in self.attack_indices:
            H_naive[idx] = H_naive[idx] + 5.0 * torch.randn(self.d, dtype=torch.float64)

        detected, z_scores, _ = shield.detect_anomalies_mad(H_naive, z_thresh=3.5)
        hits = sum(bool(detected[idx].item()) or bool(detected[idx+1].item()) for idx in self.attack_indices)
        print(f"  -> [RED-TEAM 1: Naive Static Attack] Single-Fiber Recall: {hits}/3 (100.0%) | Max Z: {z_scores[self.attack_indices].max().item():.2f}")
        self.assertEqual(hits, 3)

    def test_2_whitebox_nullspace_evades_single_fiber(self):
        shield = SingleFiberTwistorShield(d_model=self.d)
        U, _, _ = torch.linalg.svd(shield.W_proj, full_matrices=True)
        U_null = U[:, 3:]

        H_null = self.H_clean.clone()
        for idx in self.attack_indices:
            z_c = torch.randn(self.d - 3, dtype=torch.float64)
            H_null[idx] = H_null[idx] + U_null @ (z_c / torch.norm(z_c) * 25.0)

        detected, z_scores, _ = shield.detect_anomalies_mad(H_null, z_thresh=3.5)
        hits = sum(bool(detected[idx].item()) or bool(detected[idx+1].item()) for idx in self.attack_indices)
        print(f"  -> [RED-TEAM 2: White-Box Nullspace Attack] Single-Fiber Recall: {hits}/3 (EVADED!) | Max Z: {z_scores[self.attack_indices].max().item():.4f}")
        self.assertEqual(hits, 0)

    def test_3_whitebox_pgd_evades_log_dilaton_bundle(self):
        log_bundle = LogDilatonBundleShield(d_model=self.d)
        idx = 32
        delta = torch.randn(self.d, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=0.1)

        for _ in range(120):
            optimizer.zero_grad()
            H_temp = self.H_clean.clone()
            H_temp[idx] = self.H_clean[idx] + (delta / (torch.norm(delta) + 1e-12) * 8.0)
            step_anom = log_bundle.compute_bundle_metrics(H_temp)
            loss = step_anom[idx]**2 + step_anom[idx+1]**2
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            H_pgd = self.H_clean.clone()
            H_pgd[idx] = self.H_clean[idx] + (delta / torch.norm(delta) * 8.0)

        det_log, z_log, _ = log_bundle.detect_anomalies_mad(H_pgd, z_thresh=3.5)
        caught_log = bool(det_log[idx].item()) or bool(det_log[idx+1].item())
        print(f"  -> [RED-TEAM 3: White-Box PGD vs Log-Dilaton Bundle] Caught: {caught_log} (EVADED! Max Z = {max(z_log[idx].item(), z_log[idx+1].item()):.2f} < 3.5)")
        self.assertFalse(caught_log)

    def test_4_production_shield_defeats_whitebox_pgd_with_zero_false_positives(self):
        """
        Directly attacks the IMPORTED PRODUCTION detector FullTwistorIncidenceBundleShield
        (from topological_poison_shield.py) using 80-step White-Box PGD (||delta||_2 = 8.0),
        and verifies:
          1. The poisoned token idx=32 is caught by detect_and_purify().
          2. False Positive Rate across all 63 unpoisoned clean tokens (~gt_mask) is 0.0%.
          3. Clean token distortion across ALL ground-truth clean tokens (~gt_mask) is 0.00e+00.
        """
        prod_shield = FullTwistorIncidenceBundleShield(d_model=self.d, z_thresh=3.5, device=torch.device("cpu"))
        idx = 32
        gt_mask = torch.zeros(self.T, dtype=torch.bool)
        gt_mask[idx] = True

        delta = torch.randn(self.d, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=0.1)

        for _ in range(80):
            optimizer.zero_grad()
            H_temp = self.H_clean.clone()
            H_temp[idx] = self.H_clean[idx] + (delta / (torch.norm(delta) + 1e-12) * 8.0)
            diag = prod_shield.compute_bundle_diagnostics(H_temp)
            anom = diag["anomaly_score"]
            loss = anom[idx-1]**2 + anom[idx]**2 + anom[idx+1]**2
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            H_pgd_prod = self.H_clean.clone()
            H_pgd_prod[idx] = self.H_clean[idx] + (delta / torch.norm(delta) * 8.0)

        res = prod_shield.detect_and_purify(H_pgd_prod, z_thresh=3.5)
        q_mask = res["quarantine_mask"]
        caught_prod = bool(q_mask[idx].item())
        z_at_idx = res["z_scores"][idx].item()

        # Audit ALL ground-truth clean tokens (~gt_mask) for False Positives and distortion
        false_positives = int(torch.sum(q_mask & (~gt_mask)).item())
        all_clean_distortion = float(torch.max(torch.abs(res["H_healed"][~gt_mask] - self.H_clean[~gt_mask])).item())

        print(f"  -> [RED-TEAM 4: Production Shield vs White-Box PGD] Caught: {caught_prod} (Z = {z_at_idx:.2f}) | False Positives: {false_positives}/63 | All-Clean Distortion (~gt_mask): {all_clean_distortion:.2e}")
        self.assertTrue(caught_prod)
        self.assertGreater(z_at_idx, 15.0)
        self.assertEqual(false_positives, 0)
        self.assertLess(all_clean_distortion, 1e-15)

    def test_5_non_degenerate_casimir_rayleigh_quotient_in_phase4_and_phase5(self):
        """
        Verifies the second review's Casimir functional fix:
          1. In Phase 5 (FullTwistorIncidenceBundleShield), casimir_jump is genuinely non-zero (> 0.01)
             rather than vanishing identically via Unsold's addition theorem.
          2. In Phase 4 (CP3MetacognitionShell), both the Casimir Rayleigh quotient and the
             twistor incidence residual ||omega_t - i X_t pi_{t-1}||_2 are genuinely non-zero.
        """
        prod_shield = FullTwistorIncidenceBundleShield(d_model=self.d, device=torch.device("cpu"))
        H_test = self.H_clean.clone()
        H_test[20] = H_test[20] + 6.0 * torch.randn(self.d, dtype=torch.float64)

        diag5 = prod_shield.compute_bundle_diagnostics(H_test)
        max_casimir_jump = float(torch.max(diag5["casimir_jump"]).item())

        shell4 = CP3MetacognitionShell(d_model=self.d, device=torch.device("cpu"))
        diag4 = shell4.evaluate_trajectory_metrics(H_test)
        max_strain4 = float(torch.max(diag4["casimir_strain"]).item())
        max_inc_res4 = float(torch.max(diag4["incidence_residual"]).item())

        print(f"  -> [RED-TEAM 5: Non-Degenerate Casimir & Incidence] Phase 5 Casimir Jump: {max_casimir_jump:.4f} (>0) | Phase 4 Casimir: {max_strain4:.4f} | Phase 4 Incidence Residual: {max_inc_res4:.4f} (>0)")
        self.assertGreater(max_casimir_jump, 0.05)
        self.assertGreater(max_strain4, 0.05)
        self.assertGreater(max_inc_res4, 1.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)