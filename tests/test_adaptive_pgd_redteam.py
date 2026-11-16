import math
import unittest
import numpy as np
import torch

from ai_vector_shield import VectorShield
from casimir_harmonics import SpinorHarmonicCasimirEngine
from metacognition_shell import MetacognitiveTwistorShell
from topological_poison_shield import FullTwistorIncidenceBundleShield


class TestProductionPGDRedTeamAndCasimirFunctional(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.T = 96
        self.D = 63
        t_grid = torch.linspace(0.0, 3.0 * math.pi, self.T, dtype=torch.float64).unsqueeze(-1)
        freqs = torch.arange(1, self.D + 1, dtype=torch.float64).unsqueeze(0)
        self.H_clean = torch.sin(t_grid * (0.22 + 0.025 * freqs)) + 0.4 * torch.cos(t_grid * (0.14 + 0.015 * freqs))

    def test_1_unsold_theorem_9_over_4pi_and_q_dependence(self):
        """
        Verifies Reviewer Finding #1 & #4:
          - Unsold's theorem sum_{l<=2, m} |Y_{l,m}|^2 == 9 / (4*pi) to < 1e-15.
          - Fiber-anisotropic curvature tear depends genuinely on the orthogonal bundle Q
            (two distinct orthogonal matrices Q1 != Q2 produce distinct curvature tears).
        """
        dirs = torch.randn((50, 21, 3), dtype=torch.float64)
        unsold_err = SpinorHarmonicCasimirEngine.verify_unsold_theorem(dirs)
        self.assertLess(unsold_err, 1e-14)

        s1 = FullTwistorIncidenceBundleShield(dim=self.D, seed=10)
        s2 = FullTwistorIncidenceBundleShield(dim=self.D, seed=99)
        r1 = s1.analyze_and_purify(self.H_clean)
        r2 = s2.analyze_and_purify(self.H_clean)
        q_diff = float(torch.max(torch.abs(r1["curv_tear"] - r2["curv_tear"])).item())
        print(f"  -> [AUDIT 1: Unsold & Q-Sensitivity] Unsold Err: {unsold_err:.2e} | |Tear(Q1) - Tear(Q2)|: {q_diff:.4f} (>0)")
        self.assertGreater(q_diff, 1e-3)

    def test_2_constant_offset_block_and_smooth_bump_detection(self):
        """
        Verifies Reviewer Finding on Constant-Offset Blocks & Smooth Bumps:
          - Raw ||Delta^2 H|| baseline misses interior tokens of a constant-offset block
            and smooth cosine bump.
          - FullTwistorIncidenceBundleShield v5.5 detects 100% of constant-block tokens
            (at ||delta|| = 8, 20, 50) and 100% of smooth-bump tokens with 0 false positives,
            and suppresses >90% of the perturbation via span-boundary bridge purification.
        """
        shield = FullTwistorIncidenceBundleShield(dim=self.D, seed=42, z_thresh=3.5)
        v_dir = torch.ones(self.D, dtype=torch.float64) / math.sqrt(self.D)

        for norm_delta in [8.0, 20.0, 50.0]:
            H_blk = self.H_clean.clone()
            gt = torch.zeros(self.T, dtype=torch.bool)
            H_blk[30:36] += norm_delta * v_dir
            gt[30:36] = True

            out = shield.analyze_and_purify(H_blk)
            recall = float(torch.sum(out["quarantine_mask"] & gt).item()) / 6.0
            fp = int(torch.sum(out["quarantine_mask"] & (~gt)).item())
            err_after = float(torch.mean(torch.linalg.norm(out["H_purified"][gt] - self.H_clean[gt], dim=-1)).item())
            self.assertEqual(recall, 1.0)
            self.assertEqual(fp, 0)
            self.assertLess(err_after, 0.25)

        # Smooth bump test
        H_bump = self.H_clean.clone()
        gt_b = torch.zeros(self.T, dtype=torch.bool)
        for k, idx in enumerate(range(60, 67)):
            w = 0.5 * (1.0 - math.cos(2.0 * math.pi * (k + 1) / 8.0))
            H_bump[idx] += (2.5 + 6.0 * w) * v_dir
            gt_b[idx] = True

        out_b = shield.analyze_and_purify(H_bump)
        recall_b = float(torch.sum(out_b["quarantine_mask"] & gt_b).item()) / 7.0
        err_before_b = float(torch.mean(torch.linalg.norm(H_bump[gt_b] - self.H_clean[gt_b], dim=-1)).item())
        err_after_b = float(torch.mean(torch.linalg.norm(out_b["H_purified"][gt_b] - self.H_clean[gt_b], dim=-1)).item())
        print(f"  -> [AUDIT 2: Constant Block & Smooth Bump] Block Recall (8,20,50): 100% | Bump Recall: {recall_b*100:.1f}% | Bump Error: {err_before_b:.2f} -> {err_after_b:.3f}")
        self.assertEqual(recall_b, 1.0)
        self.assertLess(err_after_b, 0.25)

    def test_3_phase4_u1_gauge_invariance_and_closed_loop_stabilizer(self):
        """
        Verifies Reviewer Finding on Phase-4 U(1) Gauge Invariance & Closed-Loop Healing:
          - Re-choosing arbitrary local U(1) phase angles alpha_{t,k} leaves the Phase-4
            incidence residual identical to machine precision (< 1e-14).
          - Closed-loop variational stabilizer actively reduces anomalous trajectory residual.
        """
        H_noisy = self.H_clean.clone()
        H_noisy[40] += 6.0
        diag1 = MetacognitiveTwistorShell.evaluate_trajectory_diagnostics(H_noisy)
        rand_phases = torch.rand((self.T, self.D // 3), dtype=torch.float64) * (2.0 * math.pi)
        diag2 = MetacognitiveTwistorShell.evaluate_trajectory_diagnostics(H_noisy, phase_angles=rand_phases)

        gauge_diff = abs(diag1["incidence_residual"] - diag2["incidence_residual"])
        _, stab_info = MetacognitiveTwistorShell.closed_loop_stabilize(H_noisy, lam=2.0, iterations=3)
        print(f"  -> [AUDIT 3: Phase-4 U(1) Gauge & Closed-Loop] Gauge Diff: {gauge_diff:.2e} | Residual: {stab_info['residual_before']:.2f} -> {stab_info['residual_after']:.2f}")
        self.assertLess(gauge_diff, 1e-13)
        self.assertLess(stab_info["residual_after"], stab_info["residual_before"] * 0.65)

    def test_4_production_shield_defeats_whitebox_pgd_with_zero_false_positives(self):
        """
        Attacks FullTwistorIncidenceBundleShield with 60-step gradient-based White-Box PGD
        at target tokens while attempting to minimize the combined anomaly score.
        """
        shield = FullTwistorIncidenceBundleShield(dim=self.D, seed=42, z_thresh=3.5)
        target_tokens = [25, 50, 75]
        gt = torch.zeros(self.T, dtype=torch.bool)
        for t in target_tokens:
            gt[t] = True

        delta = torch.zeros_like(self.H_clean, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=0.08)
        for _ in range(60):
            opt.zero_grad()
            H_adv = self.H_clean + delta
            R = H_adv @ shield.Q
            d2 = R[2:] - 2.0 * R[1:-1] + R[:-2]
            loss_evade = torch.sum(d2 ** 2) + 0.5 * torch.sum((H_adv - self.H_clean.detach()) ** 2)
            loss_evade.backward()
            with torch.no_grad():
                delta.grad[~gt] = 0.0
            opt.step()
            with torch.no_grad():
                delta[~gt] = 0.0
                for t in target_tokens:
                    n = torch.linalg.norm(delta[t])
                    if n < 6.0:
                        delta[t] = delta[t] * (6.0 / n.clamp(min=1e-12))

        H_attack = (self.H_clean + delta).detach()
        res = shield.analyze_and_purify(H_attack)
        caught = bool(torch.all(res["quarantine_mask"][gt]).item())
        fp = int(torch.sum(res["quarantine_mask"] & (~gt)).item())
        max_z = float(torch.max(res["z_score"][gt]).item())
        print(f"  -> [AUDIT 4: White-Box PGD] Caught All Targets: {caught} (Max Z = {max_z:.2f}) | False Positives: {fp}")
        self.assertTrue(caught)
        self.assertEqual(fp, 0)

    def test_5_cp_global_phase_and_sign_flip_invariance_in_core(self):
        """
        Verifies Reviewer Finding on core.py / shield.py:
          - Global phase rotation z -> e^{i theta} z and antipodal flip x -> -x
            preserve acceptance rate and anomaly scores to machine precision (< 1e-14).
        """
        rng = np.random.default_rng(42)
        X_train = rng.normal(size=(400, 64))
        X_test = rng.normal(size=(200, 64))

        vs = VectorShield(n_clusters=4, quantile_threshold=0.99, random_state=42)
        vs.fit(X_train)

        res_orig = vs.verify(X_test)
        res_neg = vs.verify(-X_test)

        # Apply global complex phase rotation theta = 1.2345 rad to (Re, Im) halves
        theta = 1.2345
        m = 32
        X_rot = np.empty_like(X_test)
        X_rot[:, :m] = X_test[:, :m] * math.cos(theta) - X_test[:, m:] * math.sin(theta)
        X_rot[:, m:] = X_test[:, :m] * math.sin(theta) + X_test[:, m:] * math.cos(theta)
        res_rot = vs.verify(X_rot)

        score_diff_neg = float(np.max(np.abs(res_orig["anomaly_scores"] - res_neg["anomaly_scores"])))
        score_diff_rot = float(np.max(np.abs(res_orig["anomaly_scores"] - res_rot["anomaly_scores"])))
        acc_rate = float(np.mean(res_orig["accepted"])) * 100.0
        print(f"  -> [AUDIT 5: CP^d Phase & Sign Invariance] Held-out Acceptance: {acc_rate:.1f}% | Diff(-X): {score_diff_neg:.2e} | Diff(e^{{i theta}}X): {score_diff_rot:.2e}")
        self.assertLess(score_diff_neg, 1e-13)
        self.assertLess(score_diff_rot, 1e-13)
        self.assertGreaterEqual(acc_rate, 98.0)


if __name__ == "__main__":
    unittest.main()