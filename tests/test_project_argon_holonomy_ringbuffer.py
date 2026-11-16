import math
import unittest
import torch
from metacognition_shell import PalatialCP4StandingWaveEngine, PalatialRingBuffer


class TestProjectArgonHolonomyAndRingBuffer(unittest.TestCase):
    """Formal unit tests for Project Argon SIGMA_5, Berry Holonomy, and O(1) RingBuffer."""

    def test_1_sigma5_real_float64_and_su23_null_incidence_bracket(self):
        """Verifies SIGMA_5 is strictly float64 and vanishes on equatorial null-incidence spinors."""
        self.assertEqual(PalatialCP4StandingWaveEngine.SIGMA_5.dtype, torch.float64)
        self.assertFalse(PalatialCP4StandingWaveEngine.SIGMA_5.is_complex())

        # Construct equatorial state where omega = i * pi and Z^4 = 0 -> <Z | SIGMA_5 | Z> == 0
        pi = torch.tensor([[1.0 + 0.0j, 0.0 + 0.0j]], dtype=torch.complex128)
        omega = 1j * pi
        z4 = torch.zeros((1, 1), dtype=torch.complex128)
        Z_null = torch.cat([omega, pi, z4], dim=-1) / math.sqrt(2.0)

        bracket = PalatialCP4StandingWaveEngine.evaluate_su23_inner_product(Z_null, Z_null)
        err = float(torch.max(torch.abs(bracket)).item())
        self.assertLess(err, 1e-15)
        print(f"  -> [ARGON SHIELD 1] SIGMA_5 dtype: {PalatialCP4StandingWaveEngine.SIGMA_5.dtype} | Null Bracket |<Z|SIGMA_5|Z>|: {err:.2e}")

    def test_2_sliding_berry_holonomy_u1_invariance_and_torsion_sensitivity(self):
        """Verifies discrete Wilson loop Berry holonomy is U(1) gauge-invariant and spikes under torsion."""
        gen = torch.Generator(device="cpu").manual_seed(999)
        t = torch.linspace(0.0, 2.0 * math.pi, 64, dtype=torch.float64).unsqueeze(1)
        H_clean = torch.sin(t + torch.arange(63, dtype=torch.float64).unsqueeze(0) * 0.1)

        res_clean = PalatialCP4StandingWaveEngine.evaluate_cp4_standing_waves(H_clean)
        Z5 = res_clean["Z5_hat"]

        # Apply arbitrary local U(1) phase transformations
        phases = torch.exp(1j * (2.0 * math.pi * torch.rand(Z5.shape[:-1], dtype=torch.float64, generator=gen))).unsqueeze(-1)
        flux_clean = PalatialCP4StandingWaveEngine.compute_sliding_berry_holonomy(Z5)
        flux_gauged = PalatialCP4StandingWaveEngine.compute_sliding_berry_holonomy(Z5 * phases)
        gauge_err = float(torch.max(torch.abs(flux_clean - flux_gauged)).item())
        self.assertLess(gauge_err, 1e-15)

        # Inject complex phase-torsion twist into Z5 and verify Berry holonomy spike
        Z5_twisted = Z5.clone()
        Z5_twisted[30, :, 0] *= torch.exp(1j * torch.tensor(1.25, dtype=torch.float64))
        Z5_twisted[30, :, 4] += 0.65 + 0.45j
        Z5_twisted = Z5_twisted / torch.linalg.norm(Z5_twisted, dim=-1, keepdim=True)
        flux_twisted = PalatialCP4StandingWaveEngine.compute_sliding_berry_holonomy(Z5_twisted)

        self.assertGreater(float(torch.max(flux_twisted).item()), float(torch.max(flux_clean).item()) * 1.5)
        print(f"  -> [ARGON SHIELD 2] Berry Holonomy U(1) Gauge Diff: {gauge_err:.2e} | Peak Flux: {float(torch.max(flux_clean)):.4f} -> {float(torch.max(flux_twisted)):.4f}")

    def test_3_o1_palatial_ring_buffer_streaming_completeness(self):
        """Verifies fixed-size O(1) PalatialRingBuffer maintains 1e-16 triad completeness across wrap-arounds."""
        buf = PalatialRingBuffer(capacity=8, num_fibers=21)
        gen = torch.Generator(device="cpu").manual_seed(1234)
        max_err = 0.0
        for step in range(25):
            z_real = torch.randn((21, 5), dtype=torch.float64, generator=gen)
            z_imag = torch.randn((21, 5), dtype=torch.float64, generator=gen)
            out = buf.push(z_real + 1j * z_imag, gamma=0.25)
            max_err = max(max_err, abs(out["triad_sum"] - 1.0))

        self.assertEqual(buf.count, 8)
        self.assertLess(max_err, 1e-15)
        print(f"  -> [ARGON SHIELD 3] O(1) RingBuffer Wrap-Around (25 steps, cap=8) | Max Triad Err: {max_err:.2e}")


if __name__ == "__main__":
    unittest.main()
