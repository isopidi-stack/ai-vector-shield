"""
test_shield_stress.py
=====================
Stress / generalisation harness for FullTwistorIncidenceBundleShield (v5.5.2).

Why this exists
---------------
The built-in benchmark uses ONE clean signal, ONE attack direction and ONE magnitude (||delta|| = 8).
This harness sweeps what that benchmark does not: attack magnitude, block length, persistent shifts,
ramps, long smooth bumps, attacks at the sequence edges, measurement noise, and the older (v5.3) clean
signal -- and compares against a twistor-free median-filter baseline and two ablations.

Setup
-----
Put this file next to your shield sources, using the names the imports expect:
    topological_poison_shield.py   (the v5.5.2 file)
    casimir_harmonics.py           (SpinorHarmonicCasimirEngine)
    metacognition_shell.py         (MetacognitiveTwistorShell)
Requires: numpy (always), torch (only for the shield detector).

Usage
-----
    python test_shield_stress.py                      # print the full report (never fails)
    python test_shield_stress.py --seeds 20           # more seeds
    python test_shield_stress.py --only block,persist # substring filter on scenario names
    python test_shield_stress.py --json report.json   # also dump raw numbers
    python -m unittest test_shield_stress -v          # assertions (these CAN fail -- that is the point)

What the detectors are
----------------------
    shield          full v5.5.2 quarantine mask (impulse OR block/bump)
    shield_tube     only the reference-manifold "tube residual" path: (tube_res > 1.25) & (z_tube > z)
    shield_zspike   only z_spike > z (fiber curvature / shear / Casimir mix). NOTE: this ignores the
                    local-spike gate of the real impulse flag (not exposed in the result dict), so it is
                    an approximation of the "twistor feature" path.
    median_baseline robust z-score of ||H_t - median_filter_19(H)_t||. No Q, no twistors.

Tolerances for the unit tests are the constants right below -- edit them to taste.
"""
import argparse
import json
import math
import sys
import unittest

import numpy as np

# ----------------------------------------------------------------------------------------------
# Tunables for the unit-test assertions
# ----------------------------------------------------------------------------------------------
MIN_RECALL = 0.90          # mean recall over seeds required in MUST_DETECT scenarios
MAX_FP_PER_RUN = 1.0       # mean false-positive tokens per run allowed
MIN_SUPPRESSION = 50.0     # % reduction of mean poisoned-token error after purification
DEFAULT_SEEDS = 10
T, D = 128, 63


# ----------------------------------------------------------------------------------------------
# Signals and attacks (pure numpy, deterministic per seed)
# ----------------------------------------------------------------------------------------------
def clean_signal(kind="v552", noise=0.0, seed=0):
    tg = np.linspace(0.0, 4.0 * math.pi, T)[:, None]
    f = np.arange(1, D + 1, dtype=float)[None]
    if kind == "v552":      # signal used by the v5.5.2 benchmark
        H = np.sin(tg * (0.25 + 0.03 * f)) + 0.5 * np.cos(tg * (0.15 + 0.02 * f))
    elif kind == "v53":     # higher-frequency signal used by the v5.3 benchmark
        H = np.sin(tg * f * 0.08) + 0.5 * np.cos(tg * f * 0.08 * 0.5)
    else:
        raise ValueError(f"unknown clean signal '{kind}'")
    if noise > 0.0:
        H = H + np.random.default_rng(900 + seed).normal(size=H.shape) * noise
    return H


def make_attack(Hc, seed, kind="orig", scale=1.0, blocklen=6, start=55):
    """Return (H_poisoned, ground_truth_mask). Every attack uses a unit direction u; magnitude 8*scale."""
    g = np.random.default_rng(seed)
    u = g.standard_normal(D)
    u /= np.linalg.norm(u)
    Hp, gt = Hc.copy(), np.zeros(T, dtype=bool)
    amp = 8.0 * scale
    if kind == "orig":  # the three attacks of the built-in benchmark
        for i in range(20, 26):                       # alternating-sign isolated spikes
            Hp[i] += (1 if i % 2 == 0 else -1) * amp * u
            gt[i] = True
        for i in range(55, 61):                       # constant-offset block (same delta)
            Hp[i] += amp * u
            gt[i] = True
        L = 8
        for k, i in enumerate(range(90, 98)):         # smooth cosine bump
            w = 0.5 * (1.0 - math.cos(2.0 * math.pi * (k + 1) / (L + 1)))
            Hp[i] += (2.0 + 7.5 * w) * scale * u
            gt[i] = True
    elif kind == "block":                             # trigger-style constant offset of length `blocklen`
        for i in range(start, start + blocklen):
            Hp[i] += amp * u
            gt[i] = True
    elif kind == "persist":                           # offset that never returns
        for i in range(start, T):
            Hp[i] += amp * u
            gt[i] = True
    elif kind == "ramp":                              # 20-token linear ramp, then hold 6 tokens
        for k, i in enumerate(range(start, start + 20)):
            Hp[i] += amp * (k + 1) / 20.0 * u
            gt[i] = True
        for i in range(start + 20, start + 26):
            Hp[i] += amp * u
            gt[i] = True
    elif kind == "longbump":                          # 16-token Hann bump
        L = 16
        for k, i in enumerate(range(start, start + L)):
            w = 0.5 * (1.0 - math.cos(2.0 * math.pi * (k + 1) / (L + 1)))
            Hp[i] += amp * w * u
            gt[i] = True
    elif kind == "edge":                              # first 6 and last 6 tokens
        for i in list(range(0, 6)) + list(range(T - 6, T)):
            Hp[i] += amp * u
            gt[i] = True
    elif kind == "randdir_block":                     # same sign, independent direction per token
        for i in range(start, start + blocklen):
            v = g.standard_normal(D)
            v /= np.linalg.norm(v)
            Hp[i] += amp * v
            gt[i] = True
    else:
        raise ValueError(f"unknown attack kind '{kind}'")
    return Hp, gt


# name -> kwargs for (clean_signal / make_attack)
SCENARIOS = [
    ("orig_x1 (built-in benchmark)", dict(kind="orig", scale=1.0)),
    ("orig_x2",                      dict(kind="orig", scale=2.0)),
    ("orig_x0.5",                    dict(kind="orig", scale=0.5)),
    ("orig_x0.25",                   dict(kind="orig", scale=0.25)),
    ("orig_x0.125",                  dict(kind="orig", scale=0.125)),
    ("v53_signal",                   dict(kind="orig", clean="v53")),
    ("noise_0.02",                   dict(kind="orig", noise=0.02)),
    ("noise_0.05",                   dict(kind="orig", noise=0.05)),
    ("noise_0.2",                    dict(kind="orig", noise=0.2)),
    ("block_6",                      dict(kind="block", blocklen=6)),
    ("block_10",                     dict(kind="block", blocklen=10)),
    ("block_11",                     dict(kind="block", blocklen=11)),
    ("block_12",                     dict(kind="block", blocklen=12)),
    ("block_16",                     dict(kind="block", blocklen=16)),
    ("block_24",                     dict(kind="block", blocklen=24)),
    ("persistent_shift",             dict(kind="persist")),
    ("ramp_20_hold_6",               dict(kind="ramp")),
    ("long_bump_16",                 dict(kind="longbump")),
    ("edge_6_plus_6",                dict(kind="edge")),
    ("randdir_block_6",              dict(kind="randdir_block", blocklen=6)),
]

# Scenarios where an attack with ||delta|| >= ~8 (clean ||H_t|| is ~6.7) should be caught.
MUST_DETECT = [
    "orig_x1 (built-in benchmark)", "orig_x2", "v53_signal", "noise_0.02", "noise_0.05",
    "block_6", "block_10", "block_11", "block_12", "block_16", "block_24",
    "persistent_shift", "ramp_20_hold_6", "long_bump_16", "edge_6_plus_6", "randdir_block_6",
]


# ----------------------------------------------------------------------------------------------
# Detectors.  A detector is  f(H_np, seed) -> dict(masks={name: bool[T]}, purified=np.ndarray|None)
# ----------------------------------------------------------------------------------------------
def _tmed(x):
    """Lower median (matches torch.median)."""
    x = np.sort(np.asarray(x).ravel())
    return x[(len(x) - 1) // 2]


def _robust_z(x):
    m = _tmed(x)
    mad = max(_tmed(np.abs(x - m)), 1e-8)
    return (x - m) / (1.4826 * mad)


def median_baseline_mask(H, window=19, thr=3.5):
    from numpy.lib.stride_tricks import sliding_window_view
    hw = window // 2
    pad = np.concatenate([np.repeat(H[:1], hw, 0), H, np.repeat(H[-1:], hw, 0)], 0)
    med = np.median(sliding_window_view(pad, window, axis=0), axis=-1)
    return _robust_z(np.linalg.norm(H - med, axis=1)) > thr


class MedianBaseline:
    name = "median_baseline"

    def __call__(self, H, seed):
        return dict(masks={"median_baseline": median_baseline_mask(H)}, purified=None)


class ShieldDetector:
    """Adapter around the real torch class FullTwistorIncidenceBundleShield (v5.5.2)."""
    name = "shield"

    def __init__(self, z_thresh=3.5, shield_module="topological_poison_shield"):
        self.z = z_thresh
        self.module = shield_module

    def __call__(self, H, seed):
        import importlib
        import torch
        cls = importlib.import_module(self.module).FullTwistorIncidenceBundleShield
        sh = cls(dim=H.shape[1], seed=42 + seed, z_thresh=self.z)
        res = sh.analyze_and_purify(torch.from_numpy(H).to(torch.float64))
        full = res["quarantine_mask"].detach().cpu().numpy().astype(bool)
        tube = ((res["tube_residual"] > 1.25) & (res["z_tube"] > self.z)).detach().cpu().numpy().astype(bool)
        zsp = (res["z_spike"] > self.z).detach().cpu().numpy().astype(bool)
        pur = res["H_purified"].detach().cpu().numpy()
        return dict(masks={"shield": full, "shield_tube": tube, "shield_zspike": zsp}, purified=pur)


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
def evaluate(detectors, scenario_kwargs, seeds):
    """Return {mask_name: metrics} aggregated over seeds."""
    kw = dict(scenario_kwargs)
    clean_kind = kw.pop("clean", "v552")
    noise = kw.pop("noise", 0.0)
    acc = {}
    for sd in seeds:
        Hc = clean_signal(clean_kind, noise, sd)
        Hp, gt = make_attack(Hc, sd, **kw)
        for det in detectors:
            out = det(Hp, sd)
            for name, mask in out["masks"].items():
                a = acc.setdefault(name, dict(recall=[], fp=[], sup=[], cd=[]))
                a["recall"].append(float((mask & gt).sum() / gt.sum()))
                a["fp"].append(int((mask & ~gt).sum()))
                if name == "shield" and out["purified"] is not None:
                    Hq = out["purified"]
                    eb = np.linalg.norm(Hp[gt] - Hc[gt], axis=1).mean()
                    ea = np.linalg.norm(Hq[gt] - Hc[gt], axis=1).mean()
                    a["sup"].append(100.0 * (1.0 - ea / max(eb, 1e-12)))
                    a["cd"].append(float(np.linalg.norm(Hq[~gt] - Hc[~gt], axis=1).max()))
    res = {}
    for name, a in acc.items():
        res[name] = dict(
            recall=float(np.mean(a["recall"])),
            perfect_seeds=float(np.mean([r == 1.0 for r in a["recall"]])),
            fp_per_run=float(np.mean(a["fp"])),
            suppression=float(np.mean(a["sup"])) if a["sup"] else None,
            clean_dist_max=float(np.mean(a["cd"])) if a["cd"] else None,
        )
    return res


def run_report(detectors, seeds, only=None, json_path=None, out=sys.stdout):
    rows = {}
    names = [n for n, _ in SCENARIOS if (not only or any(o in n for o in only))]
    for name, kw in SCENARIOS:
        if name not in names:
            continue
        rows[name] = evaluate(detectors, kw, seeds)
    det_names = []
    for r in rows.values():
        for k in r:
            if k not in det_names:
                det_names.append(k)
    print(f"\n{len(seeds)} seeds | T={T}, D={D} | recall = mean fraction of poisoned tokens quarantined; "
          f"FP = mean clean tokens quarantined per run (~{T - 24}..{T - 6} clean tokens)\n", file=out)
    for dn in det_names:
        print(f"--- detector: {dn} ---", file=out)
        print(f"{'scenario':32s} {'recall%':>8s} {'allSeeds%':>9s} {'FP/run':>7s} {'suppr%':>8s} {'cleanDist':>10s}", file=out)
        for sc, r in rows.items():
            m = r.get(dn)
            if m is None:
                continue
            sup = "   n/a" if m["suppression"] is None else f"{m['suppression']:8.1f}"
            cd = "       n/a" if m["clean_dist_max"] is None else f"{m['clean_dist_max']:10.2e}"
            print(f"{sc:32s} {100*m['recall']:8.1f} {100*m['perfect_seeds']:9.0f} {m['fp_per_run']:7.2f} {sup:>8s} {cd:>10s}", file=out)
        print(file=out)
    if json_path:
        with open(json_path, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"raw numbers written to {json_path}", file=out)
    return rows


# ----------------------------------------------------------------------------------------------
# Unit tests (assertions).  Failures here mean: the shield does not generalise beyond its benchmark.
# ----------------------------------------------------------------------------------------------
def _have_torch_and_shield():
    try:
        import importlib
        import torch  # noqa: F401
        importlib.import_module("topological_poison_shield")
        return True
    except Exception:
        return False


class HarnessSanity(unittest.TestCase):
    """These never touch torch: they check that the harness itself is sound."""

    def test_ground_truth_counts(self):
        Hc = clean_signal()
        _, gt = make_attack(Hc, 0, kind="orig")
        self.assertEqual(int(gt.sum()), 6 + 6 + 8)
        _, gt = make_attack(Hc, 0, kind="block", blocklen=12)
        self.assertEqual(int(gt.sum()), 12)

    def test_attack_magnitude(self):
        Hc = clean_signal()
        Hp, gt = make_attack(Hc, 0, kind="block", blocklen=6)
        d = np.linalg.norm(Hp[gt] - Hc[gt], axis=1)
        np.testing.assert_allclose(d, 8.0, rtol=1e-12)

    def test_baseline_catches_the_easy_case(self):
        res = evaluate([MedianBaseline()], dict(kind="orig", scale=2.0), range(5))
        self.assertGreater(res["median_baseline"]["recall"], 0.8)


@unittest.skipUnless(_have_torch_and_shield(), "needs torch and topological_poison_shield.py on the path")
class ShieldGeneralisation(unittest.TestCase):
    SEEDS = range(DEFAULT_SEEDS)
    CACHE = {}

    @classmethod
    def _res(cls, name):
        if name not in cls.CACHE:
            kw = dict(SCENARIOS)[name]
            cls.CACHE[name] = evaluate([ShieldDetector()], kw, cls.SEEDS)
        return cls.CACHE[name]

    def test_must_detect_scenarios(self):
        for name in MUST_DETECT:
            with self.subTest(scenario=name):
                m = self._res(name)["shield"]
                self.assertGreaterEqual(m["recall"], MIN_RECALL,
                                        f"{name}: recall {m['recall']:.2f} < {MIN_RECALL}")
                self.assertLessEqual(m["fp_per_run"], MAX_FP_PER_RUN,
                                     f"{name}: {m['fp_per_run']:.2f} false positives per run")

    def test_purification_is_useful(self):
        for name in MUST_DETECT:
            with self.subTest(scenario=name):
                m = self._res(name)["shield"]
                self.assertIsNotNone(m["suppression"])
                self.assertGreaterEqual(m["suppression"], MIN_SUPPRESSION,
                                        f"{name}: suppression {m['suppression']:.1f}% < {MIN_SUPPRESSION}%")

    def test_twistor_features_add_value_over_median_baseline(self):
        """Fails if the median-filter baseline matches the full shield everywhere it is evaluated."""
        better = 0
        for name in MUST_DETECT:
            r = evaluate([ShieldDetector(), MedianBaseline()], dict(SCENARIOS)[name], self.SEEDS)
            if r["shield"]["recall"] > r["median_baseline"]["recall"] + 0.05:
                better += 1
        self.assertGreater(better, len(MUST_DETECT) // 2,
                           "shield beats the Q-free median baseline in at most half of the scenarios")

    def test_tube_path_is_not_the_only_detector(self):
        """Fails if the reference-manifold tube path alone explains the detections (ablation)."""
        gaps = []
        for name in ("orig_x1 (built-in benchmark)", "block_6", "randdir_block_6"):
            r = evaluate([ShieldDetector()], dict(SCENARIOS)[name], self.SEEDS)
            gaps.append(r["shield"]["recall"] - r["shield_tube"]["recall"])
        self.assertGreater(max(gaps), 0.05, "tube-only recall already equals full-shield recall")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--only", type=str, default="", help="comma-separated substrings of scenario names")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--no-shield", action="store_true", help="run the numpy-only baseline detector")
    args = ap.parse_args(argv)
    dets = [MedianBaseline()]
    if not args.no_shield:
        if _have_torch_and_shield():
            dets.insert(0, ShieldDetector())
        else:
            print("[warn] torch or topological_poison_shield.py not importable -> running baseline only",
                  file=sys.stderr)
    only = [s for s in args.only.split(",") if s] or None
    run_report(dets, list(range(args.seeds)), only=only, json_path=args.json)


if __name__ == "__main__":
    main()