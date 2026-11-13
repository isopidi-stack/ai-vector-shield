"""
Executive ROI & Latency Benchmark: AI Vector Shield vs. Secondary LLM Guardrails
Run: python examples/roi_benchmark.py
"""
import time
import numpy as np
from ai_vector_shield import VectorShield


def run_roi_benchmark():
    rng = np.random.default_rng(42)
    dim = 384  # Standard enterprise embedding dimension (e.g., BGE-small / MiniLM)
    n_train = 1500
    n_queries = 1000

    # Simulate 3 enterprise knowledge domains (Finance, Legal, Engineering)
    centers = rng.normal(loc=0.0, scale=1.0, size=(3, dim)).astype(np.float32)
    train_emb = np.vstack([
        centers[i] + rng.normal(scale=0.12, size=(n_train // 3, dim)).astype(np.float32)
        for i in range(3)
    ])

    t_fit0 = time.perf_counter()
    shield = VectorShield(latent_rank=12, n_clusters=3, quantile=0.99, mode="enforce").fit(train_emb)
    fit_ms = (time.perf_counter() - t_fit0) * 1000.0

    # 800 legitimate in-domain queries + 200 OOD / injection vectors
    inliers = centers[0] + rng.normal(scale=0.12, size=(800, dim)).astype(np.float32)
    outliers = rng.normal(loc=0.0, scale=1.5, size=(200, dim)).astype(np.float32)
    queries = np.vstack([inliers, outliers])

    # Warmup
    _ = shield.inspect(queries[:10])

    # Timed batch inspection of 1,000 queries
    t0 = time.perf_counter()
    results = shield.inspect(queries)
    total_batch_ms = (time.perf_counter() - t0) * 1000.0

    # Timed single-query p50/p99 latency (simulating real-time streaming API calls)
    single_latencies = []
    for i in range(200):
        ts = time.perf_counter()
        _ = shield.verify(queries[i])
        single_latencies.append((time.perf_counter() - ts) * 1000.0)

    p50_ms = float(np.percentile(single_latencies, 50))
    p99_ms = float(np.percentile(single_latencies, 99))

    inlier_pass_rate = sum(r["allowed"] for r in results[:800]) / 800.0 * 100.0
    outlier_block_rate = sum(not r["allowed"] for r in results[800:]) / 200.0 * 100.0

    print("=" * 78)
    print(" AI VECTOR SHIELD — ENTERPRISE LATENCY & ROI BENCHMARK (1,000 RAG QUERIES)")
    print("=" * 78)
    print(f" Corpus Fit Time ({n_train} docs, {dim}d)      : {fit_ms:.1f} ms")
    print(f" 1,000-Query Batch Inspection Time      : {total_batch_ms:.2f} ms ({total_batch_ms/1000.0:.4f} ms/query)")
    print(f" Real-Time Single-Query Latency (p50)   : {p50_ms:.3f} ms")
    print(f" Real-Time Single-Query Latency (p99)   : {p99_ms:.3f} ms")
    print(f" Legitimate Query Pass Rate             : {inlier_pass_rate:.1f}%")
    print(f" OOD / Injection Block Rate             : {outlier_block_rate:.1f}%")
    print("-" * 78)
    print(" COMMERCIAL COMPARISON (AT 10 MILLION QUERIES / MONTH):")
    print("-" * 78)
    print(" Guardrail Architecture        | p50 Latency | Dedicated GPU Needed | Est. Monthly Infra")
    print(" ------------------------------+-------------+----------------------+-------------------")
    print(" 8B LLM Guard (Llama-Guard-3)  | ~180.00 ms  | 2x NVIDIA L4 / A10G  | ~$1,400 - $2,200/mo")
    print(" Cross-Encoder Prompt Classifier|  ~28.00 ms  | 1x NVIDIA T4 / L4    |   ~$450 - $750/mo")
    print(f" AI Vector Shield (CP^d CPU)   |   ~{p50_ms:0.2f} ms  | None (Runs on CPU)   |     $0 extra GPU")
    print("=" * 78)


if __name__ == "__main__":
    run_roi_benchmark()
