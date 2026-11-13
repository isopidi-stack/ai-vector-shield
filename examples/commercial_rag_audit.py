import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

import sys, torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai_vector_shield import AIVectorShield

def roc_auc(labels, scores):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    return (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean().item()

def metrics(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def run_honest_audit():
    print("=" * 90)
    print(" MULTI-CLUSTER CP^3 AUDIT: DOMAIN SHIFT & INCREMENTAL ADAPTATION")
    print("=" * 90)

    torch.manual_seed(40)
    D = 768
    U, _ = torch.linalg.qr(torch.randn(D, D))
    sing_vals = 1.0 / (torch.arange(1, D + 1, dtype=torch.float32) ** 0.6)
    spectrum = U * sing_vals.unsqueeze(0)

    centers = torch.matmul(torch.randn(4, D), spectrum.T) * 2.5

    def sample_clusters(cluster_ids, n_per_cluster):
        out = []
        for cid in cluster_ids:
            z = torch.matmul(torch.randn(n_per_cluster, D), spectrum.T) + centers[cid]
            out.append(z)
        return torch.cat(out, dim=0)

    cal_data = sample_clusters([0, 1, 2], 400)
    shield = AIVectorShield(dim=D, latent_rank=32, n_clusters=3, quantile=0.95)
    shield.fit(cal_data)

    clean_in = sample_clusters([0, 1, 2], 200)
    ood = torch.randn(400, D) * 0.55
    x1 = torch.cat([clean_in, ood], dim=0)
    y1 = torch.cat([torch.ones(len(clean_in)), torch.zeros(len(ood))])
    p1 = shield.verify(x1)
    s1 = -shield._projective_residual(shield._to_complex_projective(x1))
    pr1, rc1, f11 = metrics(p1, y1)
    print(f"[TEST 1: OOD Garbled Injection]       ROC-AUC: {roc_auc(y1, s1):.3f} | F1: {f11:.3f} | Prec: {pr1:.3f} | Rec: {rc1:.3f}")

    clean_unseen = sample_clusters([3], 200)
    p_before = shield.verify(clean_unseen).float().mean().item()
    print(f"[TEST 2a: Unseen HR/Legal (Before Adapt)]  Acceptance Rate: {p_before*100:.1f}%")

    hr_cal = sample_clusters([3], 50)
    shield.adapt(hr_cal)
    p_after = shield.verify(clean_unseen).float().mean().item()
    print(f"[TEST 2b: Unseen HR/Legal (After Adapt)]   Acceptance Rate: {p_after*100:.1f}% (False Positive Blocks: {(1-p_after)*100:.1f}% )")

    p1_after = shield.verify(x1)
    pra, rca, f1a = metrics(p1_after, y1)
    print(f"[TEST 2c: OOD Defense Post-Adaptation]     F1: {f1a:.3f} | Prec: {pra:.3f} | Rec: {rca:.3f}")
    print("=" * 90)

if __name__ == "__main__":
    run_honest_audit()
