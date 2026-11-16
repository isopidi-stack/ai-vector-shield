# AI Vector Shield (`v6.0.0-argon-cp4`)

Geometric adversarial anomaly detection and state-purification framework powered by five-component projective twistor geometry ($\mathbb{CP}^4$), $SU(2,3)$ indefinite Hermitian holonomy, and reference-tube trajectory filtering.

---

## Visual Verification Module
 Companion rendering and mathematical verification suite hosted at [**`twistor-render-core`**](https://github.com/isopidi-stack/twistor-render-core) (`v6.0.0-argon-cp4`).

---

## Core Mathematical Guarantees (`v6.0` Argon Audit)

1. **$SU(2,3)$ & Projective Berry Holonomy (`argon_v6_core.py`):**
   Computes discrete gauge-invariant Wilson loops $W = \prod_{j=0}^{m-1} \langle Z_j, Z_{j+1} \rangle \langle Z_m, Z_0 \rangle$ supporting both the $(3,2)$-signature indefinite form $\Sigma_5$ and Euclidean inner products, with explicit degeneracy guards (`valid_mask`) when adjacent states approach orthogonality or the null cone.
2. **Exact $O(1)$ FIFO Sliding-Window Density Matrix (`PalatialRingBuffer`):**
   Explicitly subtracts outgoing rank-1 state projectors $Z_{t-N} Z_{t-N}^\dagger$ upon circular wrap-around, guaranteeing an exact sliding-window density matrix $\rho_t = \frac{1}{N} \sum_{k=0}^{N-1} Z_{t-k} Z_{t-k}^\dagger$ in $O(1)$ update time alongside optional EWMA mode.
3. **Remainder-Safe 3D Fiber Decomposition:**
   Automatically zero-pads feature dimensions $d \not\equiv 0 \pmod 3$ prior to fiber bundling so boundary dimensions are never truncated.
