import torch
import numpy as np

class FullTwistorIncidenceBundleShield:
    """Production-grade v5.5.3-argon-shield FullTwistorIncidenceBundleShield.
    Cleanly integrates reference-manifold second-order tube curvature with
    robust fiber shear and adaptive interval closure.
    """
    def __init__(self, dim: int, seed: int = 42, z_thresh: float = 3.5):
        self.dim = dim
        self.seed = seed
        self.z_thresh = z_thresh
        torch.manual_seed(seed)

    def analyze_and_purify(self, H: torch.Tensor) -> dict:
        if H.dim() == 1:
            H = H.unsqueeze(0)
        T, D = H.shape
        device = H.device
        H_np = H.detach().cpu().numpy()

        # 1. Reference-manifold tube path (second-order finite differences ||d^2 H||)
        if T > 2:
            d2 = H[2:] - 2.0 * H[1:-1] + H[:-2]
            d2_norm = torch.norm(d2, dim=1)
            d2_pad = torch.cat([d2_norm[:1], d2_norm, d2_norm[-1:]], dim=0)
        else:
            d2_pad = torch.zeros(T, device=device, dtype=H.dtype)

        m_d2 = torch.median(d2_pad)
        mad_d2 = torch.clamp(torch.median(torch.abs(d2_pad - m_d2)), min=1e-6)
        z_tube = (d2_pad - m_d2) / (1.4826 * mad_d2)
        tube_residual = 1.0 + torch.clamp(d2_pad / (torch.clamp(m_d2, min=1e-8)), min=0.0)

        # 2. Fiber shear / velocity gradient path (z_spike)
        if T > 1:
            d1 = H[1:] - H[:-1]
            d1_norm = torch.norm(d1, dim=1)
            d1_pad = torch.cat([d1_norm[:1], d1_norm], dim=0)
        else:
            d1_pad = torch.zeros(T, device=device, dtype=H.dtype)

        m_d1 = torch.median(d1_pad)
        mad_d1 = torch.clamp(torch.median(torch.abs(d1_pad - m_d1)), min=1e-6)
        z_d1 = (d1_pad - m_d1) / (1.4826 * mad_d1)

        z_spike = torch.maximum(z_tube, z_d1 * 1.05)

        # 3. Quarantine mask with robust morphological closure
        base_mask = (z_spike > self.z_thresh) | ((tube_residual > 1.2) & (z_tube > self.z_thresh))
        
        mask_np = base_mask.detach().cpu().numpy().astype(bool)
        if mask_np.any():
            indices = np.where(mask_np)[0]
            filled = mask_np.copy()
            if len(indices) >= 2:
                for i in range(len(indices) - 1):
                    start_i = indices[i]
                    end_i = indices[i+1]
                    if end_i - start_i <= 32:
                        filled[start_i:end_i+1] = True
            quarantine_mask = torch.from_numpy(filled).to(device=device)
        else:
            quarantine_mask = base_mask

        # 4. Precision geometric purification for quarantined tokens
        H_purified = H.clone()
        if quarantine_mask.any() and T > 4:
            q_np = quarantine_mask.detach().cpu().numpy()
            for t_idx in range(T):
                if q_np[t_idx]:
                    start_idx = max(0, t_idx - 5)
                    end_idx = min(T, t_idx + 6)
                    valid_mask = ~q_np[start_idx:end_idx]
                    if valid_mask.any():
                        window_vals = H_np[start_idx:end_idx][valid_mask]
                        H_purified[t_idx] = torch.from_numpy(np.median(window_vals, axis=0)).to(device=device, dtype=H.dtype)

        return {
            "quarantine_mask": quarantine_mask,
            "tube_residual": tube_residual,
            "z_tube": z_tube,
            "z_spike": z_spike,
            "H_purified": H_purified
        }
