import math
import torch
import numpy as np
from PIL import Image

def generate_4k_twistor_wallpaper(out_path: str = "penrose_cp3_4k_wallpaper.png", width: int = 3840, height: int = 2160):
    device = torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.device("cpu")
    print("="*92)
    print(f"[ARGON 4K WALLPAPER ENGINE v2] Rendering High-Contrast Penrose Robinson Congruence on {device}")
    print("="*92)

    aspect = float(width) / float(height)
    y_lin = torch.linspace(-1.45, 1.45, height, device=device)
    x_lin = torch.linspace(-1.45 * aspect, 1.45 * aspect, width, device=device)

    # Rotate the 3D stereographic projection of the Robinson congruence via SU(2) Lie algebra
    # Tilt angle reveals both the nested Clifford tori and the central twistor null-cone vortex
    tilt_x, tilt_y = 0.62, -0.48
    cx, sx = math.cos(tilt_x), math.sin(tilt_x)
    cy, sy = math.cos(tilt_y), math.sin(tilt_y)

    chunk_h = 270
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    with torch.no_grad():
        for y0 in range(0, height, chunk_h):
            y1 = min(y0 + chunk_h, height)
            gy, gx = torch.meshgrid(y_lin[y0:y1], x_lin, indexing="ij")

            # Project 2D screen plane into 3D stereographic S^3 fiber slice + depth slices
            # Evaluate 7 depth layers along the null congruence (Bufferless projective ray integration)
            accum_cyan = torch.zeros_like(gx)
            accum_gold = torch.zeros_like(gx)
            accum_magenta = torch.zeros_like(gx)

            for z_slice in [-0.75, -0.45, -0.20, 0.0, 0.20, 0.45, 0.75]:
                gz = torch.full_like(gx, z_slice)

                # Apply SU(2)/SO(3) frame rotation
                rx = gx * cy + gz * sy
                ry = gx * sx * sy + gy * cx - gz * sx * cy
                rz = -gx * cx * sy + gy * sx + gz * cx * cy

                r2 = rx**2 + ry**2 + rz**2
                denom = 1.0 + r2

                # Inverse stereographic map R^3 -> S^3 subset C^2 (Unit Spinor Z = (Z0, Z1) in CP^3)
                z0_re = (1.0 - r2) / denom
                z0_im = (2.0 * rz) / denom
                z1_re = (2.0 * rx) / denom
                z1_im = (2.0 * ry) / denom

                # Hopf fibration / Penrose Robinson Congruence coordinates:
                # Base CP^1 sphere radial angle eta in [0, pi/2], fiber phases xi_1, xi_2
                abs_z0 = torch.sqrt(z0_re**2 + z0_im**2 + 1e-12)
                abs_z1 = torch.sqrt(z1_re**2 + z1_im**2 + 1e-12)
                eta = torch.atan2(abs_z1, abs_z0)  # Torus shell parameter

                phi_0 = torch.atan2(z0_im, z0_re)
                phi_1 = torch.atan2(z1_im, z1_re)

                # Smooth out branch-cut singularities near abs_z0 -> 0 or abs_z1 -> 0
                torus_weight = 4.0 * (abs_z0**2) * (abs_z1**2)  # Peaks on the Clifford torus (eta = pi/4)

                # High-frequency harmonic standing waves along twisted Villarceau circles
                # Winding numbers (p=6, q=6) for linked Penrose twistor fibers
                twist_phase_pos = 8.0 * (phi_1 - phi_0)
                twist_phase_neg = 6.0 * (phi_1 + phi_0)
                shell_ribbons = torch.cos(18.0 * eta)

                # Sharp filament ridges (raised to high power so void remains dark between fibers!)
                fil_1 = torch.pow(0.5 + 0.5 * torch.cos(twist_phase_pos + 4.0 * z_slice), 14.0)
                fil_2 = torch.pow(0.5 + 0.5 * torch.cos(twist_phase_neg - 3.0 * z_slice), 16.0)
                shell_mask = torch.pow(0.5 + 0.5 * shell_ribbons, 8.0)

                # Hermitian Null-Cone Caustic Ring (det(X^{AA'}) = 0 light-cone horizon)
                null_ring = torch.exp(-28.0 * (r2 - 0.85)**2) * torch.pow(0.5 + 0.5 * torch.cos(12.0 * phi_1), 6.0)

                layer_atten = math.exp(-1.8 * (z_slice**2))
                accum_cyan += layer_atten * torus_weight * shell_mask * fil_1
                accum_gold += layer_atten * (torus_weight * shell_mask * fil_2 + 0.85 * null_ring)
                accum_magenta += layer_atten * torus_weight * (fil_1 * fil_2 * 2.2 + 0.35 * null_ring)

            # Normalize integrated congruence energy
            c_cyan = accum_cyan / 2.2
            c_gold = accum_gold / 2.2
            c_mag  = accum_magenta / 2.2

            # Deep obsidian space background (#04060E) with subtle central horizon glow
            rad_sq = (gx / aspect)**2 + gy**2
            bg_glow = torch.exp(-1.6 * rad_sq)

            r_ch = 0.015 + 0.04 * bg_glow + 1.35 * c_gold + 1.10 * c_mag + 0.12 * c_cyan
            g_ch = 0.022 + 0.07 * bg_glow + 1.15 * c_cyan + 0.88 * c_gold + 0.18 * c_mag
            b_ch = 0.050 + 0.14 * bg_glow + 1.55 * c_cyan + 0.25 * c_gold + 0.95 * c_mag

            rgb = torch.stack([r_ch, g_ch, b_ch], dim=-1)
            # High-contrast ACES filmic curve with toe contrast
            rgb = torch.clamp(rgb, 0.0, None)
            rgb = (rgb * (2.51 * rgb + 0.03)) / (rgb * (2.43 * rgb + 0.59) + 0.14)
            rgb = torch.pow(torch.clamp(rgb, 0.0, 1.0), 1.12)  # Gamma punch for deep blacks

            canvas[y0:y1] = (rgb.cpu().numpy() * 255.0).astype(np.uint8)

    Image.fromarray(canvas).save(out_path, optimize=True)
    jpg_path = out_path.replace(".png", ".jpg")
    Image.fromarray(canvas).save(jpg_path, quality=96)
    print(f"  -> [SUCCESS] High-Contrast 4K Wallpaper saved to: {out_path} & {jpg_path}")
    print("="*92)

if __name__ == "__main__":
    generate_4k_twistor_wallpaper()