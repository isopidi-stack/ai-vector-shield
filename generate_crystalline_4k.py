import math
import torch
import numpy as np
from PIL import Image

def generate_crystalline_4k_wallpaper(out_path: str = "penrose_cp3_4k_crystalline.png", width: int = 3840, height: int = 2160):
    device = torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.device("cpu")
    print("="*92)
    print(f"[ARGON CRYSTALLINE 4K ENGINE] Rendering Razor-Sharp Analytical Hopf Twistor Field on {device}")
    print("="*92)

    aspect = float(width) / float(height)
    y_lin = torch.linspace(-1.35, 1.35, height, device=device)
    x_lin = torch.linspace(-1.35 * aspect, 1.35 * aspect, width, device=device)

    tilt_x, tilt_y = 0.58, -0.44
    cx, sx = math.cos(tilt_x), math.sin(tilt_x)
    cy, sy = math.cos(tilt_y), math.sin(tilt_y)

    chunk_h = 270
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    with torch.no_grad():
        for y0 in range(0, height, chunk_h):
            y1 = min(y0 + chunk_h, height)
            gy, gx = torch.meshgrid(y_lin[y0:y1], x_lin, indexing="ij")

            # Analytical tangent-plane intersection with the primary Clifford Torus (r^2 = 1 in S^3)
            # Uses 3 ultra-tight analytical focal sheets for needle-sharp 4K micro-filaments
            accum_cyan = torch.zeros_like(gx)
            accum_gold = torch.zeros_like(gx)
            accum_magenta = torch.zeros_like(gx)

            for z_slice, weight in [(-0.12, 0.25), (0.0, 0.50), (0.12, 0.25)]:
                gz = torch.full_like(gx, z_slice)
                rx = gx * cy + gz * sy
                ry = gx * sx * sy + gy * cx - gz * sx * cy
                rz = -gx * cx * sy + gy * sx + gz * cx * cy

                r2 = rx**2 + ry**2 + rz**2
                denom = 1.0 + r2

                z0_re = (1.0 - r2) / denom
                z0_im = (2.0 * rz) / denom
                z1_re = (2.0 * rx) / denom
                z1_im = (2.0 * ry) / denom

                abs_z0 = torch.sqrt(z0_re**2 + z0_im**2 + 1e-12)
                abs_z1 = torch.sqrt(z1_re**2 + z1_im**2 + 1e-12)
                eta = torch.atan2(abs_z1, abs_z0)

                phi_0 = torch.atan2(z0_im, z0_re)
                phi_1 = torch.atan2(z1_im, z1_re)

                # Smooth phase damper near the two Hopf axes (abs_z0 -> 0 or abs_z1 -> 0)
                axis_damper = torch.pow(4.0 * (abs_z0**2) * (abs_z1**2), 1.5)

                # Ultra-crisp harmonic standing-wave filaments (exponent 36 for laser-like precision)
                fil_pos = torch.pow(0.5 + 0.5 * torch.cos(10.0 * (phi_1 - phi_0) + 2.0 * eta), 36.0)
                fil_neg = torch.pow(0.5 + 0.5 * torch.cos(8.0 * (phi_1 + phi_0) - 2.0 * eta), 36.0)
                torus_shells = torch.pow(0.5 + 0.5 * torch.cos(24.0 * eta), 14.0)

                # Exact Null-Cone Horizon Caustics
                horizon = torch.exp(-45.0 * (r2 - 1.0)**2) * (0.4 + 0.6 * torch.pow(0.5 + 0.5 * torch.cos(16.0 * phi_1), 10.0))

                accum_cyan += weight * axis_damper * (fil_pos * 1.8 + 0.6 * torus_shells * fil_pos)
                accum_gold += weight * (axis_damper * fil_neg * torus_shells * 2.2 + 1.1 * horizon)
                accum_magenta += weight * axis_damper * (fil_pos * fil_neg * 6.5 + 0.4 * fil_neg)

            rad_sq = (gx / aspect)**2 + gy**2
            bg_glow = torch.exp(-1.8 * rad_sq)

            r_ch = 0.012 + 0.03 * bg_glow + 1.45 * accum_gold + 1.25 * accum_magenta + 0.10 * accum_cyan
            g_ch = 0.018 + 0.05 * bg_glow + 1.25 * accum_cyan + 0.95 * accum_gold + 0.15 * accum_magenta
            b_ch = 0.042 + 0.11 * bg_glow + 1.65 * accum_cyan + 0.20 * accum_gold + 1.05 * accum_magenta

            rgb = torch.stack([r_ch, g_ch, b_ch], dim=-1)
            rgb = torch.clamp(rgb, 0.0, None)
            rgb = (rgb * (2.51 * rgb + 0.03)) / (rgb * (2.43 * rgb + 0.59) + 0.14)
            rgb = torch.pow(torch.clamp(rgb, 0.0, 1.0), 1.15)

            canvas[y0:y1] = (rgb.cpu().numpy() * 255.0).astype(np.uint8)

    Image.fromarray(canvas).save(out_path, optimize=True)
    jpg_path = out_path.replace(".png", ".jpg")
    Image.fromarray(canvas).save(jpg_path, quality=96)
    print(f"  -> [SUCCESS] Crystalline 4K Wallpaper saved to: {out_path} & {jpg_path}")
    print("="*92)

if __name__ == "__main__":
    generate_crystalline_4k_wallpaper()