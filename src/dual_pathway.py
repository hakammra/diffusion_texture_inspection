"""Dual-Pathway Anomaly Detection and Morphological Disambiguation Engine.

Developed for unsupervised industrial fabric and texture defect inspection.
Combines:
  1. Pathway A (High-Frequency DiT Diffusion Residual): Concentrated chemical droplets,
     oil stains, and yarn snags.
  2. Pathway B (Low-Frequency Photometric Bandpass Field): Weave-suppressed spatial
     intensity variations for broad diffuse watermarks and liquid puddles.
  3. Morphological and PCA Aspect-Ratio Disambiguation: Separates high-aspect structural
     creases and wrinkles from organic stain bodies.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes


def photometric_weave_suppression(
    rgb_image: np.ndarray,
    weave_radius: int = 4,
    macro_bg_radius: int = 35,
    margin_blackout: int = 20,
) -> np.ndarray:
    """Computes a weave-suppressed photometric watermark field.

    Applies a small Gaussian blur (radius=4) to remove high-frequency yarn weave,
    followed by a large Gaussian blur (radius=35) to model macro-illumination.
    The difference isolates broad faint liquid saturation puddles.
    A hard perimeter blackout prevents edge tension and reflection fringe artifacts.

    Args:
        rgb_image: (H, W, 3) uint8 image array.
        weave_radius: Radius for textile weave texture suppression.
        macro_bg_radius: Radius for macro-illumination background field.
        margin_blackout: Pixel margin at image perimeter forced to 0.

    Returns:
        (H, W) float32 photometric anomaly field.
    """
    im_pil = Image.fromarray(rgb_image).convert("L")
    sm = im_pil.filter(ImageFilter.GaussianBlur(radius=weave_radius))
    bg = sm.filter(ImageFilter.GaussianBlur(radius=macro_bg_radius))

    diff = np.maximum(np.array(bg, dtype=np.float32) - np.array(sm, dtype=np.float32), 0.0)

    if margin_blackout > 0:
        diff[:margin_blackout, :] = 0.0
        diff[-margin_blackout:, :] = 0.0
        diff[:, :margin_blackout] = 0.0
        diff[:, -margin_blackout:] = 0.0

    return diff


def analyze_defect_clusters(
    dit_residual_map: np.ndarray,
    photometric_map: np.ndarray,
    tau_strict: float,
    tau_high: float = 13.5,
    tau_low: float = 3.3,
    tau_watermark: float = 5.8,
    min_area_dit: int = 25,
    min_area_watermark: int = 260,
    crease_max_intensity: float = 35.0,
    crease_max_mean: float = 15.0,
    aspect_ratio_thresh: float = 3.0,
    linearity_thresh: float = 0.88,
) -> tuple[bool, bool, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Analyzes and disambiguates defect clusters across both detection pathways.

    Routes high-aspect, highly linear clusters to `crease_mask` (wrinkles/folds),
    while preserving organic stains in `stain_mask`.

    Returns:
        (has_stain, has_crease, stain_mask, crease_mask, total_defect_mask, cluster_details)
    """
    H, W = dit_residual_map.shape
    stain_mask = np.zeros((H, W), dtype=bool)
    crease_mask = np.zeros((H, W), dtype=bool)
    cluster_details = []

    # --- Pathway A: DiT Diffusion Residual Stream ---
    seed_a = dit_residual_map > tau_high
    body_a = dit_residual_map > tau_low

    if np.any(body_a):
        visited_a = np.zeros((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                if body_a[r, c] and not visited_a[r, c]:
                    q = [(r, c)]
                    visited_a[r, c] = True
                    cluster = []
                    has_seed = False
                    while q:
                        cr, cc = q.pop()
                        cluster.append((cr, cc))
                        if seed_a[cr, cc]:
                            has_seed = True
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W and body_a[nr, nc] and not visited_a[nr, nc]:
                                visited_a[nr, nc] = True
                                q.append((nr, nc))

                    coords = np.array(cluster)
                    min_r, min_c = coords.min(axis=0)
                    max_r, max_c = coords.max(axis=0)
                    is_border = (min_c < 15 or min_r < 15 or max_c > W - 15 or max_r > H - 15)

                    vals = [dit_residual_map[cr, cc] for cr, cc in cluster]
                    max_val = float(np.max(vals))
                    mean_val = float(np.mean(vals))

                    # Border protection: outer border seeds must cross tau_strict
                    valid = has_seed and (len(cluster) >= min_area_dit)
                    if is_border and max_val < tau_strict:
                        valid = False

                    if valid:
                        cov = np.cov(coords, rowvar=False)
                        eigvals = np.sort(np.linalg.eigvalsh(cov))
                        total_var = max(eigvals[0] + eigvals[1], 1e-6)
                        exp_linear = float(eigvals[1] / total_var)
                        axis_ratio = float(np.sqrt(max(eigvals[1], 1e-4) / max(eigvals[0], 1e-4)))
                        span = max(max_r - min_r, max_c - min_c)

                        is_intense = (max_val > crease_max_intensity) or (mean_val > crease_max_mean)
                        if is_intense:
                            is_crease = False
                        else:
                            is_crease = (axis_ratio >= 3.5 and span >= 50) or (exp_linear >= 0.92 and span >= 45)

                        cluster_details.append({
                            "pathway": "DiT",
                            "box": [int(min_c), int(min_r), int(max_c), int(max_r)],
                            "area": len(cluster),
                            "explained_linear": round(exp_linear, 3),
                            "axis_ratio": round(axis_ratio, 2),
                            "is_crease": is_crease,
                            "max_score": round(max_val, 2),
                        })

                        for cr, cc in cluster:
                            if is_crease:
                                crease_mask[cr, cc] = True
                            else:
                                stain_mask[cr, cc] = True

    # --- Pathway B: Weave-Suppressed Photometric Watermark Stream ---
    binary_b = photometric_map > tau_watermark
    if np.any(binary_b):
        visited_b = np.zeros((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                if binary_b[r, c] and not visited_b[r, c]:
                    q = [(r, c)]
                    visited_b[r, c] = True
                    cluster = []
                    while q:
                        cr, cc = q.pop()
                        cluster.append((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W and binary_b[nr, nc] and not visited_b[nr, nc]:
                                visited_b[nr, nc] = True
                                q.append((nr, nc))

                    if len(cluster) >= min_area_watermark:
                        coords = np.array(cluster)
                        min_r, min_c = coords.min(axis=0)
                        max_r, max_c = coords.max(axis=0)
                        cov = np.cov(coords, rowvar=False)
                        eigvals = np.sort(np.linalg.eigvalsh(cov))
                        total_var = max(eigvals[0] + eigvals[1], 1e-6)
                        exp_linear = float(eigvals[1] / total_var)
                        axis_ratio = float(np.sqrt(max(eigvals[1], 1e-4) / max(eigvals[0], 1e-4)))

                        is_crease = (axis_ratio >= aspect_ratio_thresh and exp_linear >= linearity_thresh)
                        max_score = float(np.max([photometric_map[cr, cc] for cr, cc in cluster]))

                        cluster_details.append({
                            "pathway": "Photometric",
                            "box": [int(min_c), int(min_r), int(max_c), int(max_r)],
                            "area": len(cluster),
                            "explained_linear": round(exp_linear, 3),
                            "axis_ratio": round(axis_ratio, 2),
                            "is_crease": is_crease,
                            "max_score": round(max_score, 2),
                        })

                        for cr, cc in cluster:
                            if is_crease:
                                crease_mask[cr, cc] = True
                            else:
                                stain_mask[cr, cc] = True

    # Morphological consolidation on organic stains
    if np.any(stain_mask):
        dilated = binary_dilation(stain_mask, iterations=2)
        eroded = binary_erosion(dilated, iterations=2)
        stain_mask = binary_fill_holes(eroded)

    has_stain = bool(np.any(stain_mask))
    has_crease = bool(np.any(crease_mask))
    total_defect = stain_mask | crease_mask
    return has_stain, has_crease, stain_mask, crease_mask, total_defect, cluster_details
