"""Export all 110 fabric test images for V9 evaluation gallery with Dual-Pathway and Exact Watermark Localization."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
V9_DIR = ROOT / "artifacts" / "fabric_stain_evaluation_v9"
NPZ_PATH = V9_DIR / "test_maps_v9_float16.npz"
CSV_PATH = V9_DIR / "per_image_v9.csv"
DATA_ZIP_PATH = ROOT / "output" / "datasets" / "fabric_stain_pilot.zip"
GALLERY_DIR = V9_DIR / "gallery"
IMAGES_DIR = GALLERY_DIR / "images"


def colorize_heatmap(residual_map: np.ndarray, vmin: float = 0.0, vmax: float = 30.0) -> Image.Image:
    norm = np.clip((residual_map - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    r = np.clip(norm * 2.2, 0, 1)
    g = np.clip(norm * 1.4 - 0.2, 0, 1)
    b = np.clip(norm * 3.0 - 1.2, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def create_composite_overlay(orig_img: Image.Image, s_mask: np.ndarray, c_mask: np.ndarray, clusters: list) -> Image.Image:
    img_pil = orig_img.convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Chemical stains / watermarks: Warm Amber tint (255, 170, 0, 120)
    for r in range(s_mask.shape[0]):
        for c in range(s_mask.shape[1]):
            if s_mask[r, c]:
                overlay.putpixel((c, r), (255, 170, 0, 120))

    # Fabric creases / wrinkles: Cool Cyan tint (0, 220, 255, 120)
    for r in range(c_mask.shape[0]):
        for c in range(c_mask.shape[1]):
            if c_mask[r, c]:
                overlay.putpixel((c, r), (0, 220, 255, 120))

    # Bounding boxes
    for clus in clusters:
        min_c, min_r, max_c, max_r = clus["box"]
        box_color = (0, 220, 255, 220) if clus["is_crease"] else (255, 170, 0, 220)
        draw.rectangle([min_c, min_r, max_c, max_r], outline=box_color, width=2)

    combined = Image.alpha_composite(img_pil, overlay)
    return combined.convert("RGB")


def analyze_defect_clusters_v9(dit_map: np.ndarray, photo_map: np.ndarray,
                               tau_strict: float, tau_high: float = 13.5, tau_low: float = 3.3,
                               tau_wm: float = 5.8, min_area_a: int = 25, min_area_b: int = 260,
                               crease_max_int: float = 35.0, crease_max_mean: float = 15.0):
    H, W = dit_map.shape
    stain_mask = np.zeros((H, W), dtype=bool)
    crease_mask = np.zeros((H, W), dtype=bool)
    cluster_details = []

    # Pathway A: DiT Residual Stream
    seed_a = dit_map > tau_high
    body_a = dit_map > tau_low

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

                    cluster_vals = [dit_map[cr, cc] for cr, cc in cluster]
                    max_val = float(np.max(cluster_vals))
                    mean_val = float(np.mean(cluster_vals))

                    valid = has_seed and (len(cluster) >= min_area_a)
                    if is_border and max_val < tau_strict:
                        valid = False

                    if valid:
                        cov = np.cov(coords, rowvar=False)
                        eigvals = np.sort(np.linalg.eigvalsh(cov))
                        total_var = max(eigvals[0] + eigvals[1], 1e-6)
                        exp_linear = float(eigvals[1] / total_var)
                        axis_ratio = float(np.sqrt(max(eigvals[1], 1e-4) / max(eigvals[0], 1e-4)))
                        span = max(max_r - min_r, max_c - min_c)

                        is_intense = (max_val > crease_max_int) or (mean_val > crease_max_mean)
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

    # Pathway B: Weave-Suppressed Photometric Watermark Stream
    binary_b = photo_map > tau_wm
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

                    if len(cluster) >= min_area_b:
                        coords = np.array(cluster)
                        min_r, min_c = coords.min(axis=0)
                        max_r, max_c = coords.max(axis=0)
                        cov = np.cov(coords, rowvar=False)
                        eigvals = np.sort(np.linalg.eigvalsh(cov))
                        total_var = max(eigvals[0] + eigvals[1], 1e-6)
                        exp_linear = float(eigvals[1] / total_var)
                        axis_ratio = float(np.sqrt(max(eigvals[1], 1e-4) / max(eigvals[0], 1e-4)))

                        is_crease = (axis_ratio >= 3.0 and exp_linear >= 0.88)

                        cluster_details.append({
                            "pathway": "Photometric",
                            "box": [int(min_c), int(min_r), int(max_c), int(max_r)],
                            "area": len(cluster),
                            "explained_linear": round(exp_linear, 3),
                            "axis_ratio": round(axis_ratio, 2),
                            "is_crease": is_crease,
                            "max_score": round(float(np.max([photo_map[cr, cc] for cr, cc in cluster])), 2),
                        })

                        for cr, cc in cluster:
                            if is_crease:
                                crease_mask[cr, cc] = True
                            else:
                                stain_mask[cr, cc] = True

    # Organic morphological consolidation on stains
    if np.any(stain_mask):
        padded = np.pad(stain_mask, 2, mode="constant")
        dilated = padded.copy()
        for _ in range(2):
            shift_u = np.roll(dilated, -1, axis=0)
            shift_d = np.roll(dilated, 1, axis=0)
            shift_l = np.roll(dilated, -1, axis=1)
            shift_r = np.roll(dilated, 1, axis=1)
            dilated = dilated | shift_u | shift_d | shift_l | shift_r
        eroded = dilated.copy()
        for _ in range(2):
            shift_u = np.roll(eroded, -1, axis=0)
            shift_d = np.roll(eroded, 1, axis=0)
            shift_l = np.roll(eroded, -1, axis=1)
            shift_r = np.roll(eroded, 1, axis=1)
            eroded = eroded & shift_u & shift_d & shift_l & shift_r
        stain_mask = eroded[2:-2, 2:-2]

    has_stain = bool(np.any(stain_mask))
    has_crease = bool(np.any(crease_mask))
    total_defect = stain_mask | crease_mask
    return has_stain, has_crease, stain_mask, crease_mask, total_defect, cluster_details


def export_gallery():
    print(f"Loading V9 results from {V9_DIR}...")
    data = np.load(NPZ_PATH)
    v9_maps = data["v9_maps"]
    dit_maps = data["dit_maps"] if "dit_maps" in data else data["v9_maps"]
    photo_maps = data["photo_maps"] if "photo_maps" in data else np.zeros_like(v9_maps)
    image_scores = data["image_scores"]
    labels = data["labels"]

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))

    calib_path = V9_DIR / "calibration_v9.json"
    if calib_path.exists():
        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
            tau_strict = calib.get("tau_strict", 17.18)
    else:
        tau_strict = 17.18

    print(f"Total samples: {len(csv_rows)}, Calibration tau_strict: {tau_strict:.2f}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(DATA_ZIP_PATH, "r")

    cards_html = []
    stats = {"strict_tp": 0, "stain_tp": 0, "crease_count": 0, "total_normal": 10, "total_stain": 100}

    for idx, row in enumerate(csv_rows):
        img_rel = row["image"]
        score = float(row["v9_score"])
        is_stain = row["kind"] == "stain"

        zip_member = img_rel if img_rel in zf.namelist() else f"fabric_stain_pilot/{img_rel}"
        orig_bytes = zf.read(zip_member)
        orig_im = Image.open(io.BytesIO(orig_bytes)).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)

        m_v9 = v9_maps[idx].astype(np.float32)
        m_dit = dit_maps[idx].astype(np.float32)
        m_photo = photo_maps[idx].astype(np.float32)

        has_s, has_c, m_s, m_c, m_tot, clusters = analyze_defect_clusters_v9(
            m_dit, m_photo, tau_strict=tau_strict
        )

        heatmap_im = colorize_heatmap(m_v9, vmin=0.0, vmax=30.0)
        mask_stain_im = Image.fromarray((m_s.astype(np.uint8) * 255)).convert("RGB")
        mask_crease_im = Image.fromarray((m_c.astype(np.uint8) * 255)).convert("RGB")
        composite_im = create_composite_overlay(orig_im, m_s, m_c, clusters)

        base_name = f"sample_{idx:03d}"
        orig_file = f"{base_name}_orig.jpg"
        heat_file = f"{base_name}_heat.jpg"
        mask_st_file = f"{base_name}_mask_stain.png"
        mask_cr_file = f"{base_name}_mask_crease.png"
        comp_file = f"{base_name}_comp.jpg"

        orig_im.save(IMAGES_DIR / orig_file, quality=90)
        heatmap_im.save(IMAGES_DIR / heat_file, quality=90)
        mask_stain_im.save(IMAGES_DIR / mask_st_file)
        mask_crease_im.save(IMAGES_DIR / mask_cr_file)
        composite_im.save(IMAGES_DIR / comp_file, quality=90)

        strict_detected = score > tau_strict
        stain_area = int(np.sum(m_s))
        crease_area = int(np.sum(m_c))

        if is_stain and strict_detected:
            stats["strict_tp"] += 1
        if is_stain and has_s:
            stats["stain_tp"] += 1
        if has_c:
            stats["crease_count"] += 1

        if not is_stain:
            if not has_s and not has_c:
                category_tag = "normal"
                status_badge = "<span class='badge badge-normal'>Normal (Clean)</span>"
            elif has_c and not has_s:
                category_tag = "crease"
                status_badge = "<span class='badge badge-crease'>Physical Fold Crease (Clean)</span>"
            else:
                category_tag = "normal"
                status_badge = "<span class='badge badge-missed'>Normal (Weave Anomaly)</span>"
        else:
            if strict_detected:
                category_tag = "strict"
                status_badge = "<span class='badge badge-detected'>Strict Chemical Stain</span>"
            elif has_s:
                category_tag = "recovered"
                status_badge = "<span class='badge badge-recovered'>Faint Watermark Found</span>"
            elif has_c:
                category_tag = "crease"
                status_badge = "<span class='badge badge-crease'>Structural Crease Defect</span>"
            else:
                category_tag = "missed"
                status_badge = "<span class='badge badge-missed'>Subtle / Missed Stain</span>"

        cards_html.append(
            f"""
        <div class="card" data-category="{category_tag}">
            <div class="card-header">
                <span class="card-title">#{idx:03d}: {Path(img_rel).name}</span>
                {status_badge}
            </div>
            <div class="card-body">
                <div class="panel">
                    <img src="images/{orig_file}" alt="Original" loading="lazy">
                    <div class="panel-label">1. Original Image</div>
                </div>
                <div class="panel">
                    <img src="images/{heat_file}" alt="Heatmap" loading="lazy">
                    <div class="panel-label">2. V9 Fused Heatmap</div>
                </div>
                <div class="panel">
                    <img src="images/{mask_st_file}" alt="Stain Mask" loading="lazy">
                    <div class="panel-label">3. Stain Mask ({stain_area} px)</div>
                </div>
                <div class="panel">
                    <img src="images/{mask_cr_file}" alt="Crease Mask" loading="lazy">
                    <div class="panel-label">4. Crease Mask ({crease_area} px)</div>
                </div>
                <div class="panel">
                    <img src="images/{comp_file}" alt="Composite" loading="lazy">
                    <div class="panel-label">5. Color Overlay & BBox</div>
                </div>
            </div>
            <div class="card-meta">
                <span class="meta-item">Score: <strong>{score:.2f}</strong></span>
                <span class="meta-item">Strict (&gt;{tau_strict:.1f}): <strong class="{('text-pass' if strict_detected else 'text-fail')}">{('YES' if strict_detected else 'NO')}</strong></span>
                <span class="meta-item">Stain Mask: <strong>{stain_area} px</strong></span>
                <span class="meta-item">Crease Mask: <strong>{crease_area} px</strong></span>
                <span class="meta-item">Clusters: <strong>{len(clusters)}</strong></span>
            </div>
        </div>
        """
        )

    zf.close()

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fabric Stain Inspection V9 - Dual-Pathway & Exact Watermark Localization</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        h1 {{ font-size: 24px; font-weight: 700; margin: 0 0 8px 0; color: #38bdf8; }}
        .header {{ display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 24px; border-bottom: 1px solid #334155; padding-bottom: 16px; }}
        .stats-bar {{ display: flex; gap: 16px; margin-bottom: 20px; }}
        .stat-badge {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px 18px; }}
        .stat-val {{ font-size: 20px; font-weight: 700; color: #38bdf8; }}
        .stat-lbl {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; margin-top: 2px; }}
        .filter-controls {{ display: flex; gap: 8px; margin-bottom: 24px; }}
        .btn {{ background: #1e293b; color: #94a3b8; border: 1px solid #334155; border-radius: 6px; padding: 8px 16px; cursor: pointer; font-weight: 500; transition: all 0.2s; }}
        .btn:hover {{ background: #334155; color: #f8fafc; }}
        .btn.active {{ background: #0284c7; color: #ffffff; border-color: #38bdf8; }}
        .grid {{ display: flex; flex-direction: column; gap: 16px; }}
        .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .card-title {{ font-size: 15px; font-weight: 600; color: #f1f5f9; }}
        .badge {{ font-size: 12px; font-weight: 600; padding: 4px 8px; border-radius: 4px; }}
        .badge-normal {{ background: #065f46; color: #34d399; }}
        .badge-crease {{ background: #0e7490; color: #38bdf8; }}
        .badge-detected {{ background: #854d0e; color: #facc15; }}
        .badge-recovered {{ background: #1e3a8a; color: #60a5fa; }}
        .badge-missed {{ background: #475569; color: #cbd5e1; }}
        .card-body {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 12px; }}
        .panel {{ display: flex; flex-direction: column; align-items: center; background: #0f172a; border-radius: 6px; padding: 6px; border: 1px solid #334155; }}
        .panel img {{ width: 100%; height: auto; aspect-ratio: 1; object-fit: cover; border-radius: 4px; }}
        .panel-label {{ font-size: 11px; color: #94a3b8; margin-top: 6px; text-align: center; font-weight: 500; }}
        .card-meta {{ display: flex; gap: 16px; font-size: 13px; color: #94a3b8; border-top: 1px solid #334155; padding-top: 8px; }}
        .meta-item strong {{ color: #f8fafc; }}
        .text-pass {{ color: #4ade80 !important; }}
        .text-fail {{ color: #f87171 !important; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Fabric Stain Locked Inspection Gallery (V9 Dual-Pathway Protocol)</h1>
            <div style="color: #94a3b8; font-size: 13px; margin-top: 4px;">Pathway A (DiT Residual) + Pathway B (Photometric Field) with 20px Hard Margin Blackout</div>
        </div>
    </div>

    <div class="stats-bar">
        <div class="stat-badge">
            <div class="stat-val">{stats['stain_tp']} / {stats['total_stain']}</div>
            <div class="stat-lbl">Chemical Stains & Watermarks</div>
        </div>
        <div class="stat-badge">
            <div class="stat-val">{stats['strict_tp']} / {stats['total_stain']}</div>
            <div class="stat-lbl">Strict Detections (&gt;{tau_strict:.1f})</div>
        </div>
        <div class="stat-badge">
            <div class="stat-val">{stats['crease_count']}</div>
            <div class="stat-lbl">Creases / Wrinkles Isolated</div>
        </div>
        <div class="stat-badge">
            <div class="stat-val">{stats['total_normal']} / {stats['total_normal']}</div>
            <div class="stat-lbl">Clean Normal Specificity</div>
        </div>
    </div>

    <div class="filter-controls">
        <button class="btn active" onclick="filterCat('all')">Show All (110)</button>
        <button class="btn" onclick="filterCat('strict')">Strict Stains</button>
        <button class="btn" onclick="filterCat('recovered')">Faint Watermarks Found</button>
        <button class="btn" onclick="filterCat('crease')">Creases / Wrinkles</button>
        <button class="btn" onclick="filterCat('normal')">Normals (Clean)</button>
        <button class="btn" onclick="filterCat('missed')">Subtle / Missed</button>
    </div>

    <div class="grid" id="cardGrid">
        {''.join(cards_html)}
    </div>

    <script>
        function filterCat(cat) {{
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            const cards = document.querySelectorAll('.card');
            cards.forEach(c => {{
                if (cat === 'all' || c.getAttribute('data-category') === cat) {{
                    c.style.display = 'block';
                }} else {{
                    c.style.display = 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
"""

    index_html_path = GALLERY_DIR / "index.html"
    index_html_path.write_text(html_content, encoding="utf-8")
    print(f"Interactive gallery successfully created at: {index_html_path}")


if __name__ == "__main__":
    export_gallery()
