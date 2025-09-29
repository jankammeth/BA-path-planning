import numpy as np
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from typing import Optional

# =========================
# Config: optional custom font for elegant lowercase italic "of"
CUSTOM_ITALIC_FONT_PATH = None  # e.g., "/path/to/EBGaramond-Italic.ttf"

# ---------- 8-color palette ----------
palette8 = [
    (0.00, 0.13, 0.18),  # deep navy (#00202E)
    (0.00, 0.25, 0.36),  # dark teal (#003F5C)
    (0.17, 0.28, 0.46),  # blue (#2C4875)
    (0.54, 0.31, 0.56),  # purple (#8A508F)
    (0.74, 0.31, 0.56),  # magenta (#BC5090)
    (1.00, 0.39, 0.38),  # coral (#FF6361)
    (1.00, 0.52, 0.19),  # orange (#FF8531)
    (1.00, 0.65, 0.00),  # yellow-orange (#FFA600)
]

def cluster_colors(cluster_ids):
    return [palette8[i % len(palette8)] for i in cluster_ids]

# ---------- Initial edge clusters (200×200 box) ----------
def generate_edge_cluster_positions(n_vehicles, min_distance=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    centers = np.array([
        [10, 10],   [100, 10],   [190, 10],
        [10, 100],                [190, 100],
        [10, 190],  [100, 190],  [190, 190],
    ], float)

    base = n_vehicles // len(centers)
    rem  = n_vehicles % len(centers)
    counts = [base + (1 if i < rem else 0) for i in range(len(centers))]

    pts = []
    cluster_ids = []
    cov = np.array([[8.0, 0.0], [0.0, 8.0]])  # std ≈ 2.83 m
    for cid, c in enumerate(centers):
        k = counts[cid]
        local = []
        attempts = 0
        while len(local) < k and attempts < 20000:
            p = rng.multivariate_normal(c, cov)
            if not (1 <= p[0] <= 199 and 1 <= p[1] <= 199):
                attempts += 1
                continue
            ok = True
            for q in local:
                if np.linalg.norm(p - q) < min_distance:
                    ok = False
                    break
            if ok:
                local.append(p)
                cluster_ids.append(cid)
            attempts += 1
        pts.extend(local)
    return np.array(pts), np.array(cluster_ids)

# ---------- Text outlines (supports FontProperties) ----------
def text_outline_points(text, n_points, box_size=(200, 200),
                        target_width_ratio=0.75, y_center_ratio=0.55,
                        fontprop: Optional[FontProperties] = None):
    if fontprop is None:
        tp = TextPath((0, 0), text, size=1.0)
    else:
        tp = TextPath((0, 0), text, size=1.0, prop=fontprop)

    polys = tp.to_polygons()
    segs, total_len = [], 0.0
    for poly in polys:
        if len(poly) < 2:
            continue
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            L = np.linalg.norm(b - a)
            if L > 0:
                segs.append((a, b, L))
                total_len += L

    if total_len <= 0 or len(segs) == 0:
        raise ValueError("Text produced no valid outline segments.")

    # Evenly sample along the outline length
    t_vals = np.linspace(0.0, total_len, n_points, endpoint=False)
    pts, acc, seg_idx = [], 0.0, 0
    for t in t_vals:
        while seg_idx < len(segs) and acc + segs[seg_idx][2] <= t:
            acc += segs[seg_idx][2]; seg_idx += 1
        if seg_idx >= len(segs):
            seg_idx = len(segs) - 1
        a, b, L = segs[seg_idx]
        s = (t - acc) / L if L > 0 else 0.0
        pts.append(a + s * (b - a))
    pts = np.array(pts)

    # Normalize and scale into the box
    min_xy, max_xy = pts.min(axis=0), pts.max(axis=0)
    size = np.maximum(max_xy - min_xy, 1e-6)
    pts = (pts - min_xy) / size

    W, H = box_size
    target_w = target_width_ratio * W
    aspect   = size[1] / size[0]
    target_h = target_w * aspect
    if target_h > 0.8 * H:
        target_h = 0.8 * H
        target_w = target_h / aspect

    x0 = (W - target_w) / 2.0
    y0 = H * y_center_ratio - target_h / 2.0

    return np.column_stack([x0 + pts[:, 0] * target_w,
                            y0 + pts[:, 1] * target_h])

# ---------- Two side-cluster target sets (mid-left & mid-right) ----------
def side_cluster_targets(n_points, box_size=(200, 200), rng=None):
    """
    Generate targets around mid-left (x≈10, y=H/2) and mid-right (x≈W-10, y=H/2).
    Splits n_points as evenly as possible between the two clusters.
    """
    if rng is None:
        rng = np.random.default_rng()
    W, H = box_size
    centers = np.array([
        [10.0,  H/2.0],     # mid-left
        [W-10.0, H/2.0],    # mid-right
    ])
    base = n_points // 2
    rem  = n_points % 2
    counts = [base + (1 if i < rem else 0) for i in range(2)]

    pts = []
    cov = np.array([[16.0, 0.0], [0.0, 16.0]])  # std ≈ 4 m
    for cid, c in enumerate(centers):
        k = counts[cid]
        local = rng.multivariate_normal(c, cov, size=k)
        # clip gently inside the box
        local[:, 0] = np.clip(local[:, 0], 2.0, W-2.0)
        local[:, 1] = np.clip(local[:, 1], 2.0, H-2.0)
        pts.append(local)
    return np.vstack(pts)

# ---------- Greedy assignment ----------
def greedy_match(src, tgt):
    N = len(src)
    unused = set(range(N))
    idx = np.empty(N, dtype=int)
    for i in range(N):
        arr = np.fromiter(unused, dtype=int)
        d2  = np.sum((tgt[arr] - src[i])**2, axis=1)
        j = arr[np.argmin(d2)]
        idx[i] = j
        unused.remove(j)
    return idx

# ---------- Plot (minimal: only dots + lines) ----------
def plot_mapping(initial, way, of_pts, water, colors):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Start = circles
    for i, p in enumerate(initial):
        ax.scatter(*p, marker='o', s=12, color=colors[i])

    # WAY/OF/WATER = diamonds
    for i, p in enumerate(way):
        ax.scatter(*p, marker='D', s=12, color=colors[i])
    for i, p in enumerate(of_pts):
        ax.scatter(*p, marker='D', s=12, color=colors[i])
    for i, p in enumerate(water):
        ax.scatter(*p, marker='D', s=12, color=colors[i])

    # Paths
    for i in range(len(initial)):
        ax.plot([initial[i, 0], way[i, 0]],   [initial[i, 1], way[i, 1]],   c=colors[i], alpha=0.2, lw=0.9)
        ax.plot([way[i, 0], of_pts[i, 0]],    [way[i, 1], of_pts[i, 1]],    c=colors[i], alpha=0.2, lw=0.9)
        ax.plot([of_pts[i, 0], water[i, 0]],  [of_pts[i, 1], water[i, 1]],  c=colors[i], alpha=0.2, lw=0.9)

    ax.set_aspect('equal')
    ax.set_xlim(-2, 202)
    ax.set_ylim(-2, 202)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig("way_of_water_200m_500crafts_splitOF.pdf", dpi=400, bbox_inches='tight', pad_inches=0)
    plt.show()

# ---------- Main ----------
def main():
    rng = np.random.default_rng(7)
    N = 500  # crafts

    # Starts + cluster colors
    init, cluster_ids = generate_edge_cluster_positions(N, min_distance=1.0, rng=rng)
    cols = cluster_colors(cluster_ids)

    # Elegant italic serif for "of"
    if CUSTOM_ITALIC_FONT_PATH:
        italic_fp = FontProperties(fname=CUSTOM_ITALIC_FONT_PATH)
    else:
        italic_fp = FontProperties(family="serif", style="italic")

    # Text outlines (200×200)
    way = text_outline_points("WAY", N, (200, 200), target_width_ratio=0.78, y_center_ratio=0.74)

    # ---- Split for the "of" stage ----
    # Use HALF for the italic "of" outline, and send the other half to side clusters
    N_of_outline = N // 2
    N_side       = N - N_of_outline

    of_outline_pts = text_outline_points("of", N_of_outline, (200, 200),
                                         target_width_ratio=0.45, y_center_ratio=0.51,
                                         fontprop=italic_fp)
    side_pts = side_cluster_targets(N_side, (200, 200), rng=rng)

    # Combine targets for the OF stage (first the text points, then side clusters)
    of_targets_combined = np.vstack([of_outline_pts, side_pts])

    # Edges -> WAY
    idx1 = greedy_match(init, way);  way = way[idx1]

    # WAY -> (OF outline OR side clusters), identity preserved
    idx2 = greedy_match(way, of_targets_combined)
    ofpts = of_targets_combined[idx2]

    # Everyone -> WATER
    water = text_outline_points("WATER", N, (200, 200), target_width_ratio=0.95, y_center_ratio=0.28)
    idx3 = greedy_match(ofpts, water);  water = water[idx3]

    # Plot
    plot_mapping(init, way, ofpts, water, cols)

if __name__ == "__main__":
    main()