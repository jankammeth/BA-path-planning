import numpy as np
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from typing import Optional

# ---------- Colors (8 clusters) ----------
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

# ---------- Initial edge clusters (200×200) ----------
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

    pts, cluster_ids = [], []
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

    # Even sampling along the total outline length
    t_vals = np.linspace(0.0, total_len, n_points, endpoint=False)
    pts, acc, seg_idx = [], 0.0, 0
    for t in t_vals:
        while seg_idx < len(segs) and acc + segs[seg_idx][2] <= t:
            acc += segs[seg_idx][2]; seg_idx += 1
        if seg_idx >= len(segs): seg_idx = len(segs) - 1
        a, b, L = segs[seg_idx]; s = (t - acc) / L if L > 0 else 0.0
        pts.append(a + s * (b - a))
    pts = np.array(pts)

    # Normalize and scale into box
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

# ---------- Side clusters (mid-left & mid-right) ----------
def side_cluster_targets(n_points, box_size=(200, 200), rng=None):
    if rng is None:
        rng = np.random.default_rng()
    W, H = box_size
    centers = np.array([[10.0, H/2.0], [W-10.0, H/2.0]])  # mid-left, mid-right
    base = n_points // 2
    rem  = n_points % 2
    counts = [base + (1 if i < rem else 0) for i in range(2)]

    pts = []
    cov = np.array([[16.0, 0.0], [0.0, 16.0]])  # std ≈ 4 m
    for cid, c in enumerate(centers):
        k = counts[cid]
        local = rng.multivariate_normal(c, cov, size=k)
        local[:, 0] = np.clip(local[:, 0], 2.0, W-2.0)
        local[:, 1] = np.clip(local[:, 1], 2.0, H-2.0)
        pts.append(local)
    return np.vstack(pts)

# ---------- Greedy assignment (equal sizes) ----------
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

# ---------- Assign a subset of crafts to a smaller target set ----------
def assign_subset_to_targets(src_points, tgt_points):
    """
    For each target point, pick the nearest *unused* source craft.
    Returns:
      chosen_src_idx: (M,) indices in [0..N-1] for crafts assigned to targets
      mapping_src_to_tgt: dict {src_idx -> target_idx}
    """
    N = len(src_points)
    M = len(tgt_points)
    unused = set(range(N))
    mapping = {}
    chosen = []
    for j in range(M):
        arr = np.fromiter(unused, dtype=int)
        d2  = np.sum((src_points[arr] - tgt_points[j])**2, axis=1)
        i = arr[np.argmin(d2)]
        mapping[i] = j
        chosen.append(i)
        unused.remove(i)
    return np.array(chosen, dtype=int), mapping

# ---------- Plot (minimal: dots + lines only) ----------
def plot_mapping(initial, wow, x_stage, zurich, colors):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Start = circles
    for i, p in enumerate(initial):
        ax.scatter(*p, marker='o', s=12, color=colors[i])
    # WOW / x stage / ZURICH = diamonds
    for i, p in enumerate(wow):
        ax.scatter(*p, marker='D', s=12, color=colors[i])
    for i, p in enumerate(x_stage):
        ax.scatter(*p, marker='D', s=12, color=colors[i])
    for i, p in enumerate(zurich):
        ax.scatter(*p, marker='D', s=12, color=colors[i])

    # Paths
    for i in range(len(initial)):
        ax.plot([initial[i,0], wow[i,0]],     [initial[i,1], wow[i,1]],     c=colors[i], alpha=0.2, lw=0.9)
        ax.plot([wow[i,0], x_stage[i,0]],     [wow[i,1], x_stage[i,1]],     c=colors[i], alpha=0.2, lw=0.9)
        ax.plot([x_stage[i,0], zurich[i,0]],  [x_stage[i,1], zurich[i,1]],  c=colors[i], alpha=0.2, lw=0.9)

    ax.set_aspect('equal'); ax.set_xlim(-2, 202); ax.set_ylim(-2, 202); ax.axis('off')
    plt.tight_layout()
    plt.savefig("wow_x_zurich_splitX_200m_500crafts.pdf", dpi=400, bbox_inches='tight', pad_inches=0)
    plt.show()

# ---------- Main ----------
def main():
    rng = np.random.default_rng(7)
    N = 500

    # Starts + colors
    init, cluster_ids = generate_edge_cluster_positions(N, min_distance=1.0, rng=rng)
    cols = cluster_colors(cluster_ids)

    # Non-italic fonts everywhere (plain sans-serif)
    plain_fp = FontProperties(family="sans-serif", style="normal")

    # Stage 1: WOW (all 500)
    wow = text_outline_points("WOW", N, (200, 200),
                              target_width_ratio=0.75, y_center_ratio=0.78,
                              fontprop=plain_fp)

    # Assign edges -> WOW
    idx1 = greedy_match(init, wow)
    wow  = wow[idx1]

    # Stage 2: split for "x" (100) vs side clusters (400)
    M_x = 100
    x_targets   = text_outline_points("x", M_x, (200, 200),
                                      target_width_ratio=0.18, y_center_ratio=0.55,
                                      fontprop=plain_fp)
    side_targets = side_cluster_targets(N - M_x, (200, 200), rng=rng)

    # Pick which 100 crafts go to "x": choose nearest crafts from current WOW positions
    chosen_idx, src_to_x_tgt = assign_subset_to_targets(wow, x_targets)  # length 100
    remaining_idx = np.array([i for i in range(N) if i not in src_to_x_tgt], dtype=int)

    # Build the x-stage positions for all N crafts
    x_stage = np.empty_like(wow)
    # Fill x-assigned
    for i in chosen_idx:
        x_stage[i] = x_targets[src_to_x_tgt[i]]
    # Remaining crafts go to side clusters (greedy match on the subset)
    idx_side = greedy_match(wow[remaining_idx], side_targets)  # map subset -> side targets
    x_stage[remaining_idx] = side_targets[idx_side]

    # Stage 3: ZURICH (all 500)
    zurich_targets = text_outline_points("ZURICH", N, (200, 200),
                                         target_width_ratio=0.95, y_center_ratio=0.32,
                                         fontprop=plain_fp)
    idx3 = greedy_match(x_stage, zurich_targets)
    zurich = zurich_targets[idx3]

    # Plot
    plot_mapping(init, wow, x_stage, zurich, cols)

if __name__ == "__main__":
    main()