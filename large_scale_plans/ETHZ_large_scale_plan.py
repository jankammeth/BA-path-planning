import numpy as np
import matplotlib.pyplot as plt
from svgpathtools import svg2paths2
from matplotlib.font_manager import FontProperties

# ---------- Colors (8 clusters) ----------
palette8 = [
    (1.00, 0.70, 0.70),  # pastel red (pinkish)
    (0.70, 0.85, 1.00),  # very light blue
    (0.70, 1.00, 0.70),  # pastel green
    (1.00, 0.65, 0.00),  # orange (keep, bright)
    (0.85, 0.75, 1.00),  # pastel purple / lilac
    (0.95, 0.85, 0.70),  # beige / light tan
    (1.00, 1.00, 1.00),  # white
    (0.95, 1.00, 0.70),  # yellow-green (keep, bright)
]
def cluster_colors(cluster_ids):
    return [palette8[i % len(palette8)] for i in cluster_ids]

# ---------- Global box (2000 × 1000) ----------
W, H = 2000.0, 1000.0
MARGIN = 100.0  # ~5% margin

# ---------- Initial edge clusters (2000×1000) ----------
def generate_edge_cluster_positions(n_vehicles, min_distance=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    centers = np.array([
        [MARGIN,       MARGIN],       [W/2,        MARGIN],       [W-MARGIN,       MARGIN],   # bottom
        [MARGIN,       H/2],                                              [W-MARGIN,       H/2],  # middle sides
        [MARGIN,       H-MARGIN],     [W/2,        H-MARGIN],     [W-MARGIN,       H-MARGIN], # top
    ], float)

    base = n_vehicles // len(centers)
    rem  = n_vehicles % len(centers)
    counts = [base + (1 if i < rem else 0) for i in range(len(centers))]

    pts, cluster_ids = [], []
    cov = np.array([[80.0, 0.0], [0.0, 40.0]])  # elliptical spread
    for cid, c in enumerate(centers):
        k = counts[cid]
        local = []
        attempts = 0
        while len(local) < k and attempts < 20000:
            p = rng.multivariate_normal(c, cov)
            if not (1 <= p[0] <= W-1 and 1 <= p[1] <= H-1):
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

# ---------- SVG outline sampler ----------
def svg_outline_points(svg_file, n_points, box_size=(W, H),
                       target_width_ratio=0.9, y_center_ratio=0.5):
    paths, attrs, svg_attr = svg2paths2(svg_file)

    lengths = [p.length() for p in paths]
    total_len = float(np.sum(lengths))
    if total_len <= 0:
        raise ValueError("SVG has zero total path length.")

    per_path = [max(1, int(round(n_points * (L / total_len)))) for L in lengths]
    delta = n_points - sum(per_path)
    if delta != 0:
        order = np.argsort([-l for l in lengths]) if delta > 0 else np.argsort(lengths)
        for i in order[:abs(delta)]:
            per_path[i] += 1 if delta > 0 else -1
            if per_path[i] < 1: per_path[i] = 1

    pts = []
    for p, m in zip(paths, per_path):
        if m <= 0: continue
        s_vals = np.linspace(0.0, p.length(), m, endpoint=False)
        for s in s_vals:
            t = p.ilength(s) if p.length() > 0 else 0.0
            z = p.point(t)
            pts.append([z.real, z.imag])

    P = np.array(pts, dtype=float)
    P[:,1] = -P[:,1]  # flip y-axis

    min_xy, max_xy = P.min(0), P.max(0)
    size = np.maximum(max_xy - min_xy, 1e-9)
    P = (P - min_xy) / size

    Wb, Hb = box_size
    target_w = target_width_ratio * Wb
    aspect   = size[1] / size[0]
    target_h = target_w * aspect
    if target_h > 0.8 * Hb:
        target_h = 0.8 * Hb
        target_w = target_h / aspect

    x0 = (Wb - target_w) / 2.0
    y0 = Hb * y_center_ratio - target_h / 2.0

    P[:, 0] = x0 + P[:, 0] * target_w
    P[:, 1] = y0 + P[:, 1] * target_h
    return P

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

# ---------- Plot ----------
def plot_mapping(initial, logo_pts, colors):
    fig, ax = plt.subplots(figsize=(12, 6))  # rectangular figure for rectangular workspace

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    for i, p in enumerate(initial):
        ax.scatter(*p, marker='o', s=12, color=colors[i])
    for i, p in enumerate(logo_pts):
        ax.scatter(*p, marker='D', s=12, color=colors[i])
    for i in range(len(initial)):
        ax.plot([initial[i,0], logo_pts[i,0]],
                [initial[i,1], logo_pts[i,1]],
                c=colors[i], alpha=0.2, lw=0.9)

    ax.set_aspect('equal')
    ax.set_xlim(-20, W+20)
    ax.set_ylim(-20, H+20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig("clusters_to_ETHZ_logo_2000x1000.png",
                dpi=400, bbox_inches='tight', pad_inches=0,
                transparent=True)
    plt.show()

# ---------- Main ----------
def main():
    rng = np.random.default_rng(7)
    N = 500

    init, cluster_ids = generate_edge_cluster_positions(N, min_distance=1.0, rng=rng)
    cols = cluster_colors(cluster_ids)

    logo = svg_outline_points("ETHZ.svg", N, box_size=(W, H),
                              target_width_ratio=0.9, y_center_ratio=0.5)

    idx = greedy_match(init, logo)
    logo_assigned = logo[idx]

    plot_mapping(init, logo_assigned, cols)

if __name__ == "__main__":
    main()