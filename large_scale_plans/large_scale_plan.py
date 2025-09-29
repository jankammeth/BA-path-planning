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


def main():
    rng = np.random.default_rng(7)
    N = 500

    init, cluster_ids = generate_edge_cluster_positions(N, min_distance= 1.0, rng = rng)


if __name__ == "__main__":
    main()