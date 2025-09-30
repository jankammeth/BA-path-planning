#!/usr/bin/env python3
"""
Make a box plot of SCP runtimes from all CSVs in a folder.

- X: N (number of robots)
- Y: time_sec (log-scale), success-only
- Median: orange, Mean: green dashed
"""

import csv
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ---------- CONFIG ----------
CONFIG = {
    "data_dir": "results/trial_2",  # folder with scp_benchmark_*.csv
    "out_path": "plots/scp_boxplot.pdf",  # where to save the plot
}
# ---------------------------


def load_rows_from_dir(data_dir: str):
    """Read all scp_benchmark_*.csv rows from a directory."""
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Directory does not exist: {data_dir}")
    files = sorted(glob.glob(str(p / "scp_benchmark_*.csv")))
    if not files:
        raise FileNotFoundError(f"No 'scp_benchmark_*.csv' files in {data_dir}")

    rows = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
        except Exception as e:
            print(f"[warn] Skipping {fp}: {e}")
    if not rows:
        raise RuntimeError("No rows loaded.")
    return rows


def group_times_by_N(rows):
    """Return {N: [time_sec, ...]} for status=='success' only."""
    byN = {}
    for r in rows:
        try:
            if r.get("status", "").strip().lower() == "success":
                N = int(r["N"])
                t = float(r["time_sec"])
                if np.isfinite(t):
                    byN.setdefault(N, []).append(t)
        except Exception:
            continue
    if not byN:
        raise RuntimeError("No successful runs found.")
    return {N: sorted(v) for N, v in sorted(byN.items())}


def plot_runtime_boxplot(times_by_N: dict, out_path: str):
    """Clean boxplot: black/white, log y, full grid, orange median, green mean."""
    Ns = list(times_by_N.keys())
    data = [np.asarray(times_by_N[N], float) for N in Ns]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(
        data,
        vert=True,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.6,
        whis=1.5,
        boxprops=dict(edgecolor="black", linewidth=1.2, facecolor="white"),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        medianprops=dict(color="#E67E22", linewidth=1.5),  # orange
        meanprops=dict(color="#2ECC71", linewidth=1.5, linestyle="--"),  # green dashed
        flierprops=dict(
            marker="o", markerfacecolor="none", markeredgecolor="black", alpha=0.35, markersize=5
        ),
    )

    ax.set_xticks(range(1, len(Ns) + 1))
    ax.set_xticklabels([str(N) for N in Ns])
    ax.set_xlabel("Number of robots N")
    ax.set_ylabel("Computation time per run [s] (log scale)")
    ax.set_title("SCP Computation Time vs Number of Robots")

    ax.set_yscale("log")
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    ax.legend(
        handles=[
            Line2D([0], [0], color="#E67E22", lw=1.5, label="Median"),
            Line2D([0], [0], color="#2ECC71", lw=1.5, ls="--", label="Mean"),
        ],
        loc="upper left",
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)
    return out_path


def make_boxplot(data_dir: str, out_path: str):
    """One-call utility: load → group → plot."""
    rows = load_rows_from_dir(data_dir)
    times_by_N = group_times_by_N(rows)
    return plot_runtime_boxplot(times_by_N, out_path)


def main():
    saved = make_boxplot(CONFIG["data_dir"], CONFIG["out_path"])
    print(f"Saved plot: {saved}")


if __name__ == "__main__":
    main()
