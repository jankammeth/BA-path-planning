#!/usr/bin/env python3
"""
main_scp_boxplot.py

Read SCP benchmark CSV results (from main_scp_benchmark.py) and create a box plot:
  - X-axis: N (number of robots)
  - Y-axis: time_sec (wall time per run)
Supports either a single CSV file or a directory containing multiple CSVs.

Usage:
  python main_scp_boxplot.py --csv results/scp_benchmark_20250101_120000.csv
  python main_scp_boxplot.py --dir results
  python main_scp_boxplot.py --dir results --out plot_scp_box.png --show
"""

import argparse
import csv
import glob
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe


def _apply_wow_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )


# light, stage-friendly palette (feel free to swap)
_LIGHTS = [
    (0.85, 0.90, 1.00),  # soft blue
    (0.93, 0.86, 0.98),  # soft purple
    (1.00, 0.90, 0.90),  # soft red
    (1.00, 0.95, 0.85),  # soft orange
    (0.90, 0.97, 0.97),  # soft teal
    (0.95, 0.95, 0.85),  # soft yellow-green
]


def plot_runtime_boxplot(
    results_by_N,
    ylabel="Runtime [s]",
    title="SCP Runtime vs Fleet Size",
    savepath="scp_boxplot.pdf",
    show_means=True,
    swarm=False,
    swarm_alpha=0.6,
):
    """
    Prettier boxplot with soft fills, crisp medians, and optional mean dots + swarm overlay.
    """
    _apply_wow_style()

    Ns = sorted(results_by_N.keys())
    data = [np.asarray(results_by_N[N], dtype=float) for N in Ns]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Optional: light swarm of raw points for each box
    if swarm:
        for i, arr in enumerate(data, start=1):
            x = np.random.normal(loc=i, scale=0.06, size=len(arr))
            ax.scatter(x, arr, s=10, alpha=swarm_alpha, edgecolors="none")

    bp = ax.boxplot(
        data,
        vert=True,
        patch_artist=True,
        showmeans=show_means,
        meanline=False,
        widths=0.6,
        whis=1.5,
        manage_ticks=False,
    )

    # Styling
    for i, box in enumerate(bp["boxes"]):
        face = _LIGHTS[i % len(_LIGHTS)]
        box.set(
            facecolor=face,
            edgecolor="black",
            linewidth=1.2,
            zorder=2,
        )
        # subtle outer stroke for crisp edges on projectors
        box.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])

    for whisker in bp["whiskers"]:
        whisker.set(color="black", linewidth=1.0, solid_capstyle="round", solid_joinstyle="round")
    for cap in bp["caps"]:
        cap.set(color="black", linewidth=1.0)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2.5, zorder=3)
        median.set_path_effects([pe.withStroke(linewidth=4.0, foreground="white")])

    # Mean markers: small, neutral
    if show_means and "means" in bp:
        for mean in bp["means"]:
            mean.set(
                marker="o",
                markersize=5,
                markerfacecolor="black",
                markeredgecolor="white",
                linestyle="None",
                zorder=4,
            )

    # Whisker fliers (outliers)
    if "fliers" in bp:
        for flier in bp["fliers"]:
            flier.set(
                marker="o",
                alpha=0.35,
                markersize=5,
                markerfacecolor="none",
                markeredgecolor="black",
            )

    # X tick labels in LaTeX
    ax.set_xticks(range(1, len(Ns) + 1))
    ax.set_xticklabels([rf"$N={N}$" for N in Ns])

    ax.set_xlabel(r"$N$ (number of crafts)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Light y-grid for readability
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    # Tight layout + PDF
    plt.tight_layout()
    plt.savefig(savepath, dpi=400)
    # plt.show()
    return fig, ax


def load_runs_from_csv(csv_path):
    """Load rows from one CSV and return a list of dicts."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def collect_runs(csv_path=None, dir_path=None):
    """Gather runs from a single CSV or from every CSV in a directory."""
    if (csv_path is None) == (dir_path is None):
        raise ValueError("Provide exactly one of --csv or --dir")

    runs = []
    files = []

    if csv_path:
        files = [csv_path]
    else:
        p = Path(dir_path)
        if not p.exists():
            raise FileNotFoundError(f"Directory does not exist: {dir_path}")
        files = sorted(glob.glob(str(p / "scp_benchmark_*.csv")))
        if not files:
            raise FileNotFoundError(f"No 'scp_benchmark_*.csv' files found in {dir_path}")

    for fp in files:
        try:
            runs.extend(load_runs_from_csv(fp))
        except Exception as e:
            print(f"[warn] Skipping {fp}: {e}")

    if not runs:
        raise RuntimeError("No runs loaded.")

    return runs, files


def prepare_data(runs):
    """
    Convert list of run dicts to grouped numeric arrays per N, success-only.
    Returns dict: N -> list of times, and error counts per N.
    """
    times_by_N = defaultdict(list)
    errors_by_N = defaultdict(int)

    for r in runs:
        try:
            N = int(r.get("N", ""))
        except Exception:
            continue

        status = (r.get("status") or "").strip().lower()
        if status == "success":
            try:
                t = float(r.get("time_sec", "nan"))
                if np.isfinite(t):
                    times_by_N[N].append(t)
            except Exception:
                pass
        else:
            errors_by_N[N] += 1

    # sort keys and keep only those with at least one time
    Ns = sorted([N for N, arr in times_by_N.items() if len(arr) > 0])
    return Ns, times_by_N, errors_by_N


def print_stats(Ns, times_by_N, errors_by_N):
    print("\nSummary (success-only):")
    for N in Ns:
        arr = np.array(times_by_N[N], dtype=float)
        cnt = len(arr)
        if cnt == 0:
            print(f"  N={N}: no successful runs (errors={errors_by_N.get(N,0)})")
            continue
        p25, med, p75 = np.percentile(arr, [25, 50, 75])
        print(
            f"  N={N}: n={cnt}, errors={errors_by_N.get(N,0)}, "
            f"min={arr.min():.3f}s, p25={p25:.3f}s, median={med:.3f}s, "
            f"p75={p75:.3f}s, max={arr.max():.3f}s, mean={arr.mean():.3f}s, std={arr.std(ddof=1):.3f}s"
        )


def make_boxplot(Ns, times_by_N, title=None, out_path=None, show=False):
    """
    Create and save a box plot: N vs time_sec.
    - Uses tick_labels instead of deprecated labels.
    - Uses whis=(0,100) to span min..max (replacement for 'range').
    - Switches to log y-scale automatically if spread is large.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    data = [sorted(times_by_N[N]) for N in Ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        data,
        tick_labels=[str(N) for N in Ns],  # <-- rename from labels=
        showmeans=True,
        meanline=True,
        whis=(0, 100),  # <-- replace "range"
        showfliers=True,
    )

    ax.set_xlabel("Number of robots N")
    ax.set_ylabel("Computation time per run N[s]")
    if title:
        ax.set_title(title)

    # If the spread is huge (e.g., > 20x), use log scale for readability
    all_vals = np.concatenate([np.array(d, dtype=float) for d in data if len(d) > 0])

    if len(all_vals) > 0 and np.nanmax(all_vals) / max(np.nanmin(all_vals), 1e-12) > 20:
        ax.set_yscale("log")
        ax.set_ylabel("Computation time per run [s] (log scale)")

    ax.set_ylabel("Computation time per run [s]")

    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"scp_boxplot_{stamp}.png"

    plt.savefig(out_path, dpi=200)
    print(f"\nSaved plot: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create box plot from SCP benchmark CSVs.")
    parser.add_argument(
        "--csv", type=str, default=None, help="Path to a single benchmark CSV file."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing multiple benchmark CSVs to merge.",
    )
    parser.add_argument("--out", type=str, default=None, help="Output PNG path for the plot.")
    parser.add_argument(
        "--title", type=str, default="SCP Computation Time vs Number of Robots", help="Plot title."
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    args = parser.parse_args()

    runs, files = collect_runs(csv_path=args.csv, dir_path=args.dir)
    print(f"Loaded {len(runs)} runs from {len(files)} file(s).")

    Ns, times_by_N, errors_by_N = prepare_data(runs)
    if not Ns:
        raise RuntimeError("No successful runs found to plot.")

    print_stats(Ns, times_by_N, errors_by_N)
    make_boxplot(Ns, times_by_N, title=args.title, out_path=args.out, show=args.show)


if __name__ == "__main__":
    main()
