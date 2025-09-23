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
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_runs_from_csv(csv_path):
    """Load rows from one CSV and return a list of dicts."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
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
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime

    data = [sorted(times_by_N[N]) for N in Ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        data,
        tick_labels=[str(N) for N in Ns],  # <-- rename from labels=
        showmeans=True,
        meanline=True,
        whis=(0, 100),                     # <-- replace "range"
        showfliers=True
    )

    ax.set_xlabel("Number of robots N")
    ax.set_ylabel("Computation time per run [s]")
    if title:
        ax.set_title(title)

    # If the spread is huge (e.g., > 20x), use log scale for readability
    all_vals = np.concatenate([np.array(d, dtype=float) for d in data if len(d) > 0])
    if len(all_vals) > 0 and np.nanmax(all_vals) / max(np.nanmin(all_vals), 1e-12) > 20:
        ax.set_yscale("log")
        ax.set_ylabel("Computation time per run [s] (log scale)")

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
    parser.add_argument("--csv", type=str, default=None, help="Path to a single benchmark CSV file.")
    parser.add_argument("--dir", type=str, default=None, help="Directory containing multiple benchmark CSVs to merge.")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path for the plot.")
    parser.add_argument("--title", type=str, default="SCP Computation Time vs Number of Robots (20 second trajectories at h = 0.1s)", help="Plot title.")
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