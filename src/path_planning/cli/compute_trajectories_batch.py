import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..scenarios.position_generator import (
    generate_positions,
)  # (initial_positions, final_positions)

# Your modules
from ..solvers.scp import SCP

# from position_generator import print_distance_analysis  # optional debug


# ---------------------------- Config ----------------------------
CONFIG = {
    "Ns": [18, 20],  # robot counts to test
    "trials_per_N": 10,  # trials for each N
    "time_horizon": 10.0,  # [s]
    "time_step": 0.2,  # [s]
    "min_distance": 0.8,  # [m]
    "space_dims": [0, 0, 20, 20],  # [x_min, y_min, x_max, y_max]
    "max_iterations": 15,  # SCP iterations
    "rng_seed": None,  # set to int for reproducibility, or None
    "results_dir": "results_new",  # output directory
}
# ---------------------------------------------------------------


def run_single_trial(N, cfg, rng):
    """
    Runs one SCP solve for N vehicles and returns a result dict.
    """
    solver = SCP(
        n_vehicles=N,
        time_horizon=cfg["time_horizon"],
        time_step=cfg["time_step"],
        min_distance=cfg["min_distance"],
        space_dims=cfg["space_dims"],
    )

    # Generate positions (TODO: mutate generator to accept rng if reproducibility is needed)
    init_pos, final_pos = generate_positions(N, cfg["min_distance"])
    # print_distance_analysis(init_pos, final_pos)

    solver.set_initial_states(init_pos)
    solver.set_final_states(final_pos)

    t0 = time.perf_counter()
    status = "success"
    err_msg = None

    try:
        _ = solver.generate_trajectories(max_iterations=cfg["max_iterations"])
    except Exception as e:
        status = "error"
        err_msg = str(e)
    t1 = time.perf_counter()

    result = {
        "N": N,
        "status": status,
        "time_sec": t1 - t0,
        "error": err_msg,
        "K": getattr(solver, "K", None),
        "T": getattr(solver, "T", cfg["time_horizon"]),
        "h": getattr(solver, "h", cfg["time_step"]),
        # You could also store the random seed for this trial if you set one
    }
    return result


def main():
    cfg = CONFIG.copy()

    # Prepare output folder + filenames
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path(cfg["results_dir"]) / f"scp_benchmark_{stamp}.json"
    csv_path = Path(cfg["results_dir"]) / f"scp_benchmark_{stamp}.csv"

    # Global RNG (optional: per-trial seeds from a SeedSequence)
    if cfg["rng_seed"] is not None:
        np.random.seed(cfg["rng_seed"])

    print("------ WOW SCP Benchmark ------")
    print(f"Robot counts: {cfg['Ns']}, Trials per N: {cfg['trials_per_N']}")
    print(
        f"T={cfg['time_horizon']}s, h={cfg['time_step']}s, R={cfg['min_distance']}m, space={cfg['space_dims']}"
    )
    print(f"Max SCP iterations: {cfg['max_iterations']}")
    print()

    all_results = {
        "meta": {
            "timestamp": stamp,
            "description": "SCP timing benchmark for multiple N; each entry is a full solve wall time.",
            "config": cfg,
            "schema_version": "1.0",
        },
        "runs": [],  # list of {N, trial_index, status, time_sec, ...}
        "summary": {},  # filled after
    }

    # Run experiments
    for N in cfg["Ns"]:
        print(f"==> N = {N}")
        for trial in range(cfg["trials_per_N"]):
            # Optional: vary RNG each trial deterministically if seed is set
            if cfg["rng_seed"] is not None:
                np.random.seed(cfg["rng_seed"] + 1000 * N + trial)

            res = run_single_trial(N, cfg, rng=np.random)
            res["trial_index"] = trial
            all_results["runs"].append(res)

            status_str = "OK" if res["status"] == "success" else f"ERR ({res['error']})"
            print(
                f"  trial {trial+1:02d}/{cfg['trials_per_N']}  time = {res['time_sec']:.3f}s  [{status_str}]"
            )

        print()

    # Build summary stats for quick inspection / boxplots later
    for N in cfg["Ns"]:
        times = [
            r["time_sec"] for r in all_results["runs"] if r["N"] == N and r["status"] == "success"
        ]
        errors = sum(1 for r in all_results["runs"] if r["N"] == N and r["status"] != "success")
        if times:
            all_results["summary"][str(N)] = {
                "count": len(times),
                "errors": errors,
                "min": float(np.min(times)),
                "max": float(np.max(times)),
                "mean": float(np.mean(times)),
                "median": float(np.median(times)),
                "p25": float(np.percentile(times, 25)),
                "p75": float(np.percentile(times, 75)),
                "std": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            }
        else:
            all_results["summary"][str(N)] = {
                "count": 0,
                "errors": errors,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "p25": None,
                "p75": None,
                "std": None,
            }

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Save CSV (flat table: one row per run; handy for Pandas/boxplots)
    fieldnames = ["N", "trial_index", "status", "time_sec", "K", "T", "h", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results["runs"]:
            w.writerow({k: r.get(k, None) for k in fieldnames})
    print(f"Saved CSV:  {csv_path}")

    # Print quick summary
    print("\nSummary (success-only times):")
    for N in cfg["Ns"]:
        s = all_results["summary"][str(N)]
        print(
            f"  N={N}: count={s['count']}, errors={s['errors']}, "
            f"mean={s['mean']}, median={s['median']}, p25={s['p25']}, p75={s['p75']}"
        )


if __name__ == "__main__":
    main()
