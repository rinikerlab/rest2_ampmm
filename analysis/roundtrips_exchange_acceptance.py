""" "
Copyright (C) 2026 ETH Zurich, Igor Gordiy, and other AMP contributors
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

TUPLE_RE = re.compile(r"\((\d+),\s*(\d+),\s*(True|False)\)")
ITER_RE = re.compile(r"Iteration\s+(\d+):")


def parse_log_file(filename: Path) -> list[list[tuple[int, int, bool]]]:
    """Parse tuple entries from a REST2 log file."""
    iterations = []
    with filename.open("r") as handle:
        for line in handle:
            tuples = TUPLE_RE.findall(line)
            if tuples:
                iterations.append([(int(a), int(b), c == "True") for a, b, c in tuples])
    return iterations


def max_iteration_index(filename: Path) -> int:
    """Return the number of iterations based on the maximum 'Iteration N' value."""
    maximum = -1
    with filename.open("r") as handle:
        for line in handle:
            match = ITER_RE.search(line)
            if match:
                maximum = max(maximum, int(match.group(1)))
    return maximum + 1 if maximum >= 0 else 0


def iter_logfiles(base_path: Path, n_steps: int) -> list[Path]:
    """Return existing REST2 log files under the expected segment layout."""
    files = []
    for segment in range(n_steps):
        logfile = base_path / f"rest2_0000_{segment:04d}" / "rest2.log"
        if logfile.is_file():
            files.append(logfile)
    return files


def calculate_average_percentage(
    iterations: Sequence[Sequence[tuple[int, int, bool]]],
) -> float:
    """Compute the mean global acceptance percentage across iteration blocks."""
    percentages = []
    for iteration in iterations:
        if not iteration:
            continue
        total = len(iteration)
        accepted = sum(1 for _, _, accepted_flag in iteration if accepted_flag)
        percentages.append(100.0 * accepted / total)
    return float(np.mean(percentages)) if percentages else 0.0


def compute_acceptance_by_pair(
    all_iterations: Sequence[Sequence[tuple[int, int, bool]]],
) -> dict[tuple[int, int], float]:
    """Compute mean acceptance percentage for each replica pair."""
    counts = defaultdict(lambda: {"acc": 0, "tot": 0})
    for iteration in all_iterations:
        for i, j, accepted in iteration:
            pair = tuple(sorted((i, j)))
            counts[pair]["tot"] += 1
            counts[pair]["acc"] += int(accepted)

    return {
        pair: 100.0 * values["acc"] / values["tot"]
        for pair, values in counts.items()
        if values["tot"] > 0
    }


def count_roundtrips_single(
    traj: Sequence[int], min_idx: int = 0, max_idx: Optional[int] = None
) -> int:
    """
    Count full roundtrips for one walker in replica-index space.

    A roundtrip is:
    - min -> max -> min
    or
    - max -> min -> max
    """
    if not traj:
        return 0

    if max_idx is None:
        max_idx = max(traj)

    ends = {min_idx, max_idx}
    state = None
    origin = None
    reached_other_end = False
    roundtrips = 0

    for index in traj:
        if state is None:
            if index in ends:
                state = "at_end"
                origin = index
            continue

        if not reached_other_end:
            if origin == min_idx and index == max_idx:
                reached_other_end = True
            elif origin == max_idx and index == min_idx:
                reached_other_end = True
        else:
            if index == origin:
                roundtrips += 1
                reached_other_end = False

    return roundtrips


def reconstruct_trajectories(
    all_iterations: Sequence[Sequence[tuple[int, int, bool]]],
) -> list[list[int]]:
    """Reconstruct walker trajectories in replica-index space."""
    if not all_iterations:
        return []

    max_index_seen = max(
        max(i, j) for iteration in all_iterations for i, j, _ in iteration
    )
    n_replicas = max_index_seen + 1

    replica_of_walker = list(range(n_replicas))
    walker_at_replica = list(range(n_replicas))
    trajectories = [[] for _ in range(n_replicas)]

    for iteration in all_iterations:
        for i, j, accepted in iteration:
            if accepted:
                walker_i = walker_at_replica[i]
                walker_j = walker_at_replica[j]
                walker_at_replica[i], walker_at_replica[j] = walker_j, walker_i
                replica_of_walker[walker_i], replica_of_walker[walker_j] = j, i

        for walker in range(n_replicas):
            trajectories[walker].append(replica_of_walker[walker])

    return trajectories


@dataclass
class RunInfo:
    """Metadata describing available and analyzed simulation time."""

    n_logs_found: int
    total_iters_available: int
    total_time_ns_available: float
    analyzed_iters: int
    analyzed_time_ns: float
    requested_time_ns: Optional[float]


def estimate_sim_time_ns(
    total_iters: int, steps_per_iter: int = 100, ps_per_step: float = 0.001
) -> float:
    """Estimate total simulation time in ns from iteration count."""
    return total_iters * steps_per_iter * ps_per_step / 1000.0


def iterations_for_time_ns(
    time_ns: float, steps_per_iter: int = 100, ps_per_step: float = 0.001
) -> int:
    """Return the number of iterations that fit within the requested time."""
    ns_per_iter = steps_per_iter * ps_per_step / 1000.0
    return int(time_ns / ns_per_iter)


def load_all_iterations(
    base_path: Path,
    n_steps: int,
    max_time_ns: Optional[float] = None,
) -> tuple[list[list[tuple[int, int, bool]]], RunInfo]:
    """Load all parsed iteration data across REST2 log files."""
    logfiles = iter_logfiles(base_path, n_steps)
    if not logfiles:
        expected = base_path / "rest2_0000_0000" / "rest2.log"
        raise FileNotFoundError(
            f"No rest2.log files found under {base_path} for n_steps={n_steps}. "
            f"Expected example path: {expected}"
        )

    total_iters_available = sum(max_iteration_index(logfile) for logfile in logfiles)
    total_time_ns_available = estimate_sim_time_ns(total_iters_available)

    target_iters = None
    if max_time_ns is not None:
        target_iters = iterations_for_time_ns(max_time_ns)

    all_iterations = []
    for logfile in logfiles:
        parsed = parse_log_file(logfile)

        if target_iters is None:
            all_iterations.extend(parsed)
            continue

        remaining = target_iters - len(all_iterations)
        if remaining <= 0:
            break

        all_iterations.extend(parsed[:remaining])

    analyzed_iters = len(all_iterations)
    analyzed_time_ns = estimate_sim_time_ns(analyzed_iters)

    runinfo = RunInfo(
        n_logs_found=len(logfiles),
        total_iters_available=total_iters_available,
        total_time_ns_available=total_time_ns_available,
        analyzed_iters=analyzed_iters,
        analyzed_time_ns=analyzed_time_ns,
        requested_time_ns=max_time_ns,
    )
    return all_iterations, runinfo


def print_acceptance(
    global_rate: float, pair_rates: dict[tuple[int, int], float], runinfo: RunInfo
) -> None:
    """Print acceptance-rate summary to stdout."""
    print(f"\nGlobal acceptance rate: {global_rate:.2f}%")
    print(
        f"Analyzed time         : {runinfo.analyzed_time_ns:.3f} ns ({runinfo.analyzed_iters} iteration blocks)"
    )
    if runinfo.requested_time_ns is not None:
        print(f"Requested time        : {runinfo.requested_time_ns:.3f} ns")
    print(
        f"Available time        : {runinfo.total_time_ns_available:.3f} ns "
        f"({runinfo.total_iters_available} iterations from Iteration lines)\n"
    )

    print("Acceptance rate per replica pair:")
    for pair, rate in sorted(pair_rates.items()):
        print(f"  Pair {pair}: {rate:.2f}%")


def print_roundtrips(roundtrip_counts: Sequence[int], analyzed_time_ns: float) -> None:
    """Print roundtrip summary to stdout."""
    if not roundtrip_counts:
        print("\nNo roundtrip data available.")
        return

    values = np.array(roundtrip_counts, dtype=float)
    mean_value = values.mean()

    print("\nRoundtrip statistics:")
    for walker, count in enumerate(roundtrip_counts):
        print(f"  Walker {walker:2d}: {count} roundtrips")

    print(f"\nMean roundtrips per walker : {mean_value:.2f}")
    print(f"Min / Max roundtrips       : {int(values.min())} / {int(values.max())}")
    if analyzed_time_ns > 1e-12:
        print(
            f"Mean roundtrips per ns     : {mean_value / analyzed_time_ns:.2f} (per walker)"
        )


def plot_roundtrips(
    roundtrip_counts: Sequence[int],
    trajectories: Sequence[Sequence[int]],
    plot_walker: int = 0,
) -> None:
    """Plot roundtrip counts and one walker trajectory."""
    if not trajectories:
        return

    n_replicas = len(trajectories)
    if plot_walker < 0 or plot_walker >= n_replicas:
        plot_walker = 0

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

    mean_roundtrips = float(np.mean(roundtrip_counts)) if roundtrip_counts else 0.0

    axes[0].bar(
        np.arange(n_replicas), roundtrip_counts, edgecolor="black", linewidth=0.7
    )
    axes[0].axhline(mean_roundtrips, linestyle="--", linewidth=1)
    axes[0].set_xlabel("Walker ID")
    axes[0].set_ylabel("Roundtrips")
    axes[0].set_title("Full ladder roundtrips per walker")
    axes[0].text(
        0.02, 0.9, f"mean = {mean_roundtrips:.2f}", transform=axes[0].transAxes
    )

    axes[1].plot(
        np.arange(len(trajectories[plot_walker])),
        trajectories[plot_walker],
        linewidth=0.8,
    )
    axes[1].set_xlabel("Iteration block")
    axes[1].set_ylabel("Replica index")
    axes[1].set_title(f"Replica index vs iteration block for walker {plot_walker}")
    axes[1].set_yticks(np.arange(n_replicas))
    axes[1].set_ylim(-0.5, n_replicas - 0.5)

    plt.tight_layout()
    plt.show()


def export_pair_rates_csv(
    pair_rates: dict[tuple[int, int], float], output_csv: Path
) -> None:
    """Write per-pair acceptance rates to CSV."""
    import csv

    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["replica_i", "replica_j", "acceptance_percent"])
        for (i, j), rate in sorted(pair_rates.items()):
            writer.writerow([i, j, f"{rate:.6f}"])


def export_summary_json(
    output_json: Path,
    global_rate: float,
    pair_rates: dict[tuple[int, int], float],
    roundtrip_counts: Optional[Sequence[int]],
    runinfo: RunInfo,
) -> None:
    """Write a JSON summary of the analysis."""
    payload = {
        "global_acceptance_percent": global_rate,
        "pair_acceptance_percent": {
            f"{i}-{j}": rate for (i, j), rate in sorted(pair_rates.items())
        },
        "roundtrips_per_walker": list(roundtrip_counts)
        if roundtrip_counts is not None
        else None,
        "n_logs_found": runinfo.n_logs_found,
        "total_iters_available": runinfo.total_iters_available,
        "total_time_ns_available": runinfo.total_time_ns_available,
        "analyzed_iters": runinfo.analyzed_iters,
        "analyzed_time_ns": runinfo.analyzed_time_ns,
        "requested_time_ns": runinfo.requested_time_ns,
    }

    with output_json.open("w") as handle:
        json.dump(payload, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Analyze REST2 exchange logs for acceptance rates and roundtrips."
    )
    parser.add_argument(
        "--molecule", required=True, help="Name of the analyzed molecule."
    )
    parser.add_argument(
        "path", type=Path, help="Base directory containing rest2_0000_####/rest2.log."
    )
    parser.add_argument(
        "--n-steps", required=True, type=int, help="Number of segments to scan."
    )
    parser.add_argument(
        "--mode",
        choices=["acceptance", "roundtrips", "both"],
        default="both",
        help="Analysis mode.",
    )
    parser.add_argument(
        "--max-time-ns", type=float, default=None, help="Analyze only the first N ns."
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show plots for roundtrip analysis."
    )
    parser.add_argument(
        "--plot-walker", type=int, default=0, help="Walker index to plot."
    )
    parser.add_argument(
        "--out-pairs-csv",
        type=Path,
        default=None,
        help="Output CSV for per-pair acceptance.",
    )
    parser.add_argument(
        "--out-json", type=Path, default=None, help="Output JSON summary."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI."""
    args = build_parser().parse_args(argv)

    all_iterations, runinfo = load_all_iterations(
        args.path, args.n_steps, max_time_ns=args.max_time_ns
    )
    if not all_iterations:
        print("No '(i, j, True|False)' iteration data found in the logs.")
        return 2

    global_rate = calculate_average_percentage(all_iterations)
    pair_rates = compute_acceptance_by_pair(all_iterations)

    roundtrip_counts = None
    trajectories = None

    print(f"Analyzing molecule: {args.molecule}")

    if args.mode in ("acceptance", "both"):
        print_acceptance(global_rate, pair_rates, runinfo)

    if args.mode in ("roundtrips", "both"):
        trajectories = reconstruct_trajectories(all_iterations)
        if trajectories:
            max_replica = len(trajectories) - 1
            roundtrip_counts = [
                count_roundtrips_single(traj, min_idx=0, max_idx=max_replica)
                for traj in trajectories
            ]
        else:
            roundtrip_counts = []

        print_roundtrips(roundtrip_counts, runinfo.analyzed_time_ns)

        if args.plot:
            plot_roundtrips(
                roundtrip_counts, trajectories, plot_walker=args.plot_walker
            )

    if args.out_pairs_csv:
        args.out_pairs_csv.parent.mkdir(parents=True, exist_ok=True)
        export_pair_rates_csv(pair_rates, args.out_pairs_csv)
        print(f"\nWrote per-pair acceptance CSV: {args.out_pairs_csv}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        export_summary_json(
            args.out_json, global_rate, pair_rates, roundtrip_counts, runinfo
        )
        print(f"Wrote summary JSON: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
