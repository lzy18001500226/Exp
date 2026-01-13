"""Summarize batch sweep outputs produced by analyze_color_channels.py.

This helper script aggregates the per-group top configurations and prints
useful statistics (parameter frequency, mean metrics, etc.).

Example:
    python summarize_sweeps.py --summary sweeps/summary.csv --per-group sweeps/per_group
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence


NUMERIC_FIELDS: tuple[str, ...] = (
    "edge_snr_mean",
    "edge_snr_std",
    "edge_contrast_mean",
    "edge_contrast_std",
    "edge_activation_total_mean",
    "texture_snr_mean",
    "texture_snr_std",
    "texture_contrast_mean",
    "texture_contrast_std",
    "texture_activation_total_mean",
    "corner_clean_snr_mean",
    "corner_clean_snr_std",
    "corner_clean_contrast_mean",
    "corner_clean_contrast_std",
    "corner_clean_activation_total_mean",
)

CATEGORICAL_FIELDS: tuple[str, ...] = (
    "edge_method",
    "edge_lambda_low",
    "edge_lambda_high",
    "edge_threshold",
    "edge_max_coverage",
    "texture_method",
    "texture_lambda_low",
    "texture_lambda_high",
    "texture_threshold",
    "texture_max_coverage",
    "corner_eta_ratio",
    "corner_mask_threshold",
    "corner_min_coverage",
)


def _load_csv(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for record in reader:
            parsed: dict[str, float | str] = {}
            for key, value in record.items():
                if value is None or value == "":
                    parsed[key] = math.nan
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _summarize_numeric(rows: Sequence[dict[str, float | str]]) -> dict[str, tuple[float, float]]:
    summary: dict[str, tuple[float, float]] = {}
    for field in NUMERIC_FIELDS:
        values = [row[field] for row in rows if isinstance(row.get(field), (int, float))]
        numeric = [float(v) for v in values if not math.isnan(float(v))]
        if numeric:
            if len(numeric) > 1:
                summary[field] = (mean(numeric), stdev(numeric))
            else:
                summary[field] = (numeric[0], float("nan"))
    return summary


def _summarize_categorical(rows: Sequence[dict[str, float | str]]) -> dict[str, Counter[str]]:
    freq: dict[str, Counter[str]] = {}
    for field in CATEGORICAL_FIELDS:
        counter: Counter[str] = Counter()
        for row in rows:
            value = row.get(field)
            if isinstance(value, str):
                counter[value] += 1
            elif isinstance(value, (int, float)) and not math.isnan(value):
                counter[f"{value:g}"] += 1
        if counter:
            freq[field] = counter
    return freq


def _load_per_group_counts(directory: Path, top_k: int) -> dict[str, Counter[str]]:
    """Count how often parameter settings appear within top-k rows per group."""
    counters: dict[str, Counter[str]] = {field: Counter() for field in CATEGORICAL_FIELDS}
    if not directory.exists():
        return counters

    for csv_path in sorted(directory.glob("*.csv")):
        rows = _load_csv(csv_path)
        if not rows:
            continue
        for row in rows[:top_k]:
            for field in CATEGORICAL_FIELDS:
                value = row.get(field)
                if isinstance(value, str):
                    counters[field][value] += 1
                elif isinstance(value, (int, float)) and not math.isnan(value):
                    counters[field][f"{value:g}"] += 1
    return counters


def _print_counter_table(title: str, counter: Counter[str], limit: int = 5) -> None:
    print(f"\n{title}")
    print("  value               count")
    for value, count in counter.most_common(limit):
        print(f"  {value:<18} {count:>5d}")


def _print_numeric_summary(summary: dict[str, tuple[float, float]]) -> None:
    if not summary:
        return
    print("\nMetric averages (mean ± stdev):")
    for field, (avg, sigma) in sorted(summary.items()):
        if math.isnan(sigma):
            print(f"  {field}: {avg:.4f}")
        else:
            print(f"  {field}: {avg:.4f} ± {sigma:.4f}")


def _print_categorical_summary(freq: dict[str, Counter[str]], limit: int) -> None:
    for field, counter in sorted(freq.items()):
        _print_counter_table(field, counter, limit=limit)


def _collect_best_by_group(summary_rows: Iterable[dict[str, float | str]]) -> dict[str, dict[str, float | str]]:
    best: dict[str, dict[str, float | str]] = {}
    for row in summary_rows:
        group = row.get("group")
        if not isinstance(group, str):
            continue
        best[group] = row
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("sweeps/summary.csv"),
        help="Path to the cross-group summary CSV (default: sweeps/summary.csv).",
    )
    parser.add_argument(
        "--per-group",
        type=Path,
        default=Path("sweeps/per_group"),
        help="Directory containing per-group sweep CSV files (default: sweeps/per_group).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top entries per group to include when counting parameter frequencies (default: 3).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many of the most common parameter values to display per field (default: 5).",
    )
    args = parser.parse_args()

    if not args.summary.exists():
        raise SystemExit(f"summary file not found: {args.summary}")

    summary_rows = _load_csv(args.summary)
    if not summary_rows:
        raise SystemExit("summary CSV is empty; run the batch sweep first.")

    best_by_group = _collect_best_by_group(summary_rows)
    print(f"Loaded {len(best_by_group)} groups from {args.summary}")

    numeric_stats = _summarize_numeric(summary_rows)
    _print_numeric_summary(numeric_stats)

    top1_freq = _summarize_categorical(summary_rows)
    print("\nTop-1 parameter frequencies across groups:")
    _print_categorical_summary(top1_freq, args.limit)

    per_group_dir = args.per_group
    if per_group_dir.exists():
        topk_counts = _load_per_group_counts(per_group_dir, args.top_k)
        print(f"\nTop-{args.top_k} parameter frequencies across all groups:")
        _print_categorical_summary(topk_counts, args.limit)
    else:
        print(f"Per-group directory not found: {per_group_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
