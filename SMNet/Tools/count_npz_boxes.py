"""Count how many NPZ samples contain bounding boxes per class folder.

This script replaces the legacy copy under ``SMNet/Test`` so that all
one-off data sanity checks live inside ``Tools``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("D:/Exp/SMNet/FCSData"),
        help="Root directory that holds per-class folders with NPZ files.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many class counts to display in the summary table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root

    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    per_class_counts: Counter[str] = Counter()
    total_files = 0
    files_with_boxes = 0

    for npz_path in root.rglob("*.npz"):
        total_files += 1
        try:
            with np.load(npz_path, allow_pickle=True) as bundle:
                boxes = bundle.get("bboxes")
                if boxes is None:
                    continue
                if np.asarray(boxes).size == 0:
                    continue
                files_with_boxes += 1
                per_class_counts[npz_path.parent.name] += np.asarray(boxes).shape[0]
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to read {npz_path}: {exc}")

    print(f"Total NPZ files: {total_files}")
    print(f"Files with at least one box: {files_with_boxes}")
    print(f"Classes containing boxes: {len(per_class_counts)}")

    for cls, count in per_class_counts.most_common(args.top):
        print(f"  {cls}: {count} boxes")


if __name__ == "__main__":
    main()
