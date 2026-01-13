import argparse
from pathlib import Path
import json
import numpy as np
import re
from collections import defaultdict

DEFAULT_LABEL_ROOT = Path(r"D:\Exp\FCSLabel")
STEM_NUM_RE = re.compile(r"^(\d+)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze label JSONs to compute per-class average box sizes and anchor suggestions")
    ap.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    ap.add_argument("--classes", type=int, nargs="*", default=None, help="Optional list of class ids to include (default: all numeric dirs under label-root)")
    ap.add_argument("--stem-min", type=int, default=0)
    ap.add_argument("--stem-max", type=int, default=999999)
    ap.add_argument("--subset-json", type=Path, default=None, help="Optional JSON (train/val list) used to filter by class/stem from converted NPZ paths")
    ap.add_argument("--topk", type=int, default=5, help="Anchors per scale when grouping (default 5 -> total 15)")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to save per-class stats as JSON")
    return ap.parse_args()


def load_subset_filter(subset_path: Path | None) -> set[tuple[int, str]] | None:
    if not subset_path:
        return None
    if not subset_path.exists():
        raise FileNotFoundError(subset_path)
    with open(subset_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    allowed: set[tuple[int, str]] = set()
    for entry in items:
        stem = Path(entry)
        try:
            cid = int(stem.parent.name)
        except ValueError:
            continue
        name = stem.stem
        m = STEM_NUM_RE.match(name)
        if not m:
            continue
        allowed.add((cid, m.group(1)))
    return allowed


def iter_label_jsons(root: Path, classes: set[int] | None, stem_min: int, stem_max: int, subset: set[tuple[int, str]] | None):
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            cid = int(cdir.name)
        except ValueError:
            continue
        if classes is not None and cid not in classes:
            continue
        for jp in cdir.glob("*.json"):
            stem = jp.stem
            m = STEM_NUM_RE.match(stem)
            if not m:
                continue
            num = int(m.group(1))
            if num < stem_min or num > stem_max:
                continue
            if subset is not None and (cid, m.group(1)) not in subset:
                continue
            yield cid, m.group(1), jp


def rect_to_wh(shape) -> tuple[float, float] | None:
    pts = shape.get("points")
    if not (isinstance(pts, list) and len(pts) >= 2):
        return None
    x1, y1 = float(pts[0][0]), float(pts[0][1])
    x2, y2 = float(pts[1][0]), float(pts[1][1])
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    if w <= 0 or h <= 0:
        return None
    return w, h


def format_anchor_list(pairs):
    return ",".join([f"{int(round(w))}x{int(round(h))}" for w, h in pairs])


def main() -> None:
    args = parse_args()
    subset_filter = load_subset_filter(args.subset_json)
    class_set = set(args.classes) if args.classes else None

    per_class_stats = defaultdict(lambda: {"width_sum": 0.0, "height_sum": 0.0, "count": 0})
    all_wh: list[tuple[float, float, int]] = []
    file_hits = 0

    for cid, stem_num, jp in iter_label_jsons(args.label_root, class_set, args.stem_min, args.stem_max, subset_filter):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        shapes = data.get("shapes", [])
        saw_box = False
        for shp in shapes:
            st = (shp.get("shape_type") or "").lower()
            if st not in ("", "rectangle", "rect"):
                continue
            wh = rect_to_wh(shp)
            if wh is None:
                continue
            w, h = wh
            per_class_stats[cid]["width_sum"] += w
            per_class_stats[cid]["height_sum"] += h
            per_class_stats[cid]["count"] += 1
            all_wh.append((w, h, cid))
            saw_box = True
        if saw_box:
            file_hits += 1

    if not all_wh:
        print("[warn] No boxes found for the selected subset/classes")
        return

    print(f"Analyzed {len(all_wh)} boxes from {file_hits} label files.\n")
    header = f"{'Class':>5}  {'Boxes':>6}  {'AvgW':>8}  {'AvgH':>8}"
    print(header)
    print("-" * len(header))
    per_class_table = []
    for cid in sorted(per_class_stats.keys()):
        stats = per_class_stats[cid]
        if stats["count"] == 0:
            continue
        avg_w = stats["width_sum"] / stats["count"]
        avg_h = stats["height_sum"] / stats["count"]
        per_class_table.append((cid, stats["count"], avg_w, avg_h))
        print(f"{cid:5d}  {stats['count']:6d}  {avg_w:8.2f}  {avg_h:8.2f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "classes": [
                        {"class_id": cid, "boxes": cnt, "avg_w": avg_w, "avg_h": avg_h}
                        for cid, cnt, avg_w, avg_h in per_class_table
                    ]
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nSaved per-class stats to {args.out}")

    # Build anchor suggestions based on per-class averages (Table II style)
    per_class_pairs = [(avg_w, avg_h) for _, _, avg_w, avg_h in per_class_table]
    per_class_pairs.sort(key=lambda x: x[0] * x[1])
    k = max(1, int(args.topk))
    total_needed = 3 * k
    if len(per_class_pairs) < total_needed:
        # fallback: duplicate last to reach desired count
        while len(per_class_pairs) < total_needed:
            per_class_pairs.append(per_class_pairs[-1])

    anchors_p2 = per_class_pairs[:k]
    anchors_p3 = per_class_pairs[k : 2 * k]
    anchors_p4 = per_class_pairs[2 * k : 3 * k]

    print("\nSuggested anchor groupings (sorted by avg area):")
    print(f"p2 (stride 8):  {format_anchor_list(anchors_p2)}")
    print(f"p3 (stride 16): {format_anchor_list(anchors_p3)}")
    print(f"p4 (stride 32): {format_anchor_list(anchors_p4)}")


if __name__ == "__main__":
    main()
