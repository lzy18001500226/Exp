import argparse
import json
import random
from pathlib import Path
import numpy as np
import re


def build_index(root: Path) -> list[str]:
    files = [str(p.resolve()) for p in root.rglob("*.npz")]
    files.sort()
    return files


def _class_id_from_path(npz_path: Path, root: Path) -> int | None:
    """Infer numeric class id from first-level directory under root.

    Assumes layout: <root>/<class_id>/<file>.npz
    Returns None if cannot parse an integer class id.
    """
    try:
        rel = npz_path.resolve().relative_to(root.resolve())
    except Exception:
        rel = npz_path
    parts = rel.parts
    if not parts:
        return None
    try:
        return int(parts[0])
    except Exception:
        return None


def _has_label_json(npz_path: Path) -> bool:
    try:
        with np.load(str(npz_path)) as z:
            if "meta" not in z:
                return False
            meta = json.loads(str(z["meta"]))
            jp = meta.get("label_json")
            return bool(jp) and Path(jp).exists()
    except Exception:
        return False


def _count_labels(npz_path: Path) -> int:
    try:
        with np.load(str(npz_path)) as z:
            if "meta" not in z:
                return 0
            meta = json.loads(str(z["meta"]))
            jp = meta.get("label_json")
            if not jp:
                return 0
        p = Path(jp)
        if not p.exists():
            return 0
        data = json.loads(p.read_text(encoding="utf-8"))
        # Case 1: plain list of boxes
        if isinstance(data, list):
            return len(data)
        # Case 2: dict with common keys
        if isinstance(data, dict):
            # Known container keys
            for k in ("boxes", "annotations", "objects"):
                if isinstance(data.get(k), list):
                    return len(data[k])
            # LabelMe format: {"shapes": [...]} with each shape carrying points/shape_type
            shapes = data.get("shapes")
            if isinstance(shapes, list):
                return len(shapes)
            # Single normalized box as dict
            keys = set(data.keys())
            if {"class_id", "x_center", "y_center", "width", "height"}.issubset(keys):
                return 1
        return 0
    except Exception:
        return 0


def _resolve_label_path_via_root(npz_path: Path, npz_root: Path, label_root: Path, label_suffix: str = ".json") -> Path | None:
    """Rebuild label path as <label_root>/<class>/<stem><suffix> based on npz location under npz_root."""
    try:
        rel = npz_path.resolve().relative_to(npz_root.resolve())
    except Exception:
        rel = npz_path
    parts = rel.parts
    if not parts:
        return None
    cls_dir = parts[0]
    stem = Path(parts[-1]).stem
    candidate = label_root / cls_dir / f"{stem}{label_suffix}"
    return candidate if candidate.exists() else None


def _has_label_json_via_root(npz_path: Path, npz_root: Path, label_root: Path, label_suffix: str = ".json") -> bool:
    p = _resolve_label_path_via_root(npz_path, npz_root, label_root, label_suffix)
    return p is not None


def _count_labels_via_root(npz_path: Path, npz_root: Path, label_root: Path, label_suffix: str = ".json") -> int:
    p = _resolve_label_path_via_root(npz_path, npz_root, label_root, label_suffix)
    if p is None or not p.exists():
        return 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # Case 1: plain list of boxes
        if isinstance(data, list):
            return len(data)
        # Case 2: dict with common keys
        if isinstance(data, dict):
            # Known container keys
            for k in ("boxes", "annotations", "objects"):
                if isinstance(data.get(k), list):
                    return len(data[k])
            # LabelMe format support
            shapes = data.get("shapes")
            if isinstance(shapes, list):
                return len(shapes)
            # Single normalized box as dict
            keys = set(data.keys())
            if {"class_id", "x_center", "y_center", "width", "height"}.issubset(keys):
                return 1
        return 0
    except Exception:
        return 0


_STEM_NUM_RE = re.compile(r"^(\d+)")


def _stem_number(npz_path: Path) -> int | None:
    """Extract leading integer from stem like '401-a' -> 401.

    Returns None if no leading integer is found.
    """
    stem = npz_path.stem
    m = _STEM_NUM_RE.match(stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Create 80/20 train/val JSON splits from NPZ root")
    ap.add_argument("npz_root", type=str, help="Root directory containing .npz files")
    ap.add_argument("out_dir", type=str, help="Directory to write train.json and val.json")
    ap.add_argument("--ratio", type=float, default=0.8, help="Train split ratio (default 0.8)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--require-label-json", action="store_true", help="Keep only samples with existing meta.label_json path")
    ap.add_argument("--min-labels", type=int, default=1, help="Require at least this many boxes when --require-label-json is set")
    ap.add_argument("--class-max", type=int, default=None, help="Keep only samples whose first-level folder numeric id <= this value")
    ap.add_argument("--class-min", type=int, default=None, help="Keep only samples whose first-level folder numeric id >= this value")
    ap.add_argument("--label-root", type=str, default=None, help="If provided, rebuild label path as <label_root>/<class>/<stem>.json for existence & count checks")
    ap.add_argument("--label-suffix", type=str, default=".json", help="Suffix to append when rebuilding label path (default .json)")
    ap.add_argument("--stem-min", type=int, default=None, help="Keep only files whose leading numeric stem >= this value (e.g., 0 for 0-a.npz)")
    ap.add_argument("--stem-max", type=int, default=None, help="Keep only files whose leading numeric stem <= this value (e.g., 200 for 200-a.npz)")
    ap.add_argument("--keep-classes", type=int, nargs="*", help="Only retain samples whose bbox classes are a subset of these ids")
    ap.add_argument("--allow-empty", action="store_true", help="When using --keep-classes, keep samples without boxes (default drops them)")
    ap.add_argument("--debug", action="store_true", help="Print debug info for first N filtered samples")
    ap.add_argument("--debug-limit", type=int, default=20, help="Max number of debug lines to print when --debug")
    args = ap.parse_args()

    root = Path(args.npz_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = build_index(root)
    if not files:
        raise SystemExit(f"No .npz files under {root}")

    # Class range filtering based on first-level folder numeric id (apply early)
    if args.class_min is not None or args.class_max is not None:
        before = len(files)
        filtered = []
        for s in files:
            cid = _class_id_from_path(Path(s), root)
            if cid is None:
                continue
            if args.class_min is not None and cid < args.class_min:
                continue
            if args.class_max is not None and cid > args.class_max:
                continue
            filtered.append(s)
        files = filtered
        after = len(files)
        print(f"[filter] class-range(min={args.class_min}, max={args.class_max}): kept {after}/{before}")

    # Stem numeric range filtering (apply early, after class filter)
    if args.stem_min is not None or args.stem_max is not None:
        before = len(files)
        filtered = []
        for s in files:
            num = _stem_number(Path(s))
            if num is None:
                continue
            if args.stem_min is not None and num < args.stem_min:
                continue
            if args.stem_max is not None and num > args.stem_max:
                continue
            filtered.append(s)
        files = filtered
        after = len(files)
        print(f"[filter] stem-range(min={args.stem_min}, max={args.stem_max}): kept {after}/{before}")

    if args.require_label_json:
        before = len(files)
        if args.label_root:
            lr = Path(args.label_root).resolve()
            kept = []
            dbg_n = 0
            for p in files:
                npz_p = Path(p)
                cand = _resolve_label_path_via_root(npz_p, root, lr, args.label_suffix)
                if cand is None:
                    if args.debug and dbg_n < args.debug_limit:
                        # Show what we expected path to be even if it doesn't exist
                        try:
                            rel = npz_p.resolve().relative_to(root.resolve())
                        except Exception:
                            rel = npz_p
                        parts = rel.parts
                        cls_dir = parts[0] if parts else "?"
                        stem = npz_p.stem
                        expected = lr / cls_dir / f"{stem}{args.label_suffix}"
                        print(f"[debug] missing label: npz={npz_p} expected={expected}")
                        dbg_n += 1
                    continue
                nbox = _count_labels_via_root(npz_p, root, lr, args.label_suffix)
                if nbox >= max(1, args.min_labels):
                    kept.append(p)
                else:
                    if args.debug and dbg_n < args.debug_limit:
                        print(f"[debug] too few boxes({nbox}): npz={npz_p} label={cand}")
                        dbg_n += 1
            files = kept
        else:
            files = [p for p in files if _has_label_json(Path(p)) and _count_labels(Path(p)) >= max(1, args.min_labels)]
        after = len(files)
        print(f"[filter] require-label-json(min_labels={args.min_labels}, use_label_root={bool(args.label_root)}): kept {after}/{before}")

    # optional class whitelist scan based on stored bboxes
    if args.keep_classes:
        allowed = {int(c) for c in args.keep_classes}
        before = len(files)
        kept: list[str] = []
        dropped_empty = 0
        dropped_other = 0
        dbg_n = 0
        for p in files:
            try:
                with np.load(p) as npz_obj:
                    boxes = np.asarray(npz_obj.get("bboxes", np.zeros((0, 5), dtype=np.float32)))
            except Exception as exc:  # noqa: BLE001
                if args.debug and dbg_n < args.debug_limit:
                    print(f"[debug] failed to read {p}: {exc}")
                    dbg_n += 1
                continue
            if boxes.size == 0:
                if args.allow_empty:
                    kept.append(p)
                else:
                    dropped_empty += 1
                continue
            cls_vals = set(int(round(c)) for c in boxes[:, 4].tolist())
            if cls_vals.issubset(allowed):
                kept.append(p)
            else:
                dropped_other += 1
                if args.debug and dbg_n < args.debug_limit:
                    unexpected = sorted(cls_vals - allowed)
                    print(f"[debug] drop mixed classes {unexpected} in {p}")
                    dbg_n += 1
        files = kept
        after = len(files)
        print(
            f"[filter] keep-classes={sorted(allowed)}: kept {after}/{before} (dropped_empty={dropped_empty}, dropped_other={dropped_other})"
        )

    random.seed(args.seed)
    random.shuffle(files)
    k = int(len(files) * args.ratio)
    train, val = files[:k], files[k:]

    with open(out_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(out_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(train)} train and {len(val)} val entries to {out_dir}")


if __name__ == "__main__":
    main()
