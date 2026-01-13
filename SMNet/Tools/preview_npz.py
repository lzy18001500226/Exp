import argparse
import json
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _list_npz_from_dir(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")
    return sorted(p for p in directory.rglob("*.npz") if p.is_file())


def _load_paths(index_input: str | None, dir_input: str | None) -> list[Path]:
    paths: list[Path] = []

    if dir_input:
        paths.extend(_list_npz_from_dir(Path(dir_input)))

    if index_input:
        index_path = Path(index_input)
        if index_path.is_dir():
            paths.extend(_list_npz_from_dir(index_path))
        elif index_path.suffix.lower() == ".json":
            with open(index_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            paths.extend(Path(s) for s in items if str(s).endswith(".npz"))
        elif index_path.suffix.lower() == ".npz":
            paths.append(index_path)
        else:
            raise SystemExit(
                "index_json argument must be a JSON list, directory, or NPZ file; "
                f"got {index_path}"
            )

    if not paths:
        raise SystemExit("No .npz files found. Provide a JSON index, directory, or single NPZ path.")

    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[Path] = []
    for p in paths:
        key = str(Path(p).resolve())
        if key in seen:
            continue
        seen.add(key)
        ordered.append(Path(key))
    return ordered


def _ensure_chw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        # Treat channel-last tensors by moving channels to the front.
        if arr.shape[0] not in {1, 3} and arr.shape[-1] in {1, 3}:
            arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported tensor rank: {arr.shape}")

    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] < 3:
        arr = np.pad(arr, ((0, 3 - arr.shape[0]), (0, 0), (0, 0)), mode="edge")
    elif arr.shape[0] > 3:
        arr = arr[:3]
    return arr.astype(np.float32)


def _extract_triplet(bundle: np.lib.npyio.NpzFile) -> np.ndarray:
    preferred_keys: tuple[str, ...] = (
        "golden_triplet",
        "X",
        "features",
    )
    arr: np.ndarray | None = None
    for key in preferred_keys:
        if key in bundle.files:
            arr = bundle[key]
            break
    if arr is None:
        if "image" in bundle.files:
            arr = bundle["image"]
        else:
            raise KeyError("NPZ missing golden_triplet/X/image keys")
    return _ensure_chw(arr)


def main():
    ap = argparse.ArgumentParser(
        description="Preview NPZ files (JSON list, directory, or single file) and optionally save composite PNGs."
    )
    ap.add_argument(
        "index_json",
        nargs="?",
        type=str,
        help="JSON list of .npz files, directory path, or single NPZ file",
    )
    ap.add_argument("--dir", type=str, default=None, help="Directory to recursively search for .npz files")
    ap.add_argument("--max", type=int, default=4, help="Max samples to preview")
    ap.add_argument("--out", type=str, default=None, help="Optional output PNG path (requires Pillow)")
    ap.add_argument("--check-labels", action="store_true", help="Verify that all samples have existing meta.label_json")
    args = ap.parse_args()

    paths = _load_paths(args.index_json, args.dir)

    H, W = 512, 512
    if args.check_labels:
        missing = []
        for p in paths:
            try:
                with np.load(str(p)) as z:
                    meta = json.loads(str(z["meta"])) if "meta" in z else {}
                jp = meta.get("label_json")
                if not jp or not Path(jp).exists():
                    missing.append(str(p))
            except Exception:
                missing.append(str(p))
        if missing:
            print(f"[labels] missing: {len(missing)}")
            for m in missing[:50]:
                print("  ", m)
        else:
            print("[labels] all present")
    tiles = []
    for i, p in enumerate(paths[: args.max]):
        with np.load(str(p)) as z:
            X = _extract_triplet(z)
            meta = json.loads(str(z["meta"])) if "meta" in z else {}
        print(f"[{i}] {p}")
        print(f"  shape: {X.shape}, S[min,max]=({X[0].min():.3f},{X[0].max():.3f})")
        print(f"  edge unique: {np.unique(X[1]).tolist()}")
        print(f"  corner unique: {np.unique(X[2]).tolist()}")
        if PIL_AVAILABLE and args.out:
            # Stack S, Edge, Corner horizontally
            s = to_uint8(X[0])
            e = to_uint8(X[1])
            c = to_uint8(X[2])
            comp = np.concatenate([s, e, c], axis=1)
            tiles.append(comp)

    if PIL_AVAILABLE and args.out and tiles:
        grid = np.concatenate(tiles, axis=0)
        Image.fromarray(grid).save(args.out)
        print(f"Saved composite preview to {args.out}")
    elif args.out and not PIL_AVAILABLE:
        print("Pillow not available; skipping image save.")


if __name__ == "__main__":
    main()
