"""Generate JSON index files listing .npz samples for NPZDataset.

Usage (PowerShell):
  python scripts/build_npz_index.py --root data_split/train --out data_split/train.json
  python scripts/build_npz_index.py --root data_split/test  --out data_split/val.json

It scans recursively for .npz and writes a JSON list of absolute paths. Each .npz is expected
to contain an array 'X' or 'image' and optional 'bboxes' or 'meta' with 'label_json' pointing to the label file.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

def scan_npz(root: Path, require_label: bool) -> list[str]:
    """Scan directory for .npz files, optionally filtering by label presence."""
    out: list[str] = []
    for fp in root.rglob("*.npz"):
        try:
            if require_label:
                # quick label presence check without loading full array
                import numpy as np
                with np.load(str(fp)) as z:
                    # Check for new format (bboxes) or old format (meta.label_json)
                    has_bboxes = 'bboxes' in z and z['bboxes'].size > 0
                    has_meta_label = False
                    if 'meta' in z:
                        try:
                            meta_raw = str(z['meta'])
                            meta = json.loads(meta_raw)
                            lj = meta.get('label_json')
                            has_meta_label = lj and Path(lj).exists()
                        except Exception:
                            pass
                    if not (has_bboxes or has_meta_label):
                        continue
            out.append(str(fp.resolve()))
        except Exception:
            continue
    return sorted(out)

def main():
    ap = argparse.ArgumentParser("Build NPZ index JSON")
    ap.add_argument('--root', type=str, required=True, help='Root directory to scan recursively for .npz files')
    ap.add_argument('--out', type=str, required=True, help='Output JSON file path (will overwrite)')
    ap.add_argument('--require-label', action='store_true', help='Skip npz without bboxes or meta.label_json present')
    ap.add_argument('--max', type=int, default=0, help='Optional: limit number of files (0=no limit)')
    args = ap.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    
    print(f"Scanning {root}...")
    files = scan_npz(root, args.require_label)
    
    if args.max > 0:
        files = files[:args.max]
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(files, f, ensure_ascii=False, indent=2)
    
    print(f"[index] wrote {len(files)} entries -> {args.out}")

if __name__ == '__main__':
    main()
