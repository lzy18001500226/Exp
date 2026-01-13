import argparse
import json
from pathlib import Path
import re
from typing import List, Optional


def parse_args():
    p = argparse.ArgumentParser("Preview NPZ->(FCS/VTS) JSON mapping and shapes counts")
    p.add_argument('--list_file', type=str, required=True, help='Text file of NPZ paths (one per line)')
    p.add_argument('--fcs_root', type=str, default='C:/Users/HP/Desktop/Exp/FCSLabel')
    p.add_argument('--vts_root', type=str, default='C:/Users/HP/Desktop/Exp/VTSLabel')
    p.add_argument('--max_items', type=int, default=20)
    return p.parse_args()


def _stem_num(stem: str) -> Optional[int]:
    m = re.match(r'^(\d+)', stem)
    return int(m.group(1)) if m else None


def _read_list(list_file: Path) -> List[Path]:
    items: List[Path] = []
    for line in list_file.read_text(encoding='utf-8').splitlines():
        s = line.strip().strip('"')
        if not s:
            continue
        p = Path(s)
        if p.suffix.lower() == '.npz':
            items.append(p)
    return items


def _json_shapes_count(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        shapes = data.get('shapes', [])
        return len(shapes) if isinstance(shapes, list) else 0
    except Exception:
        return 0


def main():
    args = parse_args()
    lst = Path(args.list_file)
    if not lst.exists():
        print(f"[ERR] list_file not found: {lst}")
        return
    fcs_root = Path(args.fcs_root)
    vts_root = Path(args.vts_root)

    npz_paths = _read_list(lst)
    print(f"Loaded {len(npz_paths)} NPZ paths from {lst}")

    for i, npz in enumerate(npz_paths[: args.max_items]):
        cls_id = npz.parent.name
        stem = npz.stem
        fcs_json = fcs_root / cls_id / f"{stem}.json"
        vts_json = vts_root / cls_id / f"{stem}.json"
        fcs_n = _json_shapes_count(fcs_json) if fcs_json.exists() else -1
        vts_n = _json_shapes_count(vts_json) if vts_json.exists() else -1
        # preferred according to "prefer existing and non-empty"
        preferred = None
        if vts_n > 0:
            preferred = 'VTSLabel'
        elif fcs_n > 0:
            preferred = 'FCSLabel'
        else:
            preferred = None
        print(f"[{i+1:02d}] NPZ={npz}")
        print(f"      FCS={fcs_json} exist={fcs_json.exists()} shapes={fcs_n if fcs_n>=0 else 'NA'}")
        print(f"      VTS={vts_json} exist={vts_json.exists()} shapes={vts_n if vts_n>=0 else 'NA'}")
        print(f"      preferred={preferred}")


if __name__ == '__main__':
    main()
