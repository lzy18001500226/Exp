import argparse
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


def parse_args():
    p = argparse.ArgumentParser("Label JSON box size statistics and small-box ratio")
    p.add_argument('--label_root', type=str, required=False, help='Root of label dir (e.g., .../FCSLabel or .../VTSLabel)')
    p.add_argument('--list_file', type=str, required=False, help='Optional: text file of JSON paths (one per line)')
    p.add_argument('--out', type=str, default='', help='Where to save stats report JSON')
    p.add_argument('--class_min', type=int, default=0)
    p.add_argument('--class_max', type=int, default=23)
    p.add_argument('--include_classes', type=str, default='', help='Comma-separated class ids to include (empty = all in range)')
    p.add_argument('--exclude_classes', type=str, default='', help='Comma-separated class ids to exclude')
    p.add_argument('--stem_min', type=int, default=0)
    p.add_argument('--stem_max', type=int, default=10**9)
    p.add_argument('--thresholds', type=str, default='4,8,12', help='Comma-separated pixel thresholds for small-box ratios (w<h or h<w independent)')
    return p.parse_args()


def _parse_id_list(s: str) -> Set[int]:
    out: Set[int] = set()
    if s and s.strip():
        for tok in s.split(','):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.add(int(tok))
            except ValueError:
                pass
    return out


def _stem_num(stem: str) -> Optional[int]:
    m = re.match(r'^(\d+)', stem)
    return int(m.group(1)) if m else None


def _iter_json_files(root: Path, include_classes: List[int], smin: int, smax: int):
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name)):
        cid = int(cls_dir.name)
        if cid not in include_classes:
            continue
        for f in sorted(cls_dir.glob('*.json')):
            sn = _stem_num(f.stem)
            if sn is None or sn < smin or sn > smax:
                continue
            yield f


def _load_json(path: Path):
    with path.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def _box_from_points(pts: List[List[float]]) -> Tuple[float, float]:
    # returns (w, h) in pixels from arbitrary list of [x,y]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = float(max(xs) - min(xs))
    h = float(max(ys) - min(ys))
    return abs(w), abs(h)


def main():
    args = parse_args()

    thresholds = [int(t.strip()) for t in args.thresholds.split(',') if t.strip().isdigit()]
    thresholds = sorted(list(set(thresholds)))

    # collect paths
    paths: List[Path] = []
    if args.list_file:
        lf = Path(args.list_file)
        if not lf.exists():
            print(f"[ERR] list_file not found: {lf}")
            return
        for line in lf.read_text(encoding='utf-8').splitlines():
            s = line.strip().strip('"')
            if not s:
                continue
            p = Path(s)
            if p.suffix.lower() == '.json':
                paths.append(p)
    else:
        if not args.label_root:
            print("[ERR] provide --label_root or --list_file")
            return
        root = Path(args.label_root)
        if not root.exists():
            print(f"[ERR] label_root not found: {root}")
            return
        inc = list(range(args.class_min, args.class_max + 1))
        if args.include_classes.strip():
            inc = [c for c in _parse_id_list(args.include_classes) if args.class_min <= c <= args.class_max]
        exc = _parse_id_list(args.exclude_classes)
        inc = [c for c in inc if c not in exc]
        paths = list(_iter_json_files(root, inc, args.stem_min, args.stem_max))

    if not paths:
        print("[ERR] no JSONs to process")
        return

    overall = {
        'files': 0,
        'boxes': 0,
        'empty_files': 0,
        'w_stats': {'min': 1e9, 'max': -1e9, 'mean': 0.0, 'p50': 0.0, 'p90': 0.0},
        'h_stats': {'min': 1e9, 'max': -1e9, 'mean': 0.0, 'p50': 0.0, 'p90': 0.0},
        'ratios': {str(t): {'w<t': 0, 'h<t': 0, 'either': 0} for t in thresholds},
    }
    per_class: Dict[str, Dict] = {}

    w_all: List[float] = []
    h_all: List[float] = []

    for jp in paths:
        try:
            data = _load_json(jp)
        except Exception as e:
            # skip unreadable
            continue
        cls_id = jp.parent.name if jp.parent.name.isdigit() else 'NA'
        pc = per_class.setdefault(cls_id, {
            'files': 0, 'boxes': 0, 'empty_files': 0,
            'ratio': {str(t): {'w<t': 0, 'h<t': 0, 'either': 0} for t in thresholds}
        })

        shapes = data.get('shapes', [])
        overall['files'] += 1
        pc['files'] += 1
        if not shapes:
            overall['empty_files'] += 1
            pc['empty_files'] += 1
            continue
        for sh in shapes:
            pts = sh.get('points') or []
            if not pts:
                continue
            w, h = _box_from_points(pts)
            w_all.append(w)
            h_all.append(h)
            overall['boxes'] += 1
            pc['boxes'] += 1
            for t in thresholds:
                if w < t:
                    overall['ratios'][str(t)]['w<t'] += 1
                    pc['ratio'][str(t)]['w<t'] += 1
                if h < t:
                    overall['ratios'][str(t)]['h<t'] += 1
                    pc['ratio'][str(t)]['h<t'] += 1
                if (w < t) or (h < t):
                    overall['ratios'][str(t)]['either'] += 1
                    pc['ratio'][str(t)]['either'] += 1

    def _agg_stats(arr: List[float]):
        if not arr:
            return {'min': None, 'max': None, 'mean': None, 'p50': None, 'p90': None}
        a = np.array(arr, dtype=np.float64)
        return {
            'min': float(np.min(a)),
            'max': float(np.max(a)),
            'mean': float(np.mean(a)),
            'p50': float(np.percentile(a, 50)),
            'p90': float(np.percentile(a, 90)),
        }

    overall['w_stats'] = _agg_stats(w_all)
    overall['h_stats'] = _agg_stats(h_all)

    report = {
        'total_json_files': len(paths),
        'overall': overall,
        'per_class': per_class,
        'thresholds': thresholds,
    }

    out_path = Path(args.out) if args.out else (Path(args.label_root) / 'label_box_stats.json' if args.label_root else Path('label_box_stats.json'))
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    # console summary
    print(f"[LABEL] files={overall['files']} boxes={overall['boxes']} empty_files={overall['empty_files']}")
    print(f"[W] min={overall['w_stats']['min']} mean={overall['w_stats']['mean']:.2f} p50={overall['w_stats']['p50']:.1f} p90={overall['w_stats']['p90']:.1f} max={overall['w_stats']['max']}")
    print(f"[H] min={overall['h_stats']['min']} mean={overall['h_stats']['mean']:.2f} p50={overall['h_stats']['p50']:.1f} p90={overall['h_stats']['p90']:.1f} max={overall['h_stats']['max']}")
    for t in thresholds:
        et = overall['ratios'][str(t)]['either']
        denom = overall['boxes'] if overall['boxes'] else 1
        print(f"[RATIO <{t}px] either={et}/{overall['boxes']} = {et/denom:.3f}")
    print(f"Report saved to {out_path}")


if __name__ == '__main__':
    main()
