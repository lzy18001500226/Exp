import argparse
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Set
import numpy as np


def parse_args():
    p = argparse.ArgumentParser("NPZ dataset health check: keys/shapes/dtypes/ranges, optional JSON alignment")
    p.add_argument('--data_root', type=str, required=True, help='Root of Data-512NPZ (class subfolders 0..23)')
    p.add_argument('--out', type=str, default='', help='Where to save health_report.json (default alongside data_root)')
    p.add_argument('--class_min', type=int, default=0)
    p.add_argument('--class_max', type=int, default=23)
    p.add_argument('--include_classes', type=str, default='', help='Comma-separated class ids (empty = all in range)')
    p.add_argument('--exclude_classes', type=str, default='', help='Comma-separated class ids to exclude')
    p.add_argument('--stem_min', type=int, default=0)
    p.add_argument('--stem_max', type=int, default=10**9)
    p.add_argument('--max_files', type=int, default=0, help='Max files to check (0 = all)')
    p.add_argument('--check_fcs', action='store_true', help='Also check FCSLabel JSON existence and shapes>0 for mapped items')
    p.add_argument('--check_vts', action='store_true', help='Also check VTSLabel JSON existence and shapes>0 for mapped items')
    return p.parse_args()


def _parse_id_list(s: str) -> Set[int]:
    out: Set[int] = set()
    if s.strip():
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


def _iter_npz_files(root: Path, include_classes: List[int], smin: int, smax: int):
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name)):
        cid = int(cls_dir.name)
        if cid not in include_classes:
            continue
        for f in sorted(cls_dir.glob('*.npz')):
            sn = _stem_num(f.stem)
            if sn is None or sn < smin or sn > smax:
                continue
            yield f


def _json_shapes_count(path: Path) -> int:
    try:
        with path.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
        shapes = data.get('shapes', [])
        return len(shapes) if isinstance(shapes, list) else 0
    except Exception:
        return 0


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[ERR] data_root not found: {data_root}")
        return

    inc = list(range(args.class_min, args.class_max + 1))
    if args.include_classes.strip():
        inc = [c for c in _parse_id_list(args.include_classes) if args.class_min <= c <= args.class_max]
    exc = _parse_id_list(args.exclude_classes)
    inc = [c for c in inc if c not in exc]
    if not inc:
        print("[ERR] No classes selected")
        return

    files = list(_iter_npz_files(data_root, inc, args.stem_min, args.stem_max))
    if args.max_files and len(files) > args.max_files:
        files = files[:args.max_files]

    total = len(files)
    report: Dict[str, any] = {
        'data_root': str(data_root.resolve()),
        'total_files': total,
        'class_range': [args.class_min, args.class_max],
        'include_classes': inc,
        'exclude_classes': sorted(list(exc)),
        'stem_range': [args.stem_min, args.stem_max],
        'max_files': args.max_files,
        'spec_512': {'missing_key': 0, 'bad_dtype': 0, 'bad_shape': 0, 'nan_inf': 0, 'out_of_range': 0,
                     'min': +1e9, 'max': -1e9, 'mean': 0.0, 'std': 0.0},
        'other_keys': {},
        'examples': {'missing_spec_512': [], 'bad_dtype': [], 'bad_shape': [], 'nan_inf': [], 'out_of_range': []},
        'json_check': {}
    }

    mins: List[float] = []
    maxs: List[float] = []
    means: List[float] = []
    stds: List[float] = []

    for f in files:
        try:
            with np.load(f, allow_pickle=False) as npz:
                keys = set(npz.files)
                # record other keys once
                for k in keys:
                    report['other_keys'][k] = report['other_keys'].get(k, 0) + 1
                if 'spec_512' not in keys:
                    report['spec_512']['missing_key'] += 1
                    if len(report['examples']['missing_spec_512']) < 20:
                        report['examples']['missing_spec_512'].append(str(f))
                    continue
                x = npz['spec_512']
                # dtype
                if x.dtype != np.float32:
                    report['spec_512']['bad_dtype'] += 1
                    if len(report['examples']['bad_dtype']) < 20:
                        report['examples']['bad_dtype'].append(f"{f}::{x.dtype}")
                # shape
                if x.shape != (512, 512):
                    report['spec_512']['bad_shape'] += 1
                    if len(report['examples']['bad_shape']) < 20:
                        report['examples']['bad_shape'].append(f"{f}::{x.shape}")
                # stats
                m = float(np.nanmin(x))
                M = float(np.nanmax(x))
                mu = float(np.nanmean(x))
                sd = float(np.nanstd(x))
                mins.append(m); maxs.append(M); means.append(mu); stds.append(sd)
                report['spec_512']['min'] = min(report['spec_512']['min'], m)
                report['spec_512']['max'] = max(report['spec_512']['max'], M)
                # nan/inf
                if not np.isfinite(x).all():
                    report['spec_512']['nan_inf'] += 1
                    if len(report['examples']['nan_inf']) < 20:
                        report['examples']['nan_inf'].append(str(f))
                # range (tolerate tiny eps)
                if (m < -1e-6) or (M > 1.0 + 1e-6):
                    report['spec_512']['out_of_range'] += 1
                    if len(report['examples']['out_of_range']) < 20:
                        report['examples']['out_of_range'].append(f"{f}::min={m:.4f},max={M:.4f}")
        except Exception as e:
            report['examples'].setdefault('load_error', [])
            if len(report['examples']['load_error']) < 20:
                report['examples']['load_error'].append(f"{f}::{e}")

    # aggregate moments
    if mins:
        report['spec_512']['mean'] = float(np.mean(means))
        report['spec_512']['std'] = float(np.mean(stds))

    # optional JSON checks
    json_report = {}
    label_root_parent = data_root.parent
    for tag, enabled in [('FCSLabel', args.check_fcs), ('VTSLabel', args.check_vts)]:
        if not enabled:
            continue
        root = label_root_parent / tag
        c = {'root': str(root.resolve()), 'missing_json': 0, 'empty_json': 0, 'checked': 0, 'examples': {'missing': [], 'empty': []}}
        for f in files:
            cls_id = Path(f).parent.name
            jp = root / cls_id / f"{Path(f).stem}.json"
            if not jp.exists():
                c['missing_json'] += 1
                if len(c['examples']['missing']) < 20:
                    c['examples']['missing'].append(str(jp))
                continue
            n_shapes = _json_shapes_count(jp)
            if n_shapes <= 0:
                c['empty_json'] += 1
                if len(c['examples']['empty']) < 20:
                    c['examples']['empty'].append(str(jp))
            c['checked'] += 1
        json_report[tag] = c
    if json_report:
        report['json_check'] = json_report

    # save
    out_path = Path(args.out) if args.out else (data_root / 'npz_health_report.json')
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    # console summary
    print(f"[NPZ] root={data_root} files={total}")
    s = report['spec_512']
    print(f"[spec_512] missing_key={s['missing_key']} bad_dtype={s['bad_dtype']} bad_shape={s['bad_shape']} nan_inf={s['nan_inf']} out_of_range={s['out_of_range']} min={s['min']:.4f} max={s['max']:.4f} mean~{s['mean']:.4f} std~{s['std']:.4f}")
    if report.get('json_check'):
        for tag, c in report['json_check'].items():
            print(f"[{tag}] checked={c['checked']} missing_json={c['missing_json']} empty_json={c['empty_json']}")
    print(f"Report saved to {out_path}")


if __name__ == '__main__':
    main()
