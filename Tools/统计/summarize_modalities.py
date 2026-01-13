import argparse
import json
import re
from pathlib import Path
import numpy as np


def stem_num(stem: str):
    m = re.match(r"^(\d+)", stem)
    return int(m.group(1)) if m else None


def load_flags(npz_path: Path):
    try:
        with np.load(str(npz_path), allow_pickle=True) as z:
            has_fcs = bool(z.get('has_fcs', False))
            has_vts = bool(z.get('has_vts', False))
            # 退路：若显式标志缺失，尝试通过掩码存在与否粗略判断（尽量避免）
            if not ('has_fcs' in z.files or 'has_vts' in z.files):
                if 'fcs_mask_512' in z.files:
                    try:
                        has_fcs = bool(np.any(z['fcs_mask_512']))
                    except Exception:
                        has_fcs = True
                if 'vts_mask_512' in z.files:
                    try:
                        has_vts = bool(np.any(z['vts_mask_512']))
                    except Exception:
                        has_vts = True
            return has_fcs, has_vts
    except Exception:
        return None, None


def summarize(root: Path, smin: int = None, smax: int = None):
    total = 0
    both = 0
    fcs_only = 0
    vts_only = 0
    classes_present = set()
    classes_with_vts = set()
    classes_without_vts_candidates = set()

    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name)):
        cls_id = int(cls_dir.name)
        any_vts_in_class = False
        any_sample_in_class = False
        for f in cls_dir.glob('*.npz'):
            sn = stem_num(f.stem)
            if smin is not None and smax is not None:
                if sn is None or sn < smin or sn > smax:
                    continue
            has_fcs, has_vts = load_flags(f)
            if has_fcs is None and has_vts is None:
                continue  # 读取失败，跳过
            total += 1
            any_sample_in_class = True
            if has_fcs and has_vts:
                both += 1
                any_vts_in_class = True
            elif has_fcs and not has_vts:
                fcs_only += 1
            elif has_vts and not has_fcs:
                vts_only += 1
            else:
                # 两者都 False 的样本，忽略统计（极少见）
                pass
        if any_sample_in_class:
            classes_present.add(cls_id)
            if any_vts_in_class:
                classes_with_vts.add(cls_id)
            else:
                classes_without_vts_candidates.add(cls_id)

    classes_without_vts = sorted(list(classes_present - classes_with_vts))

    return {
        'root': str(root),
        'stem_range': [smin, smax] if (smin is not None and smax is not None) else None,
        'total_samples': total,
        'samples_with_both_modalities': both,
        'samples_with_fcs_only': fcs_only,
        'samples_with_vts_only': vts_only,
        'classes_total_present': len(classes_present),
        'classes_with_vts_count': len(classes_with_vts),
        'classes_without_vts_count': len(classes_without_vts),
        'classes_with_vts': sorted(list(classes_with_vts)),
        'classes_without_vts': classes_without_vts,
    }


def main():
    ap = argparse.ArgumentParser('Summarize modality distribution from NPZ dataset')
    ap.add_argument('--root', type=str, required=True, help='Path to Data-512NPZ root')
    ap.add_argument('--stem_min', type=int, default=None)
    ap.add_argument('--stem_max', type=int, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    smin = args.stem_min
    smax = args.stem_max

    stats = summarize(root, smin, smax)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
