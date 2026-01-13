import argparse
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np

# add src to path for shared utils if needed
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import re

_UNIT_RE = re.compile(r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([gmkh]?hz)?$', re.IGNORECASE)

def normalize_freq_label(val) -> str:
    try:
        if val is None:
            return 'None'
        if isinstance(val, (int, float, np.number)):
            v = float(val)
            if not np.isfinite(v) or v <= 0:
                return 'Invalid'
            # Heuristic: >1e4 as Hz
            mhz = v / 1e6 if v > 1e4 else v
            return f"{mhz:.0f}MHz"
        s = str(val).strip()
        if not s:
            return 'None'
        sl = s.lower().replace(' ', '')
        m = _UNIT_RE.match(sl)
        if m:
            num = float(m.group(1))
            unit = (m.group(2) or 'mhz').lower()
            if unit == 'hz':
                mhz = num / 1e6
            elif unit == 'khz':
                mhz = num / 1e3
            elif unit == 'mhz':
                mhz = num
            elif unit == 'ghz':
                mhz = num * 1e3
            else:
                mhz = num
            return f"{mhz:.0f}MHz"
        # keyword fallback
        if '915' in sl:
            return '915MHz'
        if '2440' in sl or '2.44' in sl:
            return '2440MHz'
        if '5800' in sl or '5.8' in sl:
            return '5800MHz'
    except Exception:
        return 'Error'
    return 'Unknown'


def iter_npz_paths(root: Path, list_file: str | None):
    if list_file:
        for line in Path(list_file).read_text(encoding='utf-8').splitlines():
            p = line.strip()
            if p:
                yield Path(p)
    else:
        for p in root.rglob('*.npz'):
            yield p


def scan(list_file: str | None, root: str | None):
    cnt = Counter()
    examples = {}
    total = 0
    for p in iter_npz_paths(Path(root) if root else None, list_file):
        try:
            with np.load(p, allow_pickle=True) as d:
                cf = None
                # meta_json only
                if 'meta_json' in d:
                    mj = d['meta_json']
                    mj = mj.item() if hasattr(mj, 'item') else mj
                    if isinstance(mj, bytes):
                        mj = mj.decode('utf-8', errors='ignore')
                    try:
                        meta = json.loads(mj)
                    except Exception:
                        meta = {}
                    cf = meta.get('center_freq', None)
                label = normalize_freq_label(cf)
                cnt[label] += 1
                if label not in examples:
                    examples[label] = str(p)
                total += 1
        except Exception:
            cnt['LoadError'] += 1
    return total, cnt, examples


def main():
    ap = argparse.ArgumentParser(description='Scan NPZ center_freq distribution (from meta_json.center_freq).')
    ap.add_argument('--list_file', type=str, default=None, help='Optional: list file of NPZ paths.')
    ap.add_argument('--root', type=str, default=None, help='Optional: root dir to search *.npz recursively.')
    args = ap.parse_args()

    total, cnt, examples = scan(args.list_file, args.root)
    print(json.dumps({
        'total': total,
        'counts': cnt,
        'examples': examples,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
