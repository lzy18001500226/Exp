import argparse
from pathlib import Path
import numpy as np
import json


def parse_args():
    p = argparse.ArgumentParser('Compute global mean/std and percentiles for spec_512 over NPZ dataset or a file list')
    p.add_argument('--data_root', type=str, default='C:/Users/HP/Desktop/Exp/Data-512NPZ', help='Root of fused NPZ (class subfolders). Ignored if --list_file is set')
    p.add_argument('--list_file', type=str, default='', help='Optional: a txt file containing absolute paths to .npz (e.g., train_list_*.txt)')
    p.add_argument('--max_per_class', type=int, default=-1, help='Only when scanning data_root by class; -1 for all')
    p.add_argument('--bins', type=int, default=2048, help='Histogram bins for percentile estimation (values assumed in [0,1])')
    p.add_argument('--save', type=str, default='C:/Users/HP/Desktop/Exp/Model/DSFNet-Output/stats.json')
    return p.parse_args()


def _iter_files_from_root(root: Path, max_per_class: int):
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name)):
        files = sorted(cls_dir.glob('*.npz'))
        if max_per_class and max_per_class > 0:
            files = files[:max_per_class]
        for f in files:
            yield f


def _iter_files_from_list(list_file: Path):
    for line in list_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        if p.suffix.lower() == '.npz' and p.exists():
            yield p


def _percentile_from_hist(hist: np.ndarray, p: float) -> float:
    # hist over [0,1] with equal-width bins
    cdf = np.cumsum(hist)
    if cdf[-1] == 0:
        return 0.0
    target = p * cdf[-1]
    idx = int(np.searchsorted(cdf, target, side='left'))
    idx = np.clip(idx, 0, len(hist)-1)
    # convert bin idx to value in [0,1]
    return (idx + 0.5) / len(hist)


def main():
    args = parse_args()

    files = []
    if args.list_file:
        files = list(_iter_files_from_list(Path(args.list_file)))
    else:
        files = list(_iter_files_from_root(Path(args.data_root), args.max_per_class))

    if not files:
        print('No data found')
        return

    # Online mean/std (Welford) and histogram for percentiles
    count = 0
    mean = 0.0
    M2 = 0.0
    hist = np.zeros(args.bins, dtype=np.int64)

    for f in files:
        try:
            with np.load(str(f), allow_pickle=False) as d:
                if 'spec_512' not in d:
                    continue
                x = d['spec_512'].astype(np.float32)
        except Exception:
            continue
        # flatten
        arr = x.ravel()
        # assume in [0,1]; clip to be safe
        arr = np.clip(arr, 0.0, 1.0)
        # update histogram
        idx = np.minimum((arr * args.bins).astype(np.int64), args.bins - 1)
        # np.add.at ensures correct accumulation even with repeated indices
        np.add.at(hist, idx, 1)
        # update Welford stats in chunks to avoid precision issues
        # process as one chunk
        n = arr.size
        if n == 0:
            continue
        count_new = count + n
        delta = float(arr.mean()) - mean
        mean += delta * (n / max(count_new, 1))
        # For variance, combine variances: use per-array mean/var
        var_arr = float(arr.var())
        M2 += var_arr * n + (delta ** 2) * (count * n / max(count_new, 1))
        count = count_new

    if count == 0:
        print('No valid pixels found')
        return

    std = (M2 / max(count - 1, 1)) ** 0.5 + 1e-6
    p1 = _percentile_from_hist(hist, 0.01)
    p99 = _percentile_from_hist(hist, 0.99)

    out = {'mean': float(mean), 'std': float(std), 'p1': float(p1), 'p99': float(p99),
           'files': len(files), 'pixels': int(count), 'bins': int(args.bins), 'from_list': bool(args.list_file)}
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved stats to', args.save, out)


if __name__ == '__main__':
    main()
