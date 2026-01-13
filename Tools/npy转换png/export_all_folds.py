import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
# 进度条（可选）
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


def parse_args():
    p = argparse.ArgumentParser("Export .npy spectrograms to PNG for LabelMe and build index")
    p.add_argument('--list', dest='lists', action='append', required=True,
                   help='Path to a list file (can be repeated). Each line: "+path+ [label]")')
    p.add_argument('--out_dir', type=str, required=True, help='Output root for PNG files')
    p.add_argument('--data_root', type=str, default='C:/Users/HP/Desktop/Exp/Data', help='Root for resolving ./Data relative paths')
    p.add_argument('--write_csv', type=str, default='', help='Index CSV path (default: <out_dir>/index.csv)')
    p.add_argument('--colormap', type=str, default='jet', choices=['jet', 'gray'], help='Colormap for PNG')
    p.add_argument('--size', type=int, default=512, help='Resize to size x size')
    # 新增：进度条控制
    p.add_argument('--ascii_bar', action='store_true', help='Force ASCII-only progress bar (avoid garbled chars on Windows)')
    p.add_argument('--bar_ncols', type=int, default=100, help='Progress bar width')
    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def resolve_path(raw: str, list_file: Path, data_root: Path) -> Path:
    s = raw.strip().strip('"').strip("'")
    p = Path(s)
    if not p.is_absolute():
        if s.startswith('./Data') or s.startswith('.\\Data') or s.startswith('Data'):
            if s.startswith('./Data'):
                rest = s[len('./Data'):]
            elif s.startswith('.\\Data'):
                rest = s[len('.\\Data'):]
            elif s.startswith('Data'):
                rest = s[len('Data'):]
            else:
                rest = s
            rest = rest.lstrip('\\/').replace('/', os.sep).replace('\\', os.sep)
            p = data_root / rest
        else:
            p = (list_file.parent / p).resolve()
    if not p.exists():
        # try .npy
        if p.suffix.lower() != '.npy':
            cand = p.with_suffix('.npy')
            if cand.exists():
                return cand
        # try stem-a.npy
        cand_a = p.parent / f"{p.stem}-a.npy"
        if cand_a.exists():
            return cand_a
        # fuzzy
        if p.parent.exists():
            matches = sorted(p.parent.glob(f"{p.stem}*.npy"))
            if matches:
                # prefer contains -a
                for m in matches:
                    if (p.stem + '-a').lower() in m.stem.lower():
                        return m
                return matches[0]
    return p


def load_spec(path: Path, size: int) -> np.ndarray:
    x = np.load(str(path))
    if getattr(x, 'ndim', 2) > 2:
        x = np.squeeze(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    if x.shape != (size, size):
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_LINEAR)
    # normalize to [0,1]
    xmin, xmax = float(x.min()), float(x.max())
    if xmax - xmin > 1e-8:
        x = (x - xmin) / (xmax - xmin)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return x


def spec_to_png(x: np.ndarray, colormap: str) -> np.ndarray:
    img8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    if colormap == 'gray':
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)  # keep 3-ch for tools
    # jet
    colored = cv2.applyColorMap(img8, cv2.COLORMAP_JET)  # BGR
    return colored


def export_lists(lists: List[str], out_dir: Path, data_root: Path, csv_path: Path, size: int, colormap: str, ascii_bar: bool, bar_ncols: int):
    ensure_dir(out_dir)
    ensure_dir(csv_path.parent)
    # 预加载每个清单内容
    scanned: List[Tuple[Path, str, List[str]]] = []  # (list_path, fold, lines)
    for list_file_str in lists:
        list_path = Path(list_file_str)
        fold = list_path.name
        try:
            lines = list_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception:
            print(f"[WARN] cannot read {list_path}")
            lines = []
        scanned.append((list_path, fold, lines))

    ok, missing = 0, 0
    with open(csv_path, 'w', newline='', encoding='utf-8') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(['png_path', 'npy_path', 'label', 'fold'])
        # 每个清单单独一个进度条（36条）
        for list_path, fold, lines in scanned:
            non_empty = [ln for ln in lines if ln.strip()]
            total = len(non_empty)
            print(f"[List] {fold} | {list_path}", flush=True)
            if tqdm is not None:
                pbar = tqdm(total=total, desc=f"{fold}", unit='img', ncols=bar_ncols, ascii=ascii_bar, mininterval=0.2, leave=True)
            else:
                pbar = None
                processed = 0
            for line in lines:
                line = line.strip()
                if not line:
                    if pbar:
                        pbar.update(1)
                    else:
                        processed += 1
                        if processed % 200 == 0 or processed == total:
                            print(f"  {fold}: {processed}/{total}", flush=True)
                    continue
                # allow quoted only path
                line = line.strip().strip('"').strip("'")
                parts = line.split()
                raw_path = parts[0]
                label = parts[1] if len(parts) >= 2 and parts[1].isdigit() else ''
                npy_path = resolve_path(raw_path, list_path, data_root)
                try:
                    if not npy_path.exists():
                        print(f"[MISS] {raw_path} -> {npy_path}", flush=True)
                        missing += 1
                    else:
                        spec = load_spec(npy_path, size)
                        png = spec_to_png(spec, colormap)
                        sub = label if label != '' else 'unlabeled'
                        out_subdir = out_dir / fold / sub
                        ensure_dir(out_subdir)
                        out_png = out_subdir / (npy_path.stem + '.png')
                        cv2.imwrite(str(out_png), png)
                        w.writerow([str(out_png.resolve()), str(npy_path.resolve()), label, fold])
                        ok += 1
                except Exception as e:
                    print(f"[ERR] {npy_path}: {e}", flush=True)
                finally:
                    if pbar:
                        pbar.update(1)
                    else:
                        processed += 1
                        if processed % 200 == 0 or processed == total:
                            print(f"  {fold}: {processed}/{total}", flush=True)
            if pbar and hasattr(pbar, 'close'):
                pbar.close()
    print(f"Done. exported {ok}, missing {missing}. index: {csv_path}", flush=True)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    data_root = Path(args.data_root)
    csv_path = Path(args.write_csv) if args.write_csv else (out_dir / 'index.csv')
    export_lists(args.lists, out_dir, data_root, csv_path, args.size, args.colormap, args.ascii_bar, args.bar_ncols)


if __name__ == '__main__':
    main()
