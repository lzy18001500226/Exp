import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
import cv2

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


def parse_args():
    p = argparse.ArgumentParser("Export .npy under Data/<class> to PNG with identical folder structure")
    p.add_argument('--data_root', type=str, default='C:/Users/HP/Desktop/Exp/Data', help='Root folder containing class subfolders (e.g., 0..23)')
    p.add_argument('--out_dir', type=str, default='C:/Users/HP/Desktop/Exp/LabelMePNG', help='Output root for PNG files (mirrors data_root structure)')
    p.add_argument('--colormap', type=str, default='jet', choices=['jet', 'gray'], help='Colormap for PNG')
    p.add_argument('--size', type=int, default=512, help='Resize to size x size')
    p.add_argument('--ascii_bar', action='store_true', help='Force ASCII-only progress bar (avoid garbled chars)')
    p.add_argument('--bar_ncols', type=int, default=80, help='Progress bar width')
    p.add_argument('--write_csv', type=str, default='', help='Index CSV path (default: <out_dir>/index_all.csv)')
    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_spec(npy_path: Path, size: int) -> np.ndarray:
    arr = np.load(str(npy_path))
    arr = np.asarray(arr, dtype=np.float32)
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    norm = (arr - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    if gray.ndim == 3:
        gray = gray.squeeze()
    if gray.shape[0] != size or gray.shape[1] != size:
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return gray


def spec_to_png(gray: np.ndarray, colormap: str) -> np.ndarray:
    if colormap == 'gray':
        return gray
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def collect_numeric_class_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for d in root.iterdir():
        if d.is_dir() and d.name.isdigit():
            out.append(d)
    out.sort(key=lambda p: int(p.name))
    return out


def export_tree(data_root: Path, out_root: Path, *, colormap: str, size: int, ascii_bar: bool, bar_ncols: int, csv_path: Path):
    ensure_dir(out_root)
    ensure_dir(csv_path.parent)

    class_dirs = collect_numeric_class_dirs(data_root)
    print(f"[Scan] classes={len(class_dirs)} root={data_root}", flush=True)

    ok, miss = 0, 0
    with open(csv_path, 'w', newline='', encoding='utf-8') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(['png_path', 'npy_path', 'label'])
        for cdir in class_dirs:
            label = cdir.name
            files = sorted(cdir.rglob('*.npy'))
            print(f"[Class] {label} files={len(files)}", flush=True)
            pbar = None
            processed = 0
            if tqdm is not None:
                pbar = tqdm(total=len(files), desc=f"{label}", unit='img', ncols=bar_ncols, ascii=ascii_bar, mininterval=0.2, leave=True)
            out_dir = out_root / label
            ensure_dir(out_dir)
            for npy_path in files:
                try:
                    gray = load_spec(npy_path, size)
                    png = spec_to_png(gray, colormap)
                    out_png = out_dir / (npy_path.stem + '.png')
                    cv2.imwrite(str(out_png), png)
                    w.writerow([str(out_png.resolve()), str(npy_path.resolve()), label])
                    ok += 1
                except Exception as e:
                    print(f"[ERR] {npy_path}: {e}", flush=True)
                    miss += 1
                finally:
                    if pbar:
                        pbar.update(1)
                    else:
                        processed += 1
                        if processed % 200 == 0 or processed == len(files):
                            print(f"  {label}: {processed}/{len(files)}", flush=True)
            if pbar and hasattr(pbar, 'close'):
                pbar.close()
    print(f"Done. exported={ok} missing={miss} index={csv_path}", flush=True)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)
    csv_path = Path(args.write_csv) if args.write_csv else (out_root / 'index_all.csv')
    export_tree(
        data_root,
        out_root,
        colormap=args.colormap,
        size=args.size,
        ascii_bar=args.ascii_bar,
        bar_ncols=args.bar_ncols,
        csv_path=csv_path,
    )


if __name__ == '__main__':
    main()
