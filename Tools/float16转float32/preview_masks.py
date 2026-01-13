import argparse
from pathlib import Path
import json
import re

import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser('Preview all: overlay spec (from .npy) with FCS/VTS LabelMe masks and save images')
    p.add_argument('--data_root', type=str, default='C:/Users/HP/Desktop/Exp/Data')
    p.add_argument('--fcs_root', type=str, default='C:/Users/HP/Desktop/Exp/FCSLabel')
    p.add_argument('--vts_root', type=str, default='C:/Users/HP/Desktop/Exp/VTSLabel')
    p.add_argument('--classes', type=str, default='0-23', help='e.g. 0-23 or 0,1,2')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--out_dir', type=str, default='C:/Users/HP/Desktop/Exp/LabelPreview')
    p.add_argument('--natural_sort', action='store_true')
    return p.parse_args()


def _natural_key(path: Path):
    s = path.stem
    m = re.match(r'^(\d+)', s)
    if m:
        return (0, int(m.group(1)), s[m.end():])
    return (1, s)


def load_spec_resized(npy_path: Path, size: int) -> np.ndarray:
    arr = np.load(str(npy_path), allow_pickle=False, mmap_mode='r').astype(np.float32)
    h, w = arr.shape
    if (h, w) != (size, size):
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return np.clip(arr, 0.0, 1.0)


def rasterize_labelme(json_path: Path, size: int) -> np.ndarray:
    if not json_path.exists():
        return np.zeros((size, size), dtype=np.uint8)
    data = json.loads(Path(json_path).read_text(encoding='utf-8'))
    iw = int(data.get('imageWidth', size) or size)
    ih = int(data.get('imageHeight', size) or size)
    mask = np.zeros((ih, iw), dtype=np.uint8)
    for sh in data.get('shapes', []) or []:
        pts = sh.get('points', [])
        if not pts:
            continue
        poly = np.array(pts, dtype=np.float32)
        poly[:, 0] = np.clip(poly[:, 0], 0, iw - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, ih - 1)
        poly_i = np.round(poly).astype(np.int32)[None, ...]
        cv2.fillPoly(mask, poly_i, 1)
    if (ih, iw) != (size, size):
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return mask


def overlay_and_save(spec: np.ndarray, mask: np.ndarray, out_path: Path, colormap: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = (spec * 255).astype(np.uint8)
    base3 = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap(np.where(mask > 0, 255, 0).astype(np.uint8), colormap)
    blend = cv2.addWeighted(base3, 0.8, heat, 0.5, 0)
    cv2.imwrite(str(out_path), blend)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    fcs_root = Path(args.fcs_root)
    vts_root = Path(args.vts_root)
    out_root = Path(args.out_dir)

    # build class list
    if '-' in args.classes:
        a, b = args.classes.split('-', 1)
        class_ids = list(range(int(a), int(b) + 1))
    else:
        class_ids = [int(x) for x in args.classes.split(',') if x.strip()]

    total = 0
    miss_fcs = 0
    miss_vts = 0

    for cid in class_ids:
        cdir = data_root / str(cid)
        if not cdir.is_dir():
            continue
        files = list(cdir.glob('*.npy'))
        files.sort(key=_natural_key if args.natural_sort else None)
        print(f"[Class {cid}] {len(files)} files")
        for npy_path in files:
            total += 1
            try:
                spec = load_spec_resized(npy_path, args.size)
                stem = npy_path.stem
                fcs_json = fcs_root / str(cid) / f"{stem}.json"
                vts_json = vts_root / str(cid) / f"{stem}.json"
                fcs = rasterize_labelme(fcs_json, args.size)
                vts = rasterize_labelme(vts_json, args.size)
                if not fcs.any():
                    miss_fcs += 1
                if not vts.any():
                    miss_vts += 1
                # save overlays
                overlay_and_save(spec, fcs, out_root / str(cid) / f"{stem}_FCS.png", cv2.COLORMAP_JET)
                overlay_and_save(spec, vts, out_root / str(cid) / f"{stem}_VTS.png", cv2.COLORMAP_TURBO)
            except Exception as e:
                print(f"  [ERR] {npy_path}: {e}")
    print(f"Done. total={total} missing_fcs={miss_fcs} missing_vts={miss_vts} out={out_root}")


if __name__ == '__main__':
    main()
