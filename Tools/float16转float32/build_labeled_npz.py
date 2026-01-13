import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import re
import sys
import hashlib

import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser(
        "Build 512x512 NPZ samples by fusing .npy spectrograms with LabelMe masks (FCS/VTS)."
    )
    p.add_argument('--data_root', type=str, default='C:/Users/HP/Desktop/Exp/Data',
                   help='Root of original spectrogram npy folders (0..23).')
    p.add_argument('--fcs_root', type=str, default='C:/Users/HP/Desktop/Exp/FCSLabel',
                   help='Root of LabelMe JSON for FCS (mirrors class/filename).')
    p.add_argument('--vts_root', type=str, default='C:/Users/HP/Desktop/Exp/VTSLabel',
                   help='Root of LabelMe JSON for VTS (mirrors class/filename).')
    p.add_argument('--out_root', type=str, default='C:/Users/HP/Desktop/Exp/Data-512NPZ',
                   help='Where to save fused .npz (mirrors class structure).')
    p.add_argument('--size', type=int, default=512, help='Target size (size x size).')
    p.add_argument('--spec_mode', type=str, choices=['resize', 'center-crop'], default='resize',
                   help='How to go from (T,512) to (512,512): resize (INTER_AREA) or center-crop time axis.')
    p.add_argument('--spec_dtype', type=str, choices=['float16', 'float32'], default='float16',
                   help='Saved dtype for spectrogram inside NPZ.')
    p.add_argument('--max_per_class', type=int, default=-1,
                   help='Only process first N files per class after sorting (default: -1, no limit).')
    p.add_argument('--natural_sort', action='store_true',
                   help='Use natural sort like 2 < 10 based on numeric stem prefix (e.g., 0-200).')
    p.add_argument('--dry_run', action='store_true', help='Scan and print without writing files.')
    # 新增：中心频率写入策略
    p.add_argument('--center_freq_by_class', type=str, default='',
                   help="Class->center freq mapping. Path to JSON or inline '0=915MHz,1=915MHz,4=2440MHz' (MHz/Hz/GHz).")
    p.add_argument('--center_freq_map', type=str, default='',
                   help="Filename regex->center freq mapping. Path to JSON or inline '.*915.*=915MHz;.*2440.*=2440MHz';.*58.*=5800MHz'. First match wins.")
    p.add_argument('--center_freq_auto', action='store_true',
                   help='Try to auto-detect center frequency from npy path tokens (e.g., 915/2440/5800 or MHz/GHz patterns).')
    return p.parse_args()


def _natural_key(path: Path):
    # Expect stems like '123-a' -> (123, '-a')
    s = path.stem
    m = re.match(r'^(\d+)', s)
    if m:
        num = int(m.group(1))
        return (0, num, s[m.end():])
    # fallback keeps files after numbered ones
    return (1, s)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _resize_to_512(arr32: np.ndarray, size: int, mode: str) -> np.ndarray:
    H, W = arr32.shape
    if W != size or H != size:
        # 统一使用 INTER_AREA 到正方形（与既有流程一致，不裁剪）
        return cv2.resize(arr32, (size, size), interpolation=cv2.INTER_AREA)
    return arr32


def load_spec_raw_and_512(npy_path: Path, size: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(str(npy_path), allow_pickle=False, mmap_mode='r')
    if arr.ndim != 2:
        raise ValueError(f"Expect 2D spectrogram, got shape={arr.shape} at {npy_path}")
    # 原始按 float16 保留
    spec_raw = np.asarray(arr, dtype=np.float16)
    # 转 float32 做插值
    arr32 = spec_raw.astype(np.float32, copy=False)
    spec_512 = _resize_to_512(arr32, size, mode)
    # 明确为 float32
    spec_512 = np.asarray(spec_512, dtype=np.float32)
    return spec_raw, spec_512


def rasterize_labelme(json_path: Path, size: int) -> np.ndarray:
    if not json_path.exists():
        return np.zeros((size, size), dtype=np.uint8)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # prefer provided image size; fallback to target size
    iw = int(data.get('imageWidth', size) or size)
    ih = int(data.get('imageHeight', size) or size)
    mask = np.zeros((ih, iw), dtype=np.uint8)
    shapes = data.get('shapes', []) or []
    for sh in shapes:
        pts = sh.get('points', [])
        if not pts:
            continue
        # labelme points are [[x,y], ...]
        poly = np.array(pts, dtype=np.float32).reshape(-1, 2)
        # clip to image bounds then convert to int32
        poly[:, 0] = np.clip(poly[:, 0], 0, iw - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, ih - 1)
        poly_i = np.round(poly).astype(np.int32)[None, ...]
        cv2.fillPoly(mask, poly_i, 1)
    if (ih, iw) != (size, size):
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.uint8)


def _script_sha1() -> str:
    try:
        p = Path(__file__)
        return hashlib.sha1(p.read_bytes()).hexdigest()[:12]
    except Exception:
        return "unknown"


# ---------- 中心频率解析工具 ----------
_UNIT_RE = re.compile(r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([gmkh]?hz)?$', re.IGNORECASE)


def _normalize_freq_to_mhz(val) -> Optional[float]:
    """接受数值或字符串（Hz/kHz/MHz/GHz），返回 MHz。"""
    try:
        if isinstance(val, (int, float, np.number)):
            v = float(val)
            if not np.isfinite(v) or v <= 0:
                return None
            # 启发式：>1e4 视为 Hz
            return v / 1e6 if v > 1e4 else v
        s = str(val).strip().lower().replace(' ', '')
        if not s:
            return None
        m = _UNIT_RE.match(s)
        if m:
            num = float(m.group(1))
            unit = (m.group(2) or 'mhz').lower()
            if unit == 'hz':
                return num / 1e6
            if unit == 'khz':
                return num / 1e3
            if unit == 'mhz':
                return num
            if unit == 'ghz':
                return num * 1e3
        # 关键字启发
        if '915' in s:
            return 915.0
        if '2440' in s or '2.44' in s:
            return 2440.0
        if '5800' in s or '5.8' in s:
            return 5800.0
    except Exception:
        return None
    return None


def _load_mapping_str_or_file(arg: str) -> Dict[str, str]:
    """从文件(JSON)或内联字符串解析映射为 dict[str->str]。"""
    if not arg:
        return {}
    p = Path(arg)
    if p.exists() and p.is_file():
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    # inline: allow comma/semicolon separated "key=value"
    out: Dict[str, str] = {}
    parts = re.split(r'[;,]', arg)
    for part in parts:
        if not part.strip():
            continue
        if '=' in part:
            k, v = part.split('=', 1)
            out[k.strip()] = v.strip()
    return out


def resolve_center_freq_mhz(class_id: int, npy_path: Path, by_class_arg: str, map_arg: str, auto: bool) -> Optional[float]:
    # 1) 文件名正则映射优先（第一命中）
    pat_map = _load_mapping_str_or_file(map_arg)
    if pat_map:
        s = str(npy_path).replace('\\', '/')
        for pat, v in pat_map.items():
            try:
                if re.search(pat, s, re.IGNORECASE):
                    mhz = _normalize_freq_to_mhz(v)
                    if mhz:
                        return mhz
            except re.error:
                continue
    # 2) 按类映射
    cls_map = _load_mapping_str_or_file(by_class_arg)
    if cls_map:
        key = str(class_id)
        if key in cls_map:
            mhz = _normalize_freq_to_mhz(cls_map[key])
            if mhz:
                return mhz
    # 3) 自动从文件名启发
    if auto:
        mhz = _normalize_freq_to_mhz(str(npy_path.name))
        if mhz:
            return mhz
        mhz = _normalize_freq_to_mhz(str(npy_path.parent))
        if mhz:
            return mhz
    return None


def _try_load_sidecar_meta(npy_path: Path) -> Tuple[Optional[float], Optional[float], Optional[float], dict]:
    """尝试读取与 npy 同名的 .json 侧车（由 MAT 转换生成）。
    返回 (center_freq_mhz, bandwidth_mhz, fs_hz, raw_meta_dict)。
    """
    sidecar = npy_path.with_suffix('.json')
    if not sidecar.exists():
        return None, None, None, {}
    try:
        with open(sidecar, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        # 优先使用显式字段，其次解析字符串字段
        cf = meta.get('center_freq_mhz')
        if cf is None:
            cf = _normalize_freq_to_mhz(meta.get('center_freq')) if meta.get('center_freq') is not None else None
        bw = meta.get('bandwidth_mhz')
        if bw is None:
            fs_hz = meta.get('fs') or meta.get('fs_hz')
            bw = (float(fs_hz) / 1e6) if fs_hz else None
        fs_hz = meta.get('fs') or meta.get('fs_hz')
        fs_hz = float(fs_hz) if fs_hz is not None else None
        return (float(cf) if cf is not None else None,
                float(bw) if bw is not None else None,
                fs_hz,
                meta)
    except Exception:
        return None, None, None, {}


def build_one(npy_path: Path, cls_dir: Path, out_cls_dir: Path, fcs_root: Path, vts_root: Path,
              size: int, mode: str, spec_dtype: str, dry: bool,
              cf_by_class: str = '', cf_map: str = '', cf_auto: bool = False) -> Tuple[Optional[Path], Dict[str, int]]:
    stem = npy_path.stem  # e.g., '1-a'
    fcs_json = fcs_root / cls_dir.name / f"{stem}.json"
    vts_json = vts_root / cls_dir.name / f"{stem}.json"

    # 生成原始与512规格
    spec_raw, spec_512 = load_spec_raw_and_512(npy_path, size, mode)
    # 覆盖 spec_dtype 仅作用于 spec_raw（spec_512 固定 float32 以提升训练稳定性）
    spec_raw = spec_raw.astype(np.float16) if spec_dtype == 'float16' else spec_raw.astype(np.float32)

    fcs_mask_512 = rasterize_labelme(fcs_json, size)
    vts_mask_512 = rasterize_labelme(vts_json, size)

    # label 扩展：分类ID + 兼容开放集
    class_id = int(cls_dir.name)
    label_dict = {
        'class_id': class_id,
        'is_unknown': False,
        'open_set_flag': 0.0,
    }

    # 优先：读取 MAT 转换产生的侧车元数据
    cf_sidecar_mhz, bw_sidecar_mhz, fs_sidecar_hz, sidecar_meta = _try_load_sidecar_meta(npy_path)

    # 仅当侧车缺失时，才考虑原有的映射/启发式（可选）
    cf_mhz = cf_sidecar_mhz
    if cf_mhz is None:
        cf_mhz = resolve_center_freq_mhz(class_id, npy_path, cf_by_class, cf_map, cf_auto)

    # 带宽优先使用侧车，其次保持为空（不再硬编码 100MHz）
    bw_mhz = bw_sidecar_mhz

    # 元信息与依赖
    meta = {
        'stem': stem,
        'class_id': class_id,
        'src_npy': str(npy_path.resolve()),
        'src_fcs': str(fcs_json.resolve()) if fcs_json.exists() else '',
        'src_vts': str(vts_json.resolve()) if vts_json.exists() else '',
        'orig_shape': tuple(int(x) for x in spec_raw.shape),
        'target_size': size,
        # 不再硬编码，保留旧字段但可为空
        'bandwidth': (f"{bw_mhz:.0f}MHz" if bw_mhz is not None else None),
        # 新增显式数值字段，便于下游解析
        'bandwidth_mhz': (float(bw_mhz) if bw_mhz is not None else None),
        'center_freq': (f"{cf_mhz:.0f}MHz" if cf_mhz is not None else None),
        'center_freq_mhz': (float(cf_mhz) if cf_mhz is not None else None),
        'fs_hz': (float(fs_sidecar_hz) if fs_sidecar_hz is not None else None),
        'resize': 'INTER_AREA' if mode == 'resize' else 'center-crop/pad',
        'source_dtype': 'float16',
        'spec512_dtype': 'float32',
        'pipeline_version': 'v1',
        'deps': {
            'numpy': np.__version__,
            'opencv': cv2.__version__,
            'python': sys.version.split()[0],
            'hash_build_labeled_npz_py': _script_sha1(),
        }
    }

    out_path = out_cls_dir / f"{stem}.npz"
    if not dry:
        ensure_dir(out_cls_dir)
        np.savez_compressed(
            str(out_path),
            # 频谱
            spec_raw=spec_raw,            # (543,512) float16/float32(可选，默认float16)
            spec_512=spec_512,            # (512,512) float32
            # 掩码
            fcs_mask_512=fcs_mask_512,    # (512,512) uint8
            vts_mask_512=vts_mask_512,    # (512,512) uint8
            # 标签（显式字段）
            label_class_id=np.int32(class_id),
            label_is_unknown=np.uint8(0),
            label_open_set=np.float32(0.0),
            # 标签与元信息（JSON，便于扩展/审计）
            label_json=json.dumps(label_dict, ensure_ascii=False),
            meta_json=json.dumps(meta, ensure_ascii=False),
        )
    return (out_path if not dry else None,
            {
                'has_fcs': int(fcs_json.exists() and fcs_mask_512.any()),
                'has_vts': int(vts_json.exists() and vts_mask_512.any()),
            })


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    fcs_root = Path(args.fcs_root)
    vts_root = Path(args.vts_root)
    out_root = Path(args.out_root)
    size = int(args.size)

    classes = [p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit()]
    classes.sort(key=lambda p: int(p.name))
    print(f"[Scan] classes={len(classes)} root={data_root}")

    total, written = 0, 0
    has_fcs, has_vts = 0, 0
    for cls_dir in classes:
        files = list(cls_dir.glob('*.npy'))
        if args.natural_sort:
            files.sort(key=_natural_key)
        else:
            files.sort()
        # 仅分析每类前N个
        if args.max_per_class and args.max_per_class > 0:
            files = files[:args.max_per_class]
        out_cls_dir = out_root / cls_dir.name
        print(f"[Class {cls_dir.name}] files={len(files)}")
        for npy_path in files:
            total += 1
            try:
                out_path, flags = build_one(
                    npy_path, cls_dir, out_cls_dir,
                    fcs_root, vts_root,
                    size, args.spec_mode, args.spec_dtype, args.dry_run,
                    cf_by_class=args.center_freq_by_class,
                    cf_map=args.center_freq_map,
                    cf_auto=bool(args.center_freq_auto)
                )
                if out_path is not None:
                    written += 1
                has_fcs += flags['has_fcs']
                has_vts += flags['has_vts']
            except Exception as e:
                print(f"  [ERR] {npy_path}: {e}")
    print(f"Done. total={total} written={written} with_fcs={has_fcs} with_vts={has_vts} out={out_root}")


if __name__ == '__main__':
    main()
