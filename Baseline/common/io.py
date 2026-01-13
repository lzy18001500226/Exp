from __future__ import annotations
import json
import math
import warnings
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Set

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset

# ----------------------------
# 公共工具：直接读取 JSON/NPZ 的实时 YOLO Dataset（无缓存）
# 以及可选的离线转换函数（备用）
# ----------------------------


def _read_list_file(list_path: Path) -> List[str]:
    lines = []
    with open(list_path, 'r', encoding='utf-8') as f:
        for s in f:
            s = s.strip().strip('"')
            if s:
                lines.append(s)
    return lines


def _parse_line_to_paths(s: str) -> Tuple[Path | None, Path | None]:
    """
    兼容多种列表格式：
    1) ".../xxx.npz .../xxx.json"（空格/逗号分隔）
    2) 仅 npz 或 仅 json（尝试按同名推断另一侧路径）
    返回: (npz_path or None, json_path or None)
    """
    parts = [p for p in s.replace(',', ' ').split() if p]
    if len(parts) >= 2:
        a, b = Path(parts[0]), Path(parts[1])
        npz = a if a.suffix.lower() == '.npz' else (b if b.suffix.lower() == '.npz' else None)
        jsn = a if a.suffix.lower() == '.json' else (b if b.suffix.lower() == '.json' else None)
        return npz, jsn
    else:
        p = Path(parts[0])
        if p.suffix.lower() == '.npz':
            # 常见映射：Data-512NPZ/<cls>/<stem>.npz -> FCSLabel/<cls>/<stem>.json 或 DSFNet-Output JSON 目录
            cand = [
                p.parents[2] / 'FCSLabel' / p.parents[0].name / (p.stem + '.json'),
                p.parents[2] / 'VTSLabel' / p.parents[0].name / (p.stem + '.json'),
            ]
            for c in cand:
                if c.exists():
                    return p, c
            return p, None
        elif p.suffix.lower() == '.json':
            # 反推 npz
            # 尝试 Data-512NPZ 同名路径
            try:
                cls = p.parent.name
                base = p.parents[2] / 'Data-512NPZ' / cls / (p.stem + '.npz')
                return (base if base.exists() else None), p
            except Exception:
                return None, p
        else:
            # 不认识的扩展名
            return None, None


def _load_spec(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as d:
        # 常见键：'spec_512' 或默认第一个数组
        if 'spec_512' in d:
            arr = d['spec_512']
        else:
            # 取第一个数组
            key0 = list(d.keys())[0]
            arr = d[key0]
    arr = np.asarray(arr).astype(np.float32)
    return arr


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    # 零均值/单位方差 + 轻缩放映射到[0,1]
    x = arr.astype(np.float32)
    x = x - float(x.mean())
    std = float(x.std())
    x = x / (std + 1e-6)
    x = np.clip(x * 0.2 + 0.5, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _save_png(arr_uint8: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_uint8, mode='L').save(out_path)


def _load_boxes_from_json(json_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    返回 [ (cls, cx, cy, w, h), ... ]，坐标归一化到 [0,1]
    兼容两种常见格式：
    - LabelMe: {'shapes': [{'label': '3', 'points': [[x1,y1],[x2,y2]], ...}], 'imageWidth': 512, 'imageHeight': 512}
    - 自定义: {'boxes': [[cls, cx, cy, w, h], ...]}  均为归一化坐标
    其他格式将被跳过并警告。
    """
    try:
        j = json.loads(json_path.read_text(encoding='utf-8'))
    except Exception as e:
        warnings.warn(f"Failed to read json: {json_path} ({e})")
        return []

    boxes = []
    # 1) 显式 boxes
    if isinstance(j, dict) and 'boxes' in j and isinstance(j['boxes'], list):
        for it in j['boxes']:
            try:
                c, cx, cy, w, h = it
                boxes.append((int(c), float(cx), float(cy), float(w), float(h)))
            except Exception:
                continue
        return boxes

    # 2) LabelMe
    if isinstance(j, dict) and 'shapes' in j and isinstance(j.get('imageWidth'), (int, float)):
        W = float(j.get('imageWidth', 512))
        H = float(j.get('imageHeight', 512))
        for shp in j['shapes']:
            try:
                lbl = shp.get('label', '0')
                c = int(lbl) if isinstance(lbl, str) and lbl.isdigit() else int(lbl)
                pts = shp.get('points', None)
                if not pts or len(pts) < 2:
                    continue
                x1, y1 = float(pts[0][0]), float(pts[0][1])
                x2, y2 = float(pts[1][0]), float(pts[1][1])
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                cx = (x1 + x2) / 2.0 / W
                cy = (y1 + y2) / 2.0 / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                if w > 0 and h > 0:
                    boxes.append((c, cx, cy, w, h))
            except Exception:
                continue
        return boxes

    warnings.warn(f"Unrecognized json schema: {json_path}")
    return []


def _remap_classes(boxes: List[Tuple[int, float, float, float, float]], exclude: Set[int]) -> Tuple[List[Tuple[int, float, float, float, float]], Dict[int, int]]:
    """
    排除 exclude 类别，并把剩余类别映射到连续的 0..K-1。
    返回 (新boxes, old->new 映射字典)
    """
    kept = [(c, cx, cy, w, h) for (c, cx, cy, w, h) in boxes if c not in exclude]
    uniq = sorted({c for (c, *_rest) in kept})
    mapping = {c: i for i, c in enumerate(uniq)}
    remapped = [(mapping[c], cx, cy, w, h) for (c, cx, cy, w, h) in kept]
    return remapped, mapping


def _find_list_file(data_root: Path, split: str, preferred_name: str) -> Path:
    """
    在 data_root 下查找指定 split 的列表文件，按照一系列常见命名进行回退：
    - 首选 preferred_name
    - 回退：train/val_json.list, train/val_json_small.list, train/val_list_fcs.txt, train/val.txt
    """
    candidates = [preferred_name]
    if split == 'train':
        candidates += ['train_json.list', 'train_json_small.list', 'train_list_fcs.txt', 'train.txt']
    else:
        candidates += ['val_json.list', 'val_json_small.list', 'val_list_fcs.txt', 'val.txt']
    for name in candidates:
        p = data_root / name
        if p.exists():
            return p
    # 最后兜底：仍返回首选路径（用于抛错信息）
    return data_root / preferred_name


def ensure_yolo_dataset(
    data_root: Path,
    cache_dir: Path,
    split_files: Tuple[str, str] = ('train_list.txt', 'val_list.txt'),
    exclude_classes: Set[int] | None = None,
    imgsz: int = 512,
) -> Path:
    """
    读取 data_root 下的列表文件，解析每行的 NPZ/JSON 路径；
    - 将 NPZ 转为 PNG（灰度），写入 cache_dir/images/{split}
    - 将 JSON 转为 YOLO TXT，写入 cache_dir/labels/{split}
    - 生成 dataset.yaml 并返回其路径
    """
    exclude = exclude_classes or set()
    cache_dir = cache_dir.resolve()
    (cache_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (cache_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (cache_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (cache_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    split_map = {'train': split_files[0], 'val': split_files[1]}

    # 用于统一类别映射（以 train 集合为准）
    global_mapping: Dict[int, int] | None = None

    for split, fname in split_map.items():
        list_path = _find_list_file(data_root, split, fname)
        if not list_path.exists():
            raise FileNotFoundError(f"List file not found for {split}: {list_path}")

        lines = _read_list_file(list_path)
        img_dir = cache_dir / 'images' / split
        lab_dir = cache_dir / 'labels' / split

        for s in lines:
            npz_path, json_path = _parse_line_to_paths(s)
            if npz_path is None or not npz_path.exists():
                warnings.warn(f"Skip (npz missing): {s}")
                continue
            if json_path is None or not json_path.exists():
                warnings.warn(f"Skip (json missing): {s}")
                continue

            # 生成目标文件名（以 json stem 为主）
            stem = json_path.stem
            img_out = img_dir / f"{stem}.png"
            txt_out = lab_dir / f"{stem}.txt"

            # 增量跳过：若 PNG 和 TXT 都已存在，直接跳过该样本
            if img_out.exists() and txt_out.exists():
                continue

            # 1) npz -> png
            try:
                if not img_out.exists():
                    spec = _load_spec(npz_path)
                    spec_u8 = _normalize_to_uint8(spec)
                    if spec_u8.shape != (imgsz, imgsz):
                        # 双线性缩放到目标尺寸
                        Image.fromarray(spec_u8, mode='L').resize((imgsz, imgsz), resample=Image.BILINEAR).save(img_out)
                    else:
                        _save_png(spec_u8, img_out)
            except Exception as e:
                warnings.warn(f"Failed to save image for {npz_path} -> {img_out} ({e})")
                continue

            # 2) json -> yolo txt（过滤/重映射类别）
            try:
                if not txt_out.exists():
                    boxes = _load_boxes_from_json(json_path)
                    boxes_remap, mapping = _remap_classes(boxes, exclude)
                    if global_mapping is None:
                        global_mapping = mapping
                    else:
                        # 容错：若 val 出现新类，扩展映射
                        for k, v in mapping.items():
                            if k not in global_mapping:
                                global_mapping[k] = max(global_mapping.values()) + 1
                        boxes_remap = [(global_mapping[c], cx, cy, w, h) for (c, cx, cy, w, h) in boxes if c in global_mapping]

                    with open(txt_out, 'w', encoding='utf-8') as f:
                        for c, cx, cy, w, h in boxes_remap:
                            # YOLO 需要严格裁剪到 [0,1]
                            cx = float(min(max(cx, 0.0), 1.0))
                            cy = float(min(max(cy, 0.0), 1.0))
                            w = float(min(max(w, 1e-6), 1.0))
                            h = float(min(max(h, 1e-6), 1.0))
                            f.write(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            except Exception as e:
                warnings.warn(f"Failed to save label for {json_path} -> {txt_out} ({e})")
                # 如失败，删除对应图像，保持成对
                try:
                    if img_out.exists():
                        img_out.unlink()
                except Exception:
                    pass
                continue

    # 生成 dataset.yaml（根据 global_mapping 推出 names）
    names = {}
    if global_mapping is None or len(global_mapping) == 0:
        # 兜底：22 类占位
        names = {i: f"c{i}" for i in range(22)}
    else:
        # old->new 的映射，构造 0..K-1 -> name
        inv = {v: k for k, v in global_mapping.items()}
        max_k = max(inv.keys())
        for i in range(max_k + 1):
            names[i] = f"c{inv.get(i, i)}"

    dataset_yaml = cache_dir / 'dataset.yaml'
    dataset_yaml.write_text(
        "\n".join([
            f"path: {cache_dir.as_posix()}",
            f"train: {(cache_dir / 'images' / 'train').as_posix()}",
            f"val:   {(cache_dir / 'images' / 'val').as_posix()}",
            "names:",
            *[f"  {k}: {v}" for k, v in names.items()],
            "",
        ]),
        encoding='utf-8'
    )
    return dataset_yaml


# ----------------------------
# 实时加载 Dataset（无缓存）：直接从 JSON/NPZ 读取
# 支持 Ultralytics YOLO 自定义 Dataset 集成
# ----------------------------

class YOLORealtimeDataset(TorchDataset):
    """
    直接从 JSON/NPZ 读取的实时 YOLO Dataset，无缓存。
    列表文件每行为纯 JSON 路径（或"npz_path json_path"），脚本自动反推。
    
    输出格式兼容 Ultralytics YOLO：
    - image: uint8 numpy array [H, W, 1]（灰度图）
    - bboxes: np.array [[cls, cx, cy, w, h], ...] 归一化到 [0,1]
    """
    def __init__(self, list_path: Path, exclude_classes: Set[int] | None = None, imgsz: int = 512):
        self.list_path = Path(list_path)
        self.items = _read_list_file(self.list_path)
        self.exclude = exclude_classes or set()
        self.imgsz = imgsz
        # 统计并建立类别映射（仅从训练集推导）
        self._build_class_mapping()
    
    def _build_class_mapping(self):
        """扫描所有样本，收集类别并建立 old->new 映射"""
        all_old_cls = set()
        for s in self.items[:100]:  # 采样 100 个以加速
            _, json_path = _parse_line_to_paths(s)
            if json_path and Path(json_path).exists():
                boxes = _load_boxes_from_json(Path(json_path))
                for c, *_ in boxes:
                    if c not in self.exclude:
                        all_old_cls.add(c)
        all_old_cls = sorted(all_old_cls)
        self.class_mapping = {c: i for i, c in enumerate(all_old_cls)}
        self.num_classes = len(self.class_mapping)
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx: int):
        s = self.items[idx]
        npz_path, json_path = _parse_line_to_paths(s)
        
        # 加载图像（NPZ -> 灰度 uint8）
        if npz_path is None or not npz_path.exists():
            warnings.warn(f"NPZ missing: {s}, skip")
            return None
        try:
            spec = _load_spec(Path(npz_path))
            spec_u8 = _normalize_to_uint8(spec)
            if spec_u8.shape != (self.imgsz, self.imgsz):
                spec_u8 = np.array(Image.fromarray(spec_u8).resize((self.imgsz, self.imgsz), Image.BILINEAR))
            # 转为 [H,W,1] 以满足 YOLO 输入格式
            image = spec_u8[:, :, np.newaxis].astype(np.uint8)
        except Exception as e:
            warnings.warn(f"Failed to load image {npz_path}: {e}")
            return None
        
        # 加载标注（JSON -> YOLO 格式）
        if json_path is None or not json_path.exists():
            bboxes = np.zeros((0, 5), dtype=np.float32)
        else:
            try:
                boxes = _load_boxes_from_json(Path(json_path))
                # 过滤并重映射类别
                boxes_remap = []
                for c, cx, cy, w, h in boxes:
                    if c in self.exclude or c not in self.class_mapping:
                        continue
                    new_c = self.class_mapping[c]
                    boxes_remap.append([new_c, cx, cy, w, h])
                bboxes = np.array(boxes_remap, dtype=np.float32) if boxes_remap else np.zeros((0, 5), dtype=np.float32)
            except Exception as e:
                warnings.warn(f"Failed to load boxes {json_path}: {e}")
                bboxes = np.zeros((0, 5), dtype=np.float32)
        
        return image, bboxes


def build_realtime_dataset_and_yaml(
    data_root: Path,
    cache_dir: Path,
    split_files: Tuple[str, str] = ('train_json.list', 'val_json.list'),
    exclude_classes: Set[int] | None = None,
    imgsz: int = 512,
) -> Tuple[YOLORealtimeDataset, YOLORealtimeDataset, Path, Dict[int, int]]:
    """
    构建实时加载 dataset（无缓存），生成 dataset.yaml。
    
    Args:
        data_root: 列表文件所在目录
        cache_dir: 输出 dataset.yaml 的目录
        split_files: (train_list_name, val_list_name)
        exclude_classes: 要排除的类别集合
        imgsz: 目标图像尺寸
    
    Returns:
        (train_dataset, val_dataset, yaml_path, class_mapping)
    """
    exclude = exclude_classes or set()
    data_root = Path(data_root)
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据集实例
    train_list = _find_list_file(data_root, 'train', split_files[0])
    val_list = _find_list_file(data_root, 'val', split_files[1])
    
    if not train_list.exists():
        raise FileNotFoundError(f"Train list not found: {train_list}")
    if not val_list.exists():
        raise FileNotFoundError(f"Val list not found: {val_list}")
    
    train_ds = YOLORealtimeDataset(train_list, exclude_classes=exclude, imgsz=imgsz)
    val_ds = YOLORealtimeDataset(val_list, exclude_classes=exclude, imgsz=imgsz)
    
    # 使用 train 集的类别映射作为标准
    names = {i: f"class_{old_c}" for old_c, i in train_ds.class_mapping.items()}
    
    # 生成 dataset.yaml
    yaml_path = cache_dir / 'dataset.yaml'
    yaml_path.write_text(
        "\n".join([
            f"path: {cache_dir.as_posix()}",
            "train: __realtime__",  # 标记为实时加载
            "val:   __realtime__",
            "nc: " + str(train_ds.num_classes),
            "names:",
            *[f"  {k}: {v}" for k, v in names.items()],
            "",
        ]),
        encoding='utf-8'
    )
    
    return train_ds, val_ds, yaml_path, train_ds.class_mapping


# ----------------------------
# 仅生成 YAML（假定图像/标签均已就绪）
# 不做任何 PNG/TXT 转换
# 目录结构要求：
#   cache_dir/
#     images/train/*.png
#     images/val/*.png
#     labels/train/*.txt
#     labels/val/*.txt
# ----------------------------

def build_dataset_yaml_only(cache_dir: Path) -> Path:
    cache_dir = Path(cache_dir).resolve()
    img_train = cache_dir / 'images' / 'train'
    img_val = cache_dir / 'images' / 'val'
    lab_train = cache_dir / 'labels' / 'train'
    lab_val = cache_dir / 'labels' / 'val'

    # 基本检查
    for p in [img_train, img_val, lab_train, lab_val]:
        if not p.exists():
            raise FileNotFoundError(f"Required directory missing: {p}")

    # 扫描标签，统计类别数
    classes: Set[int] = set()
    for lab_root in [lab_train, lab_val]:
        for txt in lab_root.rglob('*.txt'):
            try:
                for line in txt.read_text(encoding='utf-8', errors='ignore').splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        c = int(float(parts[0]))
                        classes.add(c)
            except Exception:
                continue

    if not classes:
        # 若无法统计，默认 1 类，避免训练崩溃
        classes = {0}

    nc = max(classes) + 1
    names = {i: f"c{i}" for i in range(nc)}

    dataset_yaml = cache_dir / 'dataset.yaml'
    dataset_yaml.write_text(
        "\n".join([
            f"path: {cache_dir.as_posix()}",
            f"train: {(img_train).as_posix()}",
            f"val:   {(img_val).as_posix()}",
            "names:",
            *[f"  {k}: {v}" for k, v in names.items()],
            "",
        ]),
        encoding='utf-8'
    )
    return dataset_yaml

