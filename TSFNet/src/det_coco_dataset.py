"""
COCO 风格检测数据集（用于 RF 谱图），集成 RFAugmentor。
- 读取由 tools/labelme_to_coco.py 生成的 COCO JSON
- 打开图像自 LabelMePNG 镜像目录（背景/unknown 图像以空标注的方式存在）
- 训练/验证统一 resize 到 img_size，并同步按比例缩放 bbox
- 训练时应用 src/aug_rf.py 中的几何+非几何增强，并同步更新 bbox

注意：
- 增强默认在灰度域进行，输出 3 通道图像（灰度三通道），数值范围[0,1]
- 目标框使用 COCO xywh 存储；对接 torchvision 检测模型时会转换为 xyxy
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# from .aug_rf import RFAugmentor, RFAugConfig
# from .dataset import seed_worker
# 改为同目录绝对导入，便于作为脚本/模块使用
from aug_rf import RFAugmentor, RFAugConfig
from dataset import seed_worker

Box = List[float]  # [x,y,w,h] in pixels (float)


def _ensure_3ch_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return img


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    # boxes: (N,4) [x,y,w,h] -> [x1,y1,x2,y2]
    out = boxes.copy()
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


@dataclass
class CocoPaths:
    images_dir: str
    annotations_json: str


class CocoRFDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        annotations_json: str,
        img_size: int = 512,
        augment: bool = True,
        rf_aug: Optional[RFAugmentor] = None,
        filter_empty_images: bool = False,
        category_whitelist: Optional[List[int]] = None,
        categories_ref_json: Optional[str] = None,
    ) -> None:
        """
        images_dir: LabelMePNG 根目录（包含 0..23 子目录和图片文件）
        annotations_json: COCO 标注文件路径
        img_size: 读取后统一 resize 到正方形（默认 512）
        augment: 是否启用增强（仅训练集 True）
        rf_aug: 可传入自定义 RFAugmentor；不传则用默认配置
        filter_empty_images: 是否过滤掉无标注图片（默认 False：保留背景图做开集）
        category_whitelist: 只保留 id 在该列表内的类别（None 则保留全部）
        categories_ref_json: 使用该 JSON 的 categories 列表来建立一致的类别映射（保证train/val相同映射）
        """
        super().__init__()
        self.root = Path(images_dir).resolve()
        self.ann_path = Path(annotations_json).resolve()
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.filter_empty = bool(filter_empty_images)
        self.rf_aug = rf_aug or RFAugmentor(RFAugConfig(img_size=self.img_size))

        with open(self.ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        # 类别映射（可能非连续 id）；可从参考JSON中读取，确保一致
        cats_ref = None
        if categories_ref_json:
            try:
                with open(categories_ref_json, "r", encoding="utf-8") as rf:
                    coco_ref = json.load(rf)
                    cats_ref = coco_ref.get("categories", None)
            except Exception:
                cats_ref = None
        cats = cats_ref if cats_ref is not None else coco.get("categories", [])
        self.cat_id_to_contig: Dict[int, int] = {}
        self.contig_to_cat_id: List[int] = []
        if category_whitelist is None:
            ordered = cats
        else:
            wl = set(category_whitelist)
            ordered = [c for c in cats if c.get("id") in wl]
        for i, c in enumerate(sorted(ordered, key=lambda x: x.get("id", 0))):
            self.cat_id_to_contig[int(c["id"])] = i + 1  # 从 1 开始
            self.contig_to_cat_id.append(int(c["id"]))

        # 建立图像与标注索引
        self.images = coco.get("images", [])
        anns = coco.get("annotations", [])
        self.imgid_to_anns: Dict[int, List[Dict]] = {}
        for a in anns:
            if category_whitelist is not None and int(a.get("category_id")) not in self.cat_id_to_contig:
                continue
            img_id = int(a["image_id"])
            self.imgid_to_anns.setdefault(img_id, []).append(a)

        # 可选：过滤空图
        if self.filter_empty:
            self.images = [im for im in self.images if int(im.get("id")) in self.imgid_to_anns]

        # 记录 file_name 绝对路径
        self.file_paths: List[Tuple[int, Path]] = []  # (image_id, abs_path)
        for im in self.images:
            fid = int(im["id"])
            fname = im.get("file_name")
            p = Path(fname)
            if not p.is_absolute():
                p = (self.root / fname).resolve()
            else:
                p = p.resolve()
            self.file_paths.append((fid, p))

    def __len__(self) -> int:
        return len(self.file_paths)

    def _load_image(self, path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        H0, W0 = img.shape[:2]
        if H0 != self.img_size or W0 != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return img, (H0, W0)  # 返回resize后的图与原始尺寸

    def __getitem__(self, idx: int):
        img_id, path = self.file_paths[idx]
        gray, (H0, W0) = self._load_image(path)

        # 收集该图像的 COCO xywh 框和类别
        anns = self.imgid_to_anns.get(img_id, [])
        boxes_xywh: List[Box] = []
        labels: List[int] = []
        for a in anns:
            x, y, w, h = [float(v) for v in a.get("bbox", [0, 0, 0, 0])]
            if w <= 1e-3 or h <= 1e-3:
                continue
            boxes_xywh.append([x, y, w, h])
            cat_id = int(a.get("category_id"))
            labels.append(self.cat_id_to_contig.get(cat_id, 0))

        # 若图片在读取时被resize到 img_size，则将标注同步缩放到 img_size 坐标系
        if (H0, W0) != (self.img_size, self.img_size) and boxes_xywh:
            sx = float(self.img_size) / max(W0, 1)
            sy = float(self.img_size) / max(H0, 1)
            for i in range(len(boxes_xywh)):
                x, y, w, h = boxes_xywh[i]
                boxes_xywh[i] = [x * sx, y * sy, w * sx, h * sy]

        # 应用 RF 增强（训练集）；验证集不增强
        if self.augment:
            # 传入的 gray 已是 img_size；boxes 已处于同一坐标系
            img_aug, boxes_aug = self.rf_aug(gray, boxes_xywh, apply_color_map=False)
        else:
            img_aug = gray
            boxes_aug = boxes_xywh

        # 转 3 通道，标注转为张量
        img_aug = _ensure_3ch_uint8(img_aug)
        img_t = torch.from_numpy(img_aug.transpose(2, 0, 1)).float() / 255.0  # (3,H,W)

        if boxes_aug:
            boxes_arr = np.array(boxes_aug, dtype=np.float32)
            boxes_xyxy = _xywh_to_xyxy(boxes_arr)
            areas = boxes_arr[:, 2] * boxes_arr[:, 3]
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            boxes_t = torch.from_numpy(boxes_xyxy)
            area_t = torch.from_numpy(areas)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area_t = torch.zeros((0,), dtype=torch.float32)

        target = {
            "image_id": torch.as_tensor([img_id], dtype=torch.int64),
            "boxes": boxes_t,  # xyxy
            "labels": labels_t,  # 1..K
            "area": area_t,
            "iscrowd": torch.zeros((labels_t.numel(),), dtype=torch.int64),
            "orig_size": torch.as_tensor([self.img_size, self.img_size], dtype=torch.int64),
            "path": str(path),
        }
        return img_t, target


def create_coco_rf_dataloader(
    images_dir: str,
    annotations_json: str,
    batch_size: int = 4,
    num_workers: int = 2,
    augment: bool = True,
    rf_aug_cfg: Optional[RFAugConfig] = None,
    filter_empty_images: bool = False,
    category_whitelist: Optional[List[int]] = None,
    seed: int = 42,
) -> DataLoader:
    rf_aug = RFAugmentor(rf_aug_cfg or RFAugConfig())
    ds = CocoRFDataset(
        images_dir=images_dir,
        annotations_json=annotations_json,
        augment=augment,
        rf_aug=rf_aug,
        filter_empty_images=filter_empty_images,
        category_whitelist=category_whitelist,
    )
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
        collate_fn=lambda batch: tuple(zip(*batch)),  # for variable-size targets
    )
    return loader
