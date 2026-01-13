import argparse
import json
import os
import random
import time
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler  # deprecated
from torch.utils.data import DataLoader, Subset
# 新增：调度器
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
# 新增：保存参数与日志
import json

# 新增：限制OpenCV线程，避免Windows下I/O阻塞
try:
    import cv2
    cv2.setNumThreads(1)
except Exception:
    pass

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

import sys
FILE_DIR = os.path.dirname(__file__)
if FILE_DIR not in sys.path:
    sys.path.append(FILE_DIR)

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou, MultiScaleRoIAlign
from torchvision.ops import batched_nms  # 新增：用于TTA合并
from torchvision.models.detection.rpn import AnchorGenerator

from det_coco_dataset import CocoRFDataset
from dataset import seed_worker
# 新增：论文版骨干
from model import build_two_stream_det_backbone

# 新增：验证集GT快速统计
def _quick_count_gt(ds_obj) -> tuple:
    """返回 (pos_imgs, total_imgs, total_boxes)；尽量轻量，不做任何张量搬运。"""
    try:
        from torch.utils.data import Subset
        base_ds = ds_obj
        indices = None
        if isinstance(ds_obj, Subset):
            indices = ds_obj.indices
            base_ds = ds_obj.dataset
        idx_iter = indices if indices is not None else range(len(base_ds))
        pos_imgs = 0
        total_imgs = 0
        total_boxes = 0
        for i in idx_iter:
            try:
                _, tgt = base_ds[i]
                b = tgt.get('boxes', None)
                n = int(b.shape[0]) if (b is not None and hasattr(b, 'shape')) else (len(b) if b is not None else 0)
                total_boxes += n
                if n > 0:
                    pos_imgs += 1
                total_imgs += 1
            except Exception:
                total_imgs += 1
                continue
        return pos_imgs, total_imgs, total_boxes
    except Exception:
        return -1, -1, -1


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        pass


# 新增：EMA（用于检测模型）
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        self.ema = self._clone_model(model).eval()
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        # 缓存名称到参数/缓冲的映射，便于按名匹配且跳过形状不一致者
        self._ema_params = {n: p for n, p in self.ema.named_parameters()}
        self._ema_buffers = {n: b for n, b in self.ema.named_buffers()}

    def _clone_model(self, model: nn.Module) -> nn.Module:
        import copy
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self, model: nn.Module):
        # 轻量就地EMA：按名称匹配并做形状检查，避免检测模型中部分缓冲区尺寸变化导致崩溃
        d = self.decay
        one_minus_d = 1.0 - d
        # 参数
        for n, p in model.named_parameters():
            if not p.dtype.is_floating_point:
                continue
            ep = self._ema_params.get(n, None)
            if ep is None or ep.shape != p.shape:
                continue
            ep.data.mul_(d).add_(p.data, alpha=one_minus_d)
        # 浮点缓冲（如BN均值方差、BoxCoder/AnchorGenerator权重等）
        for n, b in model.named_buffers():
            if not (torch.is_tensor(b) and b.dtype.is_floating_point):
                continue
            eb = self._ema_buffers.get(n, None)
            if eb is None or eb.shape != b.shape:
                continue
            eb.data.copy_(b.data)


def build_model(num_classes: int, imagenet_backbone: bool = True, no_pretrained: bool = False, backbone_ckpt: str = "",
                use_twostream: bool = False, pos_weights: str = "", swin_weights: str = "", align_resize: bool = False,
                texture_yolov8m_cls: str = "", yolo_det: str = "",
                roi_pos_frac: float = 0.25, roi_batch_size: int = 512,
                roi_nms_thresh: float = 0.5, roi_fg_iou: float = 0.5, roi_bg_iou: float = 0.5) -> FasterRCNN:
    if use_twostream:
        # 论文版：两流 + PANet(out [192,384,768]) + Swin@P5 -> 统一 256 通道供 RPN/ROI
        backbone = build_two_stream_det_backbone(out_chs=(192,384,768), use_swin=True, align_resize=align_resize, pos_backbone='mobilenetv3')
        # 本地预训练加载（若提供路径）
        try:
            backbone.load_local_pretrained(pos_weights=pos_weights, swin_weights=swin_weights,
                                           texture_yolov8m_cls=texture_yolov8m_cls, yolo_det=yolo_det)
        except Exception as e:
            if tqdm is not None:
                tqdm.write(f"Warn: local pretrained load failed: {e}")
        # 小目标友好：更小的锚尺寸与更丰富的长宽比（保持3层金字塔以兼容当前backbone输出）
        anchor_generator = AnchorGenerator(
            sizes=((4, 8, 16), (8, 16, 32), (16, 32, 64)),
            aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 3
        )
        # FPN层级（当前backbone导出3层：'0','1','2'）。若后续扩展P2，则同步在此处加入'3'并在上方sizes增加一组尺寸。
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2'], output_size=7, sampling_ratio=2)
        # Faster R-CNN 头部参数（提升小目标召回，配合低阈值评估）
        model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            # 输入尺寸策略：与(543,512)相近，避免不必要放大/缩小
            min_size=544,
            max_size=1024,
            # RPN配置
            rpn_fg_iou_thresh=0.5,
            rpn_bg_iou_thresh=0.3,
            rpn_pre_nms_top_n_train=6000,
            rpn_post_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=3000,
            rpn_post_nms_top_n_test=1000,
            # ROI采样与NMS（可调）
            box_batch_size_per_image=int(roi_batch_size),
            box_positive_fraction=float(roi_pos_frac),
            box_score_thresh=0.0,   # 训练阶段不过早剪枝，评估由外部score_thr控制
            box_nms_thresh=float(roi_nms_thresh),
            box_fg_iou_thresh=float(roi_fg_iou),
            box_bg_iou_thresh=float(roi_bg_iou),
            box_detections_per_img=300,
        )
        if tqdm is not None:
            tqdm.write("Two-stream DET configured: small anchors + relaxed RPN/ROI for small objects (P3–P5).")
        return model
    # 否则回退到 ResNet50-FPN（占位）
    use_weights = (imagenet_backbone and not no_pretrained)
    weights = "DEFAULT" if use_weights else None
    weights_backbone = None if (no_pretrained or not imagenet_backbone) else "DEFAULT"
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=weights_backbone)
    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # 可选：从本地检测权重部分加载（如提供完整 fasterrcnn state_dict）
    if backbone_ckpt and os.path.isfile(backbone_ckpt):
        try:
            sd = torch.load(backbone_ckpt, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if tqdm is not None:
                tqdm.write(f"Loaded local ckpt (partial). missing={len(missing)} unexpected={len(unexpected)}")
        except Exception as e:
            if tqdm is not None:
                tqdm.write(f"Warn: failed to load backbone_ckpt: {e}")
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def to_device(images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]], device: torch.device):
    images = [img.to(device, non_blocking=True) for img in images]
    new_targets = []
    for t in targets:
        nt = {
            'boxes': t['boxes'].to(device, non_blocking=True),
            'labels': t['labels'].to(device, non_blocking=True),
            'image_id': t['image_id'].to(device, non_blocking=True),
        }
        if 'area' in t:
            nt['area'] = t['area'].to(device, non_blocking=True)
        if 'iscrowd' in t:
            nt['iscrowd'] = t['iscrowd'].to(device, non_blocking=True)
        new_targets.append(nt)
    return images, new_targets


def sanitize_targets(images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
    # Clip boxes to image size; ensure labels length matches boxes; drop degenerate boxes
    for i, t in enumerate(targets):
        if t.get('boxes', None) is None or t['boxes'].numel() == 0:
            # ensure empty labels/area tensors when boxes are empty
            if 'labels' in t and t['labels'].numel() > 0:
                t['labels'] = t['labels'][:0]
            if 'area' in t and t['area'].numel() > 0:
                t['area'] = t['area'][:0]
            continue
        _, H, W = images[i].shape
        boxes = t['boxes']
        labels = t['labels'] if 'labels' in t else None
        # clamp to image bounds
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H - 1)
        # align lengths (in case dataset returns mismatched sizes)
        if labels is not None:
            n = min(boxes.size(0), labels.size(0))
            if boxes.size(0) != n:
                boxes = boxes[:n]
            if labels.size(0) != n:
                labels = labels[:n]
        # remove degenerate boxes
        wh = boxes[:, 2:] - boxes[:, :2]
        keep = (wh[:, 0] >= 1) & (wh[:, 1] >= 1)
        if keep.numel() == 0 or keep.sum().item() == 0:
            # no valid boxes left
            t['boxes'] = boxes[:0]
            if labels is not None:
                t['labels'] = labels[:0]
            if 'area' in t:
                t['area'] = wh[:0, 0]
            continue
        boxes = boxes[keep]
        t['boxes'] = boxes
        if labels is not None:
            t['labels'] = labels[keep]
        if 'area' in t:
            t['area'] = (wh[keep, 0] * wh[keep, 1])


def greedy_match_iou(pred_boxes: torch.Tensor, pred_labels: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor, iou_thr: float = 0.5) -> Tuple[int, int, int]:
    # 返回 (TP, FP, FN) 用于简单评估
    if gt_boxes.numel() == 0 and pred_boxes.numel() == 0:
        return 0, 0, 0
    if pred_boxes.numel() == 0:
        return 0, 0, gt_boxes.size(0)
    if gt_boxes.numel() == 0:
        return 0, pred_boxes.size(0), 0
    ious = box_iou(pred_boxes, gt_boxes)  # (Np, Ng)
    tp, fp = 0, 0
    matched_gt = set()
    # 先按分数已排序传入，这里按每个预测寻找最佳匹配
    for i in range(pred_boxes.size(0)):
        best_j = -1
        best_iou = 0.0
        for j in range(gt_boxes.size(0)):
            if j in matched_gt:
                continue
            if pred_labels[i] != gt_labels[j]:
                continue
            iou = float(ious[i, j].item())
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1
    fn = gt_boxes.size(0) - len(matched_gt)
    return tp, fp, fn


def evaluate_simple(model: nn.Module, loader: DataLoader, device: torch.device, score_thr: float = 0.5, iou_thr: float = 0.5, log_interval: int = 0,
                    eval_max_dets: int = -1, eval_nms: float = -1.0, eval_rpn_post_nms_topn: int = -1,
                    eval_tta_hflip: bool = False) -> Dict[str, float]:
    model.eval()
    # 评估期临时收紧候选与NMS阈值，提高精度；评估完成后恢复
    old_max_dets = getattr(model.roi_heads, 'detections_per_img', None)
    old_nms = getattr(model.roi_heads, 'nms_thresh', None)
    old_rpn_test = None
    try:
        if eval_max_dets is not None and isinstance(eval_max_dets, int) and eval_max_dets > 0:
            model.roi_heads.detections_per_img = eval_max_dets
        if eval_nms is not None and isinstance(eval_nms, float) and eval_nms > 0:
            model.roi_heads.nms_thresh = eval_nms
        # RPN测试期候选上限
        if hasattr(model, 'rpn') and eval_rpn_post_nms_topn is not None and eval_rpn_post_nms_topn > 0:
            if hasattr(model.rpn, 'post_nms_top_n') and isinstance(model.rpn.post_nms_top_n, dict):
                old_rpn_test = model.rpn.post_nms_top_n.get('testing', None)
                model.rpn.post_nms_top_n['testing'] = eval_rpn_post_nms_topn
            elif hasattr(model.rpn, 'post_nms_top_n_test'):
                old_rpn_test = model.rpn.post_nms_top_n_test
                model.rpn.post_nms_top_n_test = eval_rpn_post_nms_topn
    except Exception:
        pass

    TP = FP = FN = 0
    use_periodic_log = (log_interval is not None and log_interval > 0)
    use_silent = (log_interval is not None and log_interval < 0)
    total_steps = len(loader)
    with torch.no_grad():
        if use_periodic_log or tqdm is None or use_silent:
            it = loader
            steps = 0
            for images, targets in it:
                steps += 1
                images, targets = to_device(list(images), list(targets), device)
                sanitize_targets(images, targets)
                if not eval_tta_hflip:
                    outputs = model(images)
                    outs = list(zip(outputs, targets))
                else:
                    # TTA: 水平翻转一次，并回投坐标后合并+NMS
                    outputs = model(images)
                    images_flipped = [torch.flip(im, dims=[2]) for im in images]
                    outputs_flip = model(images_flipped)
                    outs = []
                    for out, out_f, im, tgt in zip(outputs, outputs_flip, images, targets):
                        _, H, W = im.shape
                        boxes = out['boxes']
                        scores = out['scores']
                        labels = out['labels']
                        b2 = out_f['boxes']
                        if b2.numel() > 0:
                            x1 = W - 1 - b2[:, 2]
                            x2 = W - 1 - b2[:, 0]
                            b2 = torch.stack([x1, b2[:, 1], x2, b2[:, 3]], dim=1).clamp(0, max(W - 1, 1))
                        boxes_cat = boxes
                        scores_cat = scores
                        labels_cat = labels
                        if b2.numel() > 0:
                            boxes_cat = torch.cat([boxes_cat, b2], dim=0)
                            scores_cat = torch.cat([scores_cat, out_f['scores']], dim=0)
                            labels_cat = torch.cat([labels_cat, out_f['labels']], dim=0)
                        # 合并NMS
                        nms_thr = eval_nms if (eval_nms is not None and eval_nms > 0) else 0.5
                        keep_idx = batched_nms(boxes_cat, scores_cat, labels_cat, nms_thr)
                        if eval_max_dets is not None and eval_max_dets > 0:
                            # 先按分数排序再截断
                            order = torch.argsort(scores_cat[keep_idx], descending=True)
                            keep_idx = keep_idx[order][:eval_max_dets]
                        merged = {
                            'boxes': boxes_cat[keep_idx],
                            'scores': scores_cat[keep_idx],
                            'labels': labels_cat[keep_idx],
                        }
                        outs.append((merged, tgt))
                for out, tgt in outs:
                    scores = out['scores']
                    keep = scores >= score_thr
                    boxes_p = out['boxes'][keep]
                    labels_p = out['labels'][keep]
                    boxes_g = tgt['boxes']
                    labels_g = tgt['labels']
                    if boxes_p.numel() > 0:
                        order = torch.argsort(scores[keep], descending=True)
                        boxes_p = boxes_p[order]
                        labels_p = labels_p[order]
                    tp, fp, fn = greedy_match_iou(boxes_p, labels_p, boxes_g, labels_g, iou_thr=iou_thr)
                    TP += tp; FP += fp; FN += fn
                if use_periodic_log and (steps % log_interval == 0 or steps == total_steps):
                    print(f"Eval: {steps}/{total_steps}")
        else:
            it = tqdm(loader, desc="Eval", ncols=100, ascii=True)
            for images, targets in it:
                images, targets = to_device(list(images), list(targets), device)
                sanitize_targets(images, targets)
                if not eval_tta_hflip:
                    outputs = model(images)
                    outs = list(zip(outputs, targets))
                else:
                    outputs = model(images)
                    images_flipped = [torch.flip(im, dims=[2]) for im in images]
                    outputs_flip = model(images_flipped)
                    outs = []
                    for out, out_f, im, tgt in zip(outputs, outputs_flip, images, targets):
                        _, H, W = im.shape
                        boxes = out['boxes']
                        scores = out['scores']
                        labels = out['labels']
                        b2 = out_f['boxes']
                        if b2.numel() > 0:
                            x1 = W - 1 - b2[:, 2]
                            x2 = W - 1 - b2[:, 0]
                            b2 = torch.stack([x1, b2[:, 1], x2, b2[:, 3]], dim=1).clamp(0, max(W - 1, 1))
                        boxes_cat = boxes
                        scores_cat = scores
                        labels_cat = labels
                        if b2.numel() > 0:
                            boxes_cat = torch.cat([boxes_cat, b2], dim=0)
                            scores_cat = torch.cat([scores_cat, out_f['scores']], dim=0)
                            labels_cat = torch.cat([labels_cat, out_f['labels']], dim=0)
                        nms_thr = eval_nms if (eval_nms is not None and eval_nms > 0) else 0.5
                        keep_idx = batched_nms(boxes_cat, scores_cat, labels_cat, nms_thr)
                        if eval_max_dets is not None and eval_max_dets > 0:
                            order = torch.argsort(scores_cat[keep_idx], descending=True)
                            keep_idx = keep_idx[order][:eval_max_dets]
                        merged = {
                            'boxes': boxes_cat[keep_idx],
                            'scores': scores_cat[keep_idx],
                            'labels': labels_cat[keep_idx],
                        }
                        outs.append((merged, tgt))
                for out, tgt in outs:
                    scores = out['scores']
                    keep = scores >= score_thr
                    boxes_p = out['boxes'][keep]
                    labels_p = out['labels'][keep]
                    boxes_g = tgt['boxes']
                    labels_g = tgt['labels']
                    if boxes_p.numel() > 0:
                        order = torch.argsort(scores[keep], descending=True)
                        boxes_p = boxes_p[order]
                        labels_p = labels_p[order]
                    tp, fp, fn = greedy_match_iou(boxes_p, labels_p, boxes_g, labels_g, iou_thr=iou_thr)
                    TP += tp; FP += fp; FN += fn
            if hasattr(it, 'close'):
                it.close()
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    # 恢复评估前配置
    try:
        if old_max_dets is not None:
            model.roi_heads.detections_per_img = old_max_dets
        if old_nms is not None:
            model.roi_heads.nms_thresh = old_nms
        if old_rpn_test is not None:
            if hasattr(model.rpn, 'post_nms_top_n') and isinstance(model.rpn.post_nms_top_n, dict):
                model.rpn.post_nms_top_n['testing'] = old_rpn_test
            elif hasattr(model.rpn, 'post_nms_top_n_test'):
                model.rpn.post_nms_top_n_test = old_rpn_test
    except Exception:
        pass
    return {"precision@0.5": 100.0 * precision, "recall@0.5": 100.0 * recall, "f1@0.5": 100.0 * f1,
            "TP": float(TP), "FP": float(FP), "FN": float(FN)}


def _parse_thr_sweep(s: str) -> List[float]:
    """解析形如 "0.3:0.8:0.05" 或 "0.3,0.4,0.5" 的阈值扫参数，返回有效阈值列表(0~1)。"""
    vals: List[float] = []
    try:
        if not s:
            return vals
        s = s.strip()
        if ':' in s:
            parts = s.split(':')
            if len(parts) >= 2:
                start = float(parts[0]); end = float(parts[1])
                step = float(parts[2]) if len(parts) >= 3 else 0.05
                if step <= 0:
                    step = 0.05
                n = int(max(1, round((end - start) / step)))
                vals = [round(start + i * step, 6) for i in range(n + 1)]
        else:
            vals = [float(x) for x in s.split(',') if x.strip()]
    except Exception:
        vals = []
    # 过滤到 [0,1]
    return [v for v in vals if 0.0 <= v <= 1.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default=r"C:\\Users\\HP\\Desktop\\Exp\\LabelMePNG")
    parser.add_argument('--annotations', type=str, default=r"C:\\Users\\HP\\Desktop\\Exp\\LabelMePNG\\coco_labeled.json")
    parser.add_argument('--train_json', type=str, default='', help='If set, use this as fixed train COCO json')
    parser.add_argument('--val_json', type=str, default='', help='If set, use this as fixed val COCO json')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    # 旧版占位开关（默认关闭，避免联网）
    parser.add_argument('--backbone_imagenet', action='store_true', help='use ImageNet-pretrained backbone')
    parser.add_argument('--no_pretrained', action='store_true', help='disable any pretrained loading (default for offline)')
    parser.add_argument('--backbone_ckpt', type=str, default='', help='path to local fasterrcnn checkpoint (optional)')
    # 新增：论文版骨干相关
    parser.add_argument('--twostream', action='store_true', help='use two-stream+PANet+Swin detection backbone (paper version)')
    parser.add_argument('--pos_weights', type=str, default='', help='local path to MobileNetV3-Large weights (.pth)')
    parser.add_argument('--swin_weights', type=str, default='', help='local path to Swin-Small weights (.pth)')
    parser.add_argument('--texture_yolov8m_cls', type=str, default='', help='local path to yolov8m-cls.pt for initializing texture stream')
    parser.add_argument('--yolo_det', type=str, default='', help='local path to yolov8m.pt for fallback texture init')
    parser.add_argument('--align_resize', action='store_true', help='force nearest align resize when strides mismatch in fusion')
    parser.add_argument('--freeze_backbone', type=int, default=0, help='freeze backbone for first N epochs (two-stream only)')
    parser.add_argument('--backbone_lr_mult', type=float, default=0.25, help='LR multiplier for backbone params (two-stream only)')
    # 数据与评估
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--score_thr', type=float, default=0.5)
    parser.add_argument('--iou_thr', type=float, default=0.5)
    parser.add_argument('--dry_run', type=int, default=0, help='if >0, run only first N training batches then exit')
    parser.add_argument('--out_dir', type=str, default='', help='output dir for detection checkpoints (default: <project_root>/checkpoints_det)')
    # 新增：与分类脚本一致的训练增强与调度开关
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing scheduler')
    parser.add_argument('--step_size', type=int, default=30, help='StepLR step size (ignored if --cosine)')
    parser.add_argument('--gamma', type=float, default=0.1, help='StepLR gamma (ignored if --cosine)')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='linear warmup epochs before main scheduler')
    parser.add_argument('--ema', action='store_true', help='enable EMA for model weights (eval/save use EMA)')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay factor')
    parser.add_argument('--clip_grad', type=float, default=0.0, help='max grad norm (0 to disable)')
    # 新增：评估期收紧候选/阈值以提升精度
    parser.add_argument('--eval_max_dets', type=int, default=-1, help='eval-only: max detections per image (override)')
    parser.add_argument('--eval_nms', type=float, default=-1.0, help='eval-only: NMS threshold for ROI heads (override)')
    parser.add_argument('--eval_rpn_post_nms_topn', type=int, default=-1, help='eval-only: RPN post-NMS top-N (testing) override')
    parser.add_argument('--eval_tta_hflip', action='store_true', help='eval-only: enable horizontal flip TTA merge')
    # 新增：ROI训练期参数（求精导向）
    parser.add_argument('--roi_fg_iou', type=float, default=0.6, help='ROI positive IoU threshold (train)')
    parser.add_argument('--roi_bg_iou', type=float, default=0.4, help='ROI negative IoU threshold (train)')
    parser.add_argument('--roi_pos_frac', type=float, default=0.20, help='ROI positive fraction in batch (train)')
    parser.add_argument('--roi_batch_size', type=int, default=512, help='ROI samples per image (train)')
    # 新增：评估专用模式与阈值扫描
    parser.add_argument('--eval_only', action='store_true', help='only run evaluation, no training')
    parser.add_argument('--eval_ckpt', type=str, default='', help='path to checkpoint to evaluate (default: best.pt in out_dir)')
    parser.add_argument('--thr_sweep', type=str, default='', help='score threshold sweep, e.g. "0.3:0.8:0.05" or "0.3,0.4,0.5"')
    # 新增：日志输出频率控制
    parser.add_argument('--log_interval', type=int, default=0, help='if >0, disable tqdm and print one log every N batches')
    parser.add_argument('--eval_log_interval', type=int, default=0, help='eval logging: >0 periodic print, 0 tqdm, <0 silent')

    args = parser.parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    # 统一输出目录到项目根目录：Exp/Model/Output/50标签训练X（X 自增）
    if not args.out_dir or args.out_dir.strip() == '':
        project_root = Path(FILE_DIR).parent.parent  # .../TS‑Swin/src -> .../Exp
        base_out = project_root / 'Model' / 'Output'
        base_out.mkdir(parents=True, exist_ok=True)
        pat = re.compile(r'^50标签训练(\d+)$')
        max_idx = 0
        for d in base_out.iterdir():
            if d.is_dir():
                m = pat.match(d.name)
                if m:
                    try:
                        max_idx = max(max_idx, int(m.group(1)))
                    except Exception:
                        pass
        next_idx = max_idx + 1
        args.out_dir = str((base_out / f"50标签训练{next_idx}").resolve())
    os.makedirs(args.out_dir, exist_ok=True)

    # 启动时保存参数文件（包含 categories 参考）
    try:
        args_dict = vars(args).copy()
        if args.train_json:
            args_dict['categories_ref_json_for_val'] = args.train_json
        with open(os.path.join(args.out_dir, 'args.json'), 'w', encoding='utf-8') as f:
            json.dump(args_dict, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建数据集
    if args.train_json and args.val_json:
        print(f"Using fixed COCO splits:\n  train={args.train_json}\n  val={args.val_json}")
        ds_train = CocoRFDataset(images_dir=args.images_dir, annotations_json=args.train_json, augment=True)
        # 验证集使用训练集的categories作为参考，确保类别映射一致
        ds_val = CocoRFDataset(images_dir=args.images_dir, annotations_json=args.val_json, augment=False,
                               categories_ref_json=args.train_json)
    else:
        ds_train_full = CocoRFDataset(images_dir=args.images_dir, annotations_json=args.annotations, augment=True)
        ds_val_full = CocoRFDataset(images_dir=args.images_dir, annotations_json=args.annotations, augment=False)
        N = len(ds_train_full)
        idx_all = list(range(N))
        # 新增：按seed随机打乱索引后再划分，避免验证集偶然集中无目标样本
        rnd = random.Random(args.seed)
        rnd.shuffle(idx_all)
        n_val = int(N * max(0.0, min(0.9, args.val_split)))
        val_idx = idx_all[:n_val]
        train_idx = idx_all[n_val:]
        from torch.utils.data import Subset
        ds_train = Subset(ds_train_full, train_idx)
        ds_val = Subset(ds_val_full, val_idx)

    # 新增：启动时对验证集做一次GT统计自检
    try:
        _pos, _tot, _boxes = _quick_count_gt(ds_val)
        if _tot > 0:
            print(f"[Sanity] Val GT images with boxes: {_pos}/{_tot} | total boxes: {_boxes}")
    except Exception:
        pass

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=(args.workers > 0), drop_last=False, worker_init_fn=seed_worker, generator=g,
                              persistent_workers=(args.workers > 0), collate_fn=collate_fn)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=(args.workers > 0), drop_last=False, worker_init_fn=seed_worker, generator=g,
                            persistent_workers=(args.workers > 0), collate_fn=collate_fn)

    # 打印数据规模与每轮批次数，便于确认 5794 的来源
    try:
        print(f"Train images: {len(ds_train)} | Batches/epoch: {len(train_loader)} | Val images: {len(ds_val)}")
    except Exception:
        pass

    # 类别数：从数据集中读取映射（背景为 0）
    def _infer_num_classes(ds_obj) -> int:
        base = ds_obj
        for _ in range(3):
            if hasattr(base, 'contig_to_cat_id'):
                return len(getattr(base, 'contig_to_cat_id')) + 1
            base = getattr(base, 'dataset', None)
            if base is None:
                break
        # fallback
        return 2

    num_classes = _infer_num_classes(ds_train)

    # 构建模型（默认不联网下载权重；论文版更推荐）
    model = build_model(num_classes=num_classes,
                        imagenet_backbone=args.backbone_imagenet,
                        no_pretrained=True if args.twostream else args.no_pretrained,
                        backbone_ckpt=args.backbone_ckpt,
                        use_twostream=args.twostream,
                        pos_weights=args.pos_weights,
                        swin_weights=args.swin_weights,
                        align_resize=args.align_resize,
                        texture_yolov8m_cls=args.texture_yolov8m_cls,
                        yolo_det=args.yolo_det,
                        roi_pos_frac=args.roi_pos_frac,
                        roi_batch_size=args.roi_batch_size,
                        roi_nms_thresh=(args.eval_nms if args.eval_nms > 0 else 0.5),
                        roi_fg_iou=args.roi_fg_iou,
                        roi_bg_iou=args.roi_bg_iou).to(device)

    # eval-only：仅评估指定权重，支持阈值扫描与TTA
    if args.eval_only:
        ckpt_path = args.eval_ckpt if args.eval_ckpt else os.path.join(args.out_dir, 'best.pt')
        sd = torch.load(ckpt_path, map_location=device)
        state = sd.get('model', sd)
        model.load_state_dict(state, strict=False)
        thr_list = _parse_thr_sweep(args.thr_sweep) if args.thr_sweep else [args.score_thr]
        best = None
        for thr in thr_list:
            stats = evaluate_simple(model, val_loader, device, score_thr=thr, iou_thr=args.iou_thr,
                                    log_interval=args.eval_log_interval,
                                    eval_max_dets=args.eval_max_dets,
                                    eval_nms=args.eval_nms,
                                    eval_rpn_post_nms_topn=args.eval_rpn_post_nms_topn,
                                    eval_tta_hflip=args.eval_tta_hflip)
            print(f"Eval thr={thr:.3f} | P {stats['precision@0.5']:.2f} R {stats['recall@0.5']:.2f} F1 {stats['f1@0.5']:.2f}")
            if (best is None) or (stats['f1@0.5'] > best['f1@0.5']):
                best = {**stats, 'thr': thr}
        if best is not None:
            print(f"Best@thr={best['thr']:.3f} -> P {best['precision@0.5']:.2f} R {best['recall@0.5']:.2f} F1 {best['f1@0.5']:.2f}")
        return

    # 参数组：两流骨干分组 LR；其它维持基准 LR
    if args.twostream:
        backbone_params = []
        head_params = []
        for n, p in model.named_parameters():
            if n.startswith('backbone.'):
                backbone_params.append(p)
            else:
                head_params.append(p)
        param_groups = [
            {'params': [p for p in backbone_params if p.requires_grad], 'lr': args.lr * args.backbone_lr_mult},
            {'params': [p for p in head_params if p.requires_grad], 'lr': args.lr},
        ]
    else:
        param_groups = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # 调度器：Warmup + 主调度器
    if args.cosine:
        main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        main_scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    warmup_epochs = max(0, int(args.warmup_epochs))
    if warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            return 1.0
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        warmup_scheduler = None

    # AMP（新版API）
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    ema = ModelEMA(model, decay=args.ema_decay, device=device) if args.ema else None

    best_f1 = -1.0
    # 规范化文件名
    best_path = os.path.join(args.out_dir, 'best.pt')
    last_path = os.path.join(args.out_dir, 'last.pt')
    log_csv = os.path.join(args.out_dir, 'log.csv')

    # 移除 results_csv，改为在 log.csv 中记录全部常用指标
    # if (not args.results_csv) or args.results_csv.strip() == '':
    #     args.results_csv = os.path.join(args.out_dir, 'results_det.csv')
    # try:
    #     if not os.path.exists(args.results_csv):
    #         with open(args.results_csv, 'w', encoding='utf-8', newline='') as f:
    #             f.write('epoch,lr,loss,precision@0.5,recall@0.5,f1@0.5,TP,FP,FN\n')
    # except Exception:
    #     pass

    # 冻结/解冻逻辑（两流骨干）
    def set_backbone_trainable(flag: bool):
        if hasattr(model, 'backbone'):
            for p in model.backbone.parameters():
                p.requires_grad = flag

    if args.twostream and args.freeze_backbone > 0:
        set_backbone_trainable(False)

    for epoch in range(args.epochs):
        # 触发解冻
        if args.twostream and args.freeze_backbone > 0 and epoch == args.freeze_backbone:
            set_backbone_trainable(True)
        model.train()
        total_steps = len(train_loader)
        # 当指定 log_interval 时，关闭 tqdm，改为周期性打印
        use_periodic_log = (args.log_interval is not None and args.log_interval > 0)
        it = train_loader if (use_periodic_log or tqdm is None) else tqdm(train_loader, desc=f"Train {epoch+1}/{args.epochs}", ncols=100, ascii=True)
        running_loss = 0.0
        n_images = 0
        steps = 0
        for images, targets in it:
            images, targets = to_device(list(images), list(targets), device)
            sanitize_targets(images, targets)
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.amp.autocast("cuda"):
                    losses = model(images, targets)
                    loss = sum(v for v in losses.values())
                scaler.scale(loss).backward()
                if args.clip_grad and args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses = model(images, targets)
                loss = sum(v for v in losses.values())
                loss.backward()
                if args.clip_grad and args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
            if ema is not None:
                ema.update(model)
            bs = len(images)
            running_loss += float(loss.item()) * bs
            n_images += bs
            steps += 1
            if use_periodic_log:
                if (steps % args.log_interval == 0) or (steps == total_steps):
                    avg_loss = (running_loss / max(n_images, 1))
                    print(f"Train {epoch+1}/{args.epochs}: {steps}/{total_steps} loss={avg_loss:.4f}")
            else:
                if tqdm is not None:
                    avg_loss = (running_loss / max(n_images, 1))
                    it.set_postfix({"loss": f"{avg_loss:.4f}"})
            if args.dry_run > 0 and steps >= args.dry_run:
                break
        if tqdm is not None and hasattr(it, 'close') and not use_periodic_log:
            it.close()

        if args.dry_run > 0:
            print(f"Dry-run done. Seen batches: {steps}. Avg loss: {(running_loss/max(n_images,1)):.4f}")
            return

        # 调度器步进
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        # 验证（优先使用 EMA）
        eval_model = ema.ema if ema is not None else model
        stats = evaluate_simple(
            eval_model,
            val_loader,
            device,
            score_thr=args.score_thr,
            iou_thr=args.iou_thr,
            log_interval=args.eval_log_interval,
            eval_max_dets=args.eval_max_dets,
            eval_nms=args.eval_nms,
            eval_rpn_post_nms_topn=args.eval_rpn_post_nms_topn,
            eval_tta_hflip=args.eval_tta_hflip,
        )
        avg_loss = (running_loss / max(n_images, 1))
        # 当前学习率（取各 param group 平均）
        try:
            lrs = [pg.get('lr', 0.0) for pg in optimizer.param_groups]
            lr_now = float(sum(lrs) / max(1, len(lrs)))
        except Exception:
            lr_now = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} | loss {avg_loss:.4f} | P {stats['precision@0.5']:.2f} R {stats['recall@0.5']:.2f} F1 {stats['f1@0.5']:.2f}")

        # 统一写 log.csv（包含 lr）
        try:
            if not os.path.exists(log_csv):
                with open(log_csv, 'w', encoding='utf-8') as f:
                    f.write('epoch,loss,precision@0.5,recall@0.5,f1@0.5,TP,FP,FN\n')
            with open(log_csv, 'a', encoding='utf-8') as f:
                f.write(f"{epoch+1},{lr_now:.6g},{avg_loss:.6f},{stats['precision@0.5']:.6f},{stats['recall@0.5']:.6f},{stats['f1@0.5']:.6f},{int(stats['TP'])},{int(stats['FP'])},{int(stats['FN'])}\n")
        except Exception:
            pass

        # 保存 last.pt（使用 EMA 权重以便推理稳定）
        ckpt_last = {
            'epoch': epoch,
            'model': (ema.ema.state_dict() if ema is not None else model.state_dict()),
            'optimizer': optimizer.state_dict(),
            'scheduler': (main_scheduler.state_dict() if hasattr(main_scheduler, 'state_dict') else {}),
            'args': vars(args),
            'stats': stats,
        }
        try:
            torch.save(ckpt_last, last_path)
        except Exception:
            pass

        # 保存 best.pt
        if stats['f1@0.5'] > best_f1:
            best_f1 = stats['f1@0.5']
            try:
                torch.save(ckpt_last, best_path)
            except Exception:
                pass

    print(f"Done. Best F1@0.5: {best_f1:.2f}. Checkpoint: {best_path}")


if __name__ == '__main__':
    main()
