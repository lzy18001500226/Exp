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
import torch.nn.functional as F
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
# 新增：用于级联ROI裁剪
from torchvision.ops import roi_align

from det_coco_dataset import CocoRFDataset
from dataset import seed_worker
# 新增：论文版骨干
from model import build_two_stream_det_backbone

from torch.nn import functional as F  # KD用

# 新增：argparse 布尔解析器（支持 true/false/1/0/on/off/yes/no.
def str2bool(v):
    """Robust bool parser for argparse that accepts true/false/1/0/on/off/yes/no.
    用法: parser.add_argument('--flag', type=str2bool, nargs='?', const=True, default=False)
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("y", "yes", "t", "true", "1", "on"):
        return True
    if s in ("n", "no", "f", "false", "0", "off"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false/1/0)")


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
	# 初始化即深拷贝一份当前模型参数作为 EMA，并与当前模型权重对齐，保证首轮评估不为随机权重。
	def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
		import copy as _copy
		self.decay = float(decay)
		self.device = device
		self.updates = 0  # 新增：记录已累计的EMA更新步数
		# 深拷贝结构与权重，设为 eval 并冻结梯度
		self.ema = _copy.deepcopy(model).eval()
		for p in self.ema.parameters():
			p.requires_grad_(False)
		if device is not None:
			self.ema.to(device)
		# 用当前模型权重进行一次性同步，避免“未同步 EMA”导致前几轮评估为 0
		self._load_from_model(model)

	@torch.no_grad()
	def _load_from_model(self, model: nn.Module):
		# 严格对齐初始权重与 buffers
		self.ema.load_state_dict(model.state_dict(), strict=True)

	@torch.no_grad()
	def update(self, model: nn.Module):
		# 对可学习参数做滑动平均；对 buffers（如BN的running_mean/var等）直接复制
		ema_sd = self.ema.state_dict()
		msd = model.state_dict()
		# 识别buffers名称，便于区分参数与buffers
		buf_keys = set(dict(self.ema.named_buffers()).keys())
		for k, v_ema in ema_sd.items():
			v = msd.get(k, None)
			if v is None:
				continue
			if k in buf_keys:
				# buffers：直接对齐（保持dtype/device一致）
				v_ema.copy_(v.detach().to(v_ema.device, dtype=v_ema.dtype))
			else:
				# 参数：仅对浮点做EMA，其他（例如整型）直接复制
				if v_ema.dtype.is_floating_point:
					v_ema.mul_(self.decay).add_(v.detach().to(v_ema.device, dtype=v_ema.dtype), alpha=(1.0 - self.decay))
				else:
					v_ema.copy_(v)
		self.updates += 1

	def state_dict(self):
		# 便于与现有保存逻辑兼容
		return self.ema.state_dict()


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
    """Clip boxes to image size; ensure x1<=x2,y1<=y2; drop degenerate/NaN boxes; align labels length."""
    for i, t in enumerate(targets):
        if t is None:
            continue
        img = images[i]
        h, w = int(img.shape[-2]), int(img.shape[-1])
        boxes = t.get('boxes', None)
        labels = t.get('labels', None)
        if boxes is None or not torch.is_tensor(boxes) or boxes.numel() == 0:
            t['boxes'] = torch.empty((0, 4), device=img.device, dtype=torch.float32)
            t['labels'] = torch.empty((0,), device=img.device, dtype=torch.int64)
            if 'area' in t:
                t['area'] = torch.empty((0,), device=img.device, dtype=torch.float32)
            if 'iscrowd' in t:
                t['iscrowd'] = torch.zeros((0,), device=img.device, dtype=torch.int64)
            continue
        boxes = boxes.to(img.device, dtype=torch.float32)
        # 修正顺序并裁剪到图像内
        x1, y1, x2, y2 = boxes.unbind(1)
        x_min = torch.minimum(x1, x2).clamp_(0, max(0, w - 1))
        y_min = torch.minimum(y1, y2).clamp_(0, max(0, h - 1))
        x_max = torch.maximum(x1, x2).clamp_(0, max(0, w - 1))
        y_max = torch.maximum(y1, y2).clamp_(0, max(0, h - 1))
        boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
        # 过滤退化或非有限
        bw = (boxes[:, 2] - boxes[:, 0])
        bh = (boxes[:, 3] - boxes[:, 1])
        finite_mask = torch.isfinite(boxes).all(dim=1)
        keep = (bw > 1e-2) & (bh > 1e-2) & finite_mask
        if labels is not None and torch.is_tensor(labels):
            labels = labels.to(img.device, dtype=torch.int64)
            # 对齐长度（取最小）
            n = min(labels.numel(), boxes.size(0))
            boxes = boxes[:n]
            labels = labels[:n]
            keep = keep[:n]
        else:
            labels = torch.zeros((boxes.size(0),), device=img.device, dtype=torch.int64)
        boxes = boxes[keep]
        labels = labels[keep]
        t['boxes'] = boxes
        t['labels'] = labels
        if 'area' in t:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            t['area'] = area
        if 'iscrowd' in t:
            # 若长度不匹配，重置为0
            if not torch.is_tensor(t['iscrowd']) or t['iscrowd'].numel() != boxes.size(0):
                t['iscrowd'] = torch.zeros((boxes.size(0),), device=img.device, dtype=torch.int64)


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


def evaluate_simple(model, data_loader, device,
                    # 基础评估参数
                    score_thr: float = 0.5,
                    iou_thr: float = 0.5,
                    log_interval: int = 0,
                    eval_tta_hflip: bool = False,
                    # 评估期覆盖阈值
                    eval_max_dets: Optional[int] = None,
                    eval_nms: Optional[float] = None,
                    eval_rpn_post_nms_top_n: Optional[int] = None,
                    # 开放集能量过滤
                    energy_open_set: bool = False,
                    gamma: float = 5.0,
                    energy_temp: float = 1.0,
                    energy_exclude_bg: bool = False,
                    # P0: 打分/后处理
                    conf_temp: float = 1.0,
                    energy_rescore: bool = False,
                    rescore_mode: str = 'none',
                    rescore_k: float = 10.0,
                    rescore_lambda: float = 0.5,
                    soft_nms: bool = False,
                    soft_nms_sigma: float = 0.5,
                    soft_nms_iou: float = 0.5,
                    # 级联：ConvNeXt + LogitNorm + 自适应γ2 + 双重重打分 + 稳健衰减
                    cascade_enable: bool = True,
                    cascade_model: Optional[nn.Module] = None,
                    cascade_model_name: str = 'convnext_large',
                    cascade_ckpt_path: Optional[str] = None,
                    cascade_input_size: int = 224,
                    cascade_use_logitnorm: bool = True,
                    cascade_temp: float = 1.0,
                    cascade_gamma2: float = 5.0,
                    cascade_adapt_gamma2: bool = True,
                    cascade_gamma2_min: float = 3.5,
                    cascade_gamma2_max: float = 8.5,
                    cascade_gamma2_w_area: float = 0.4,
                    cascade_gamma2_w_score: float = 0.4,
                    cascade_gamma2_w_context: float = 0.2,
                    cascade_dual_rescore: bool = True,
                    cascade_dual_alpha: float = 0.55,
                    cascade_sig_k2: float = 1.0,
                    cascade_decay_enable: bool = True,
                    cascade_decay_iou: float = 0.5,
                    cascade_decay_alpha: float = 0.5,
                    **kwargs):
    """简易评估（可选开放集能量过滤 + P0 后处理 + 最小版级联）。"""
    model.eval()
    from torchvision.ops.boxes import clip_boxes_to_image
    # 调试标志与计数器，避免 NameError
    dbg_enabled = bool(kwargs.get('debug_eval', False))
    dbg_interval = int(kwargs.get('debug_eval_interval', 0) or 0)
    dbg_seen = 0
    dbg_tot_pred = 0
    dbg_tot_kept = 0
    dbg_tot_gt = 0
    dbg_iou_sum = 0.0

    # 临时覆盖测试期设置
    old_max_dets = getattr(model.roi_heads, 'detections_per_img', None)
    old_nms = getattr(model.roi_heads, 'nms_thresh', None)
    old_rpn_test = None
    try:
        if isinstance(eval_max_dets, int) and eval_max_dets > 0:
            model.roi_heads.detections_per_img = eval_max_dets
        if isinstance(eval_nms, float) and eval_nms > 0:
            model.roi_heads.nms_thresh = eval_nms
        if hasattr(model, 'rpn') and isinstance(eval_rpn_post_nms_top_n, int) and eval_rpn_post_nms_top_n > 0:
            if hasattr(model.rpn, 'post_nms_top_n') and isinstance(model.rpn.post_nms_top_n, dict):
                old_rpn_test = model.rpn.post_nms_top_n.get('testing', None)
                model.rpn.post_nms_top_n['testing'] = eval_rpn_post_nms_top_n
            elif hasattr(model.rpn, 'post_nms_top_n_test'):
                old_rpn_test = model.rpn.post_nms_top_n_test
                model.rpn.post_nms_top_n_test = eval_rpn_post_nms_top_n
    except Exception:
        pass

    def _detect_with_energy(images):
        # 原有检测逻辑
        t_out = model.transform(images)
        # torchvision GeneralizedRCNNTransform.__call__ 返回 (images, targets)
        t_inp = t_out[0] if isinstance(t_out, (tuple, list)) else t_out
        img_tensors = t_inp.tensors
        img_sizes = t_inp.image_sizes
        features = model.backbone(img_tensors)
        if isinstance(features, torch.Tensor):
            features = {0: features}
        proposals, _ = model.rpn(t_inp, features)
        box_features = model.roi_heads.box_roi_pool(features, proposals, img_sizes)
        box_features = model.roi_heads.box_head(box_features)
        class_logits, box_regression = model.roi_heads.box_predictor(box_features)
        # energy per ROI
        if energy_exclude_bg and class_logits.size(1) > 1:
            logits_use = class_logits[:, 1:] / max(energy_temp, 1.0e-6)
        else:
            logits_use = class_logits / max(energy_temp, 1.0e-6)
        energy_all = (-torch.logsumexp(logits_use, dim=1)).detach()
        # 温度缩放用于分数
        conf_t = max(float(conf_temp), 1.0e-6)
        logits_for_score = class_logits / conf_t
        score_base_thresh = float(getattr(model.roi_heads, 'score_thresh', 0.0))
        nms_thr = float(eval_nms if (eval_nms is not None and eval_nms > 0) else getattr(model.roi_heads, 'nms_thresh', 0.5))
        detections_per_img = int(eval_max_dets if (eval_max_dets is not None and eval_max_dets > 0) else getattr(model.roi_heads, 'detections_per_img', 300))
        box_coder = model.roi_heads.box_coder
        # split per image
        boxes_per_image = [len(p) for p in proposals]
        logits_split = logits_for_score.split(boxes_per_image, dim=0)
        raw_logits_split = class_logits.split(boxes_per_image, dim=0)
        regs_split = box_regression.split(boxes_per_image, dim=0)
        energy_split = energy_all.split(boxes_per_image, dim=0)
        results = []
        for img_idx, (logits_i, raw_logits_i, reg_i, props_i, energy_i, img_size) in enumerate(zip(logits_split, raw_logits_split, regs_split, proposals, energy_split, img_sizes)):
            scores_i = F.softmax(logits_i, dim=-1)
            num_classes = scores_i.shape[-1]
            # 按类别解码回归（与 torchvision ROIHeads 一致）
            boxes_all: List[torch.Tensor] = []
            scores_all: List[torch.Tensor] = []
            labels_all: List[torch.Tensor] = []
            energies_all: List[torch.Tensor] = []
            for c in range(1, num_classes):
                sc = scores_i[:, c]
                keep0 = sc > score_base_thresh
                if keep0.sum().item() == 0:
                    continue
                # 每类 4 维回归切片
                reg_c = reg_i[:, 4 * c: 4 * (c + 1)]
                # 兼容 torchvision: decode 期望 boxes 为 List[Tensor]；返回可能是 List 或 Tensor
                _dec = box_coder.decode(reg_c, [props_i])
                boxes_c = _dec[0] if isinstance(_dec, (list, tuple)) else _dec
                boxes_c = clip_boxes_to_image(boxes_c, img_size)
                b = boxes_c[keep0]
                s = sc[keep0]
                e = energy_i[keep0]
                # 形状/类型安全：确保传入 NMS 的张量维度正确
                if b.dim() == 3 and b.size(0) == 1:
                    b = b.squeeze(0)
                if b.dim() != 2:
                    b = b.reshape(-1, 4)
                if s.dim() != 1:
                    s = s.reshape(-1)
                if e.dim() != 1:
                    e = e.reshape(-1)
                b = b.contiguous().to(dtype=boxes_c.dtype)
                s = s.contiguous()
                # 能量重打分（可选）
                if energy_rescore:
                    if rescore_mode == 'sigmoid':
                        k = float(rescore_k)
                        g = float(gamma)
                        s = s * torch.sigmoid(k * (g - e))
                    else:  # linear
                        lam = float(rescore_lambda)
                        s = s - lam * e
                    s = torch.clamp(s, 0.0, 1.0)
                # NMS/Soft-NMS
                if b.numel() == 0:
                    continue
                if soft_nms:
                    try:
                        order = torch.argsort(s, descending=True)
                        b_sorted = b[order]; s_sorted = s[order]
                        sigma = float(soft_nms_sigma)
                        iou_t = float(soft_nms_iou)
                        for i in range(b_sorted.size(0)):
                            if s_sorted[i] <= 0:
                                continue
                            ious = box_iou(b_sorted[i].unsqueeze(0), b_sorted[i+1:]).squeeze(0)
                            decay = torch.exp(- (ious * ious) / max(1.0e-6, sigma))
                            s_sorted[i+1:] = torch.where(ious > iou_t, s_sorted[i+1:] * decay, s_sorted[i+1:])
                        s = s_sorted
                        b = b_sorted
                        keep_idx = torch.nonzero(s > score_base_thresh, as_tuple=False).squeeze(1)
                    except Exception:
                        keep_idx = batched_nms(b, s, torch.full_like(s, c), nms_thr)
                else:
                    keep_idx = batched_nms(b, s, torch.full_like(s, c), nms_thr)
                if keep_idx.numel() == 0:
                    continue
                b = b[keep_idx]
                s = s[keep_idx]
                e = e[keep_idx]
                boxes_all.append(b)
                scores_all.append(s)
                energies_all.append(e)
                labels_all.append(torch.full((keep_idx.numel(),), c, dtype=torch.int64, device=b.device))
            if boxes_all:
                b_cat = torch.cat(boxes_all, 0)
                s_cat = torch.cat(scores_all, 0)
                e_cat = torch.cat(energies_all, 0)
                l_cat = torch.cat(labels_all, 0)
                # 级联过滤（增强版）
                if cascade_enable and (cascade_model is not None) and b_cat.numel() > 0:
                    try:
                        img_b = img_tensors[img_idx:img_idx+1]
                        idxs = torch.full((b_cat.size(0),1), 0, device=b_cat.device, dtype=b_cat.dtype)
                        rois = torch.cat([idxs, b_cat], dim=1)
                        crops = roi_align(img_b, [rois], output_size=(cascade_input_size, cascade_input_size), spatial_scale=1.0, sampling_ratio=-1, aligned=True)
                        with torch.no_grad():
                            try:
                                logits_c = cascade_model(crops)
                            except Exception:
                                feat = cascade_model.features(crops) if hasattr(cascade_model, 'features') else cascade_model.forward_features(crops)
                                pooled = feat.mean([-2, -1]) if feat.ndim == 4 else feat
                                if hasattr(cascade_model, 'classifier') and isinstance(cascade_model.classifier, nn.Sequential):
                                    logits_c = cascade_model.classifier(pooled)
                                elif hasattr(cascade_model, 'fc') and isinstance(cascade_model.fc, nn.Linear):
                                    logits_c = cascade_model.fc(pooled)
                                else:
                                    logits_c = pooled
                        if cascade_use_logitnorm:
                            logits_c = logits_c / (logits_c.norm(dim=1, keepdim=True) + 1.0e-6)
                        logits_c = logits_c / max(float(cascade_temp), 1.0e-6)
                        # 能量与概率
                        energy_c = -torch.logsumexp(logits_c, dim=1)
                        prob_c = torch.softmax(logits_c, dim=1).max(dim=1).values
                        # 自适应 γ2
                        if cascade_adapt_gamma2 and b_cat.size(0) > 1:
                            # 归一化面积与一阶段分数
                            wh = (b_cat[:, 2] - b_cat[:, 0]).clamp(min=1) * (b_cat[:, 3] - b_cat[:, 1]).clamp(min=1)
                            area = wh.sqrt()
                            area_n = (area - area.min()) / (area.max() - area.min() + 1.0e-6)
                            s_n = (s_cat - s_cat.min()) / (s_cat.max() - s_cat.min() + 1.0e-6)
                            # 上下文密度（IoU>阈值的邻居占比）
                            ious_all = box_iou(b_cat, b_cat)
                            neigh = (ious_all > max(0.3, float(cascade_decay_iou))) & (~torch.eye(b_cat.size(0), dtype=torch.bool, device=b_cat.device))
                            ctx = neigh.sum(dim=1).float()
                            ctx_n = (ctx - ctx.min()) / (ctx.max() - ctx.min() + 1.0e-6)
                            w_a = float(cascade_gamma2_w_area); w_s = float(cascade_gamma2_w_score); w_c = float(cascade_gamma2_w_context)
                            w_sum = max(w_a + w_s + w_c, 1.0e-6)
                            comb = (w_a * (1.0 - area_n) + w_s * (1.0 - s_n) + w_c * ctx_n) / w_sum
                            comb = comb.clamp(0.0, 1.0)
                            gamma2_vec = float(cascade_gamma2_min) + (float(cascade_gamma2_max) - float(cascade_gamma2_min)) * comb
                        else:
                            gamma2_vec = torch.full_like(energy_c, float(cascade_gamma2))
                        # 双重重打分：融合 prob 与 能量边距
                        if cascade_dual_rescore:
                            alpha = float(cascade_dual_alpha)
                            k2 = float(cascade_sig_k2)
                            s_c = alpha * prob_c + (1.0 - alpha) * torch.sigmoid(k2 * (gamma2_vec - energy_c))
                            s_cat = s_cat * s_c
                            s_cat = s_cat.clamp(0.0, 1.0)
                        # 级联判别
                        keep_c = (energy_c < gamma2_vec)
                        if keep_c.any():
                            keep_idx2 = torch.nonzero(keep_c, as_tuple=False).squeeze(1)
                            b_cat = b_cat[keep_idx2]
                            s_cat = s_cat[keep_idx2]
                            e_cat = e_cat[keep_idx2]
                            l_cat = l_cat[keep_idx2]
                        else:
                            # 全部丢弃
                            b_cat = b_cat[:0]; s_cat = s_cat[:0]; e_cat = e_cat[:0]; l_cat = l_cat[:0]
                        # 稳健化衰减：对高 IoU 邻居按最大 IoU 指数衰减（同类优先）
                        if cascade_decay_enable and b_cat.numel() > 0:
                            ious_kept = box_iou(b_cat, b_cat)
                            if ious_kept.numel() > 0:
                                same = (l_cat.unsqueeze(1) == l_cat.unsqueeze(0))
                                mask = (ious_kept > float(cascade_decay_iou)) & (~torch.eye(b_cat.size(0), dtype=torch.bool, device=b_cat.device)) & same
                                # 每个框的最大邻居 IoU
                                max_iou, _ = torch.where(mask, ious_kept, torch.zeros_like(ious_kept)).max(dim=1)
                                decay = torch.exp(-float(cascade_decay_alpha) * max_iou)
                                s_cat = s_cat * decay
                    except Exception:
                        pass
                # 截断到 detections_per_img
                if b_cat.size(0) > detections_per_img:
                    ord_idx = torch.argsort(s_cat, descending=True)[:detections_per_img]
                    b_cat = b_cat[ord_idx]; s_cat = s_cat[ord_idx]; e_cat = e_cat[ord_idx]; l_cat = l_cat[ord_idx]
                res = {'boxes': b_cat, 'scores': s_cat, 'labels': l_cat, 'energy': e_cat}
            else:
                # ...existing empty result...
                res = {'boxes': torch.empty((0, 4), device=img_tensors.device), 'scores': torch.empty((0,), device=img_tensors.device), 'labels': torch.empty((0,), dtype=torch.int64, device=img_tensors.device), 'energy': torch.empty((0,), device=img_tensors.device)}
            results.append(res)
        return results

    TP = FP = FN = 0
    use_periodic_log = (log_interval is not None and log_interval > 0)
    use_silent = (log_interval is not None and log_interval < 0)
    total_steps = len(data_loader)
    with torch.no_grad():
        if use_periodic_log or tqdm is None or use_silent:
            it = data_loader
            steps = 0
            for images, targets in it:
                steps += 1
                images, targets = to_device(list(images), list(targets), device)
                sanitize_targets(images, targets)
                if not eval_tta_hflip:
                    outputs = _detect_with_energy(images) if energy_open_set else model(images)
                    outs = list(zip(outputs, targets))
                else:
                    outputs = _detect_with_energy(images) if energy_open_set else model(images)
                    images_flipped = [torch.flip(im, dims=[2]) for im in images]
                    outputs_flip = _detect_with_energy(images_flipped) if energy_open_set else model(images_flipped)
                    outs = []
                    for out, out_f, im, tgt in zip(outputs, outputs_flip, images, targets):
                        _, H, W = im.shape
                        boxes = out['boxes']; scores = out['scores']; labels = out['labels']
                        energy = out.get('energy', None)
                        b2 = out_f['boxes']
                        if b2.numel() > 0:
                            x1 = W - 1 - b2[:, 2]; x2 = W - 1 - b2[:, 0]
                            b2 = torch.stack([x1, b2[:, 1], x2, b2[:, 3]], dim=1).clamp(0, max(W - 1, 1))
                        boxes_cat, scores_cat, labels_cat = boxes, scores, labels
                        energy_cat = energy
                        if b2.numel() > 0:
                            boxes_cat = torch.cat([boxes_cat, b2], 0)
                            scores_cat = torch.cat([scores_cat, out_f['scores']], 0)
                            labels_cat = torch.cat([labels_cat, out_f['labels']], 0)
                            if energy is not None and ('energy' in out_f):
                                energy_cat = torch.cat([energy_cat, out_f['energy']], 0) if energy_cat is not None else out_f['energy']
                        nms_thr2 = float(eval_nms if (eval_nms is not None and eval_nms > 0) else 0.5)
                        keep_idx = batched_nms(boxes_cat, scores_cat, labels_cat, nms_thr2)
                        if isinstance(eval_max_dets, int) and eval_max_dets > 0:
                            order = torch.argsort(scores_cat[keep_idx], descending=True)
                            keep_idx = keep_idx[order][:eval_max_dets]
                        merged = {'boxes': boxes_cat[keep_idx], 'scores': scores_cat[keep_idx], 'labels': labels_cat[keep_idx]}
                        if energy_cat is not None:
                            merged['energy'] = energy_cat[keep_idx]
                        outs.append((merged, tgt))
                for out, tgt in outs:
                    scores = out['scores']
                    keep = scores >= score_thr
                    if energy_open_set and ('energy' in out) and (out['energy'] is not None) and out['energy'].numel() == scores.numel():
                        keep = keep & (out['energy'] < gamma)
                    boxes_p = out['boxes'][keep]
                    labels_p = out['labels'][keep]
                    boxes_g = tgt['boxes']
                    labels_g = tgt['labels']
                    # 调试统计：每张图更新
                    if dbg_enabled:
                        try:
                            dbg_seen += 1
                            dbg_tot_pred += int(scores.numel())
                            dbg_tot_kept += int(keep.sum().item())
                            dbg_tot_gt += int(boxes_g.size(0))
                            if boxes_p.numel() > 0 and boxes_g.numel() > 0:
                                iou_mat = box_iou(boxes_p, boxes_g)
                                if iou_mat.numel() > 0:
                                    max_iou, _ = iou_mat.max(dim=1)
                                    dbg_iou_sum += float(max_iou.mean().item())
                        except Exception:
                            pass
                    if boxes_p.numel() > 0:
                        order = torch.argsort(scores[keep], descending=True)
                        boxes_p = boxes_p[order]
                        labels_p = labels_p[order]
                    tp, fp, fn = greedy_match_iou(boxes_p, labels_p, boxes_g, labels_g, iou_thr=iou_thr)
                    TP += tp; FP += fp; FN += fn
                # 调试周期打印
                if dbg_enabled and dbg_interval > 0 and (steps % dbg_interval == 0 or steps == total_steps):
                    try:
                        avg_iou_step = (dbg_iou_sum / max(1, dbg_tot_kept)) if dbg_tot_kept > 0 else 0.0
                        print(f"[Eval-Debug] seen {dbg_seen} | preds {dbg_tot_pred} | kept {dbg_tot_kept} | gt {dbg_tot_gt} | avgIoU {avg_iou_step:.3f}")
                    except Exception:
                        pass
        else:
            it = tqdm(data_loader, desc="Eval", ncols=100, ascii=True)
            step_tq = 0
            for images, targets in it:
                step_tq += 1
                images, targets = to_device(list(images), list(targets), device)
                sanitize_targets(images, targets)
                if not eval_tta_hflip:
                    outputs = _detect_with_energy(images) if energy_open_set else model(images)
                    outs = list(zip(outputs, targets))
                else:
                    outputs = _detect_with_energy(images) if energy_open_set else model(images)
                    images_flipped = [torch.flip(im, dims=[2]) for im in images]
                    outputs_flip = _detect_with_energy(images_flipped) if energy_open_set else model(images_flipped)
                    outs = []
                    for out, out_f, im, tgt in zip(outputs, outputs_flip, images, targets):
                        _, H, W = im.shape
                        boxes = out['boxes']; scores = out['scores']; labels = out['labels']
                        energy = out.get('energy', None)
                        b2 = out_f['boxes']
                        if b2.numel() > 0:
                            x1 = W - 1 - b2[:, 2]; x2 = W - 1 - b2[:, 0]
                            b2 = torch.stack([x1, b2[:, 1], x2, b2[:, 3]], dim=1).clamp(0, max(W - 1, 1))
                        boxes_cat, scores_cat, labels_cat = boxes, scores, labels
                        energy_cat = energy
                        if b2.numel() > 0:
                            boxes_cat = torch.cat([boxes_cat, b2], 0)
                            scores_cat = torch.cat([scores_cat, out_f['scores']], 0)
                            labels_cat = torch.cat([labels_cat, out_f['labels']], 0)
                            if energy is not None and ('energy' in out_f):
                                energy_cat = torch.cat([energy_cat, out_f['energy']], 0) if energy_cat is not None else out_f['energy']
                        nms_thr2 = float(eval_nms if (eval_nms is not None and eval_nms > 0) else 0.5)
                        keep_idx = batched_nms(boxes_cat, scores_cat, labels_cat, nms_thr2)
                        if isinstance(eval_max_dets, int) and eval_max_dets > 0:
                            order = torch.argsort(scores_cat[keep_idx], descending=True)
                            keep_idx = keep_idx[order][:eval_max_dets]
                        merged = {'boxes': boxes_cat[keep_idx], 'scores': scores_cat[keep_idx], 'labels': labels_cat[keep_idx]}
                        if energy_cat is not None:
                            merged['energy'] = energy_cat[keep_idx]
                        outs.append((merged, tgt))
                for out, tgt in outs:
                    scores = out['scores']
                    keep = scores >= score_thr
                    if energy_open_set and ('energy' in out) and (out['energy'] is not None) and out['energy'].numel() == scores.numel():
                        keep = keep & (out['energy'] < gamma)
                    boxes_p = out['boxes'][keep]
                    labels_p = out['labels'][keep]
                    boxes_g = tgt['boxes'
                    labels_g = tgt['labels']
                    # 调试统计：每张图更新
                    if dbg_enabled:
                        try:
                            dbg_seen += 1
                            dbg_tot_pred += int(scores.numel())
                            dbg_tot_kept += int(keep.sum().item())
                            dbg_tot_gt += int(boxes_g.size(0))
                            if boxes_p.numel() > 0 and boxes_g.numel() > 0:
                                iou_mat = box_iou(boxes_p, boxes_g)
                                if iou_mat.numel() > 0:
                                    max_iou, _ = iou_mat.max(dim=1)
                                    dbg_iou_sum += float(max_iou.mean().item())
                        except Exception:
                            pass
                    if boxes_p.numel() > 0:
                        order = torch.argsort(scores[keep], descending=True)
                        boxes_p = boxes_p[order]
                        labels_p = labels_p[order]
                    tp, fp, fn = greedy_match_iou(boxes_p, labels_p, boxes_g, labels_g, iou_thr=iou_thr)
                    TP += tp; FP += fp; FN += fn
                # 调试周期打印
                if dbg_enabled and dbg_interval > 0 and (step_tq % dbg_interval == 0):
                    try:
                        avg_iou_step = (dbg_iou_sum / max(1, dbg_tot_kept)) if dbg_tot_kept > 0 else 0.0
                        print(f"[Eval-Debug] seen {dbg_seen} | preds {dbg_tot_pred} | kept {dbg_tot_kept} | gt {dbg_tot_gt} | avgIoU {avg_iou_step:.3f}")
                    except Exception:
                        pass
            if hasattr(it, 'close'):
                it.close()
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1.0e-8)
    if dbg_enabled:
        try:
            avg_iou = (dbg_iou_sum / max(1, dbg_tot_kept)) if dbg_tot_kept > 0 else 0.0
            print(f"[Eval-Debug-Final] seen {dbg_seen} | preds {dbg_tot_pred} | kept {dbg_tot_kept} | gt {dbg_tot_gt} | avgIoU {avg_iou:.3f}")
        except Exception:
            pass
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
    """解析形如 "0.3:0.8:0.05" 或 "0.3,0.4,0.5" 的阈值扫参数，返回有效阈值列表(0~1)."""
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


def build_cascade_classifier(model_name: str, ckpt: str, device: torch.device) -> Optional[nn.Module]:
    """新增：级联分类器构建（默认使用 torchvision ConvNeXt 并从本地 pth 加载）"""
    try:
        name = (model_name or 'convnext_large').lower()
        if name in ['convnext_large', 'convnext-l', 'cnx_l', 'cnx-large']:
            net = torchvision.models.convnext_large(weights=None)
        elif name in ['convnext_base', 'convnext-b', 'cnx_b', 'cnx-base']:
            net = torchvision.models.convnext_base(weights=None)
        elif name in ['convnext_tiny', 'convnext-t', 'cnx_t', 'cnx-tiny']:
            net = torchvision.models.convnext_tiny(weights=None)
        elif hasattr(torchvision.models, name):
            net = getattr(torchvision.models, name)(weights=None)
        else:
            net = torchvision.models.convnext_large(weights=None)
        if ckpt and os.path.isfile(ckpt):
            sd = torch.load(ckpt, map_location='cpu')
            if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
                sd = sd['model']
            net.load_state_dict(sd, strict=False)
        net.eval().to(device)
        for p in net.parameters():
            p.requires_grad = False
        return net
    except Exception:
        return None


# 新增：安全的 state_dict 加载（忽略形状不匹配的条目）
def load_state_dict_forgiving(model: nn.Module, state: dict):
    msd = model.state_dict()
    ok = {}
    skipped = []
    for k, v in state.items():
        if k in msd and hasattr(v, 'shape') and hasattr(msd[k], 'shape') and tuple(v.shape) == tuple(msd[k].shape):
            ok[k] = v
        else:
            skipped.append(k)
    missing = [k for k in msd.keys() if k not in ok]
    model.load_state_dict(ok, strict=False)
    try:
        print(f"[Load] loaded={len(ok)} skipped={len(skipped)} missing={len(missing)}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default=r"C:\\Users\\HP\\Desktop\\Exp\\LabelMePNG")
    parser.add_argument('--annotations', type=str, default=r"C:\\Users\\HP\\Desktop\\Exp\\LabelMePNG\\coco_labeled.json")
    parser.add_argument('--train_json', type=str, default='', help='If set, use this as fixed train COCO json')
    parser.add_argument('--val_json', type=str, default='', help='If set, use this as fixed val COCO json')
    parser.add_argument('--epochs', type=int, default=200)
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
    parser.add_argument('--twostream', type=str2bool, nargs='?', const=True, default=True,
                        help='use two-stream+PANet+Swin detection backbone (paper version), accept true/false')
    parser.add_argument('--no-twostream', dest='twostream', action='store_false', help='disable two-stream backbone')
    # 本机默认预训练权重路径（可直接覆盖）
    parser.add_argument('--pos_weights', type=str, default=r"C:\\Users\\HP\\Desktop\\Exp\\Model\\Pretrain\\Backbone\\Position\\mobilenet_v3_large-8738ca79.pth", help='local path to MobileNetV3-Large weights (.pth)')
    parser.add_argument('--swin_weights', type=str, default=r"C:\\Users\\HP\\Desktop\\Exp\\Model\\Pretrain\\Swin\\pth\\swin_small_patch4_window7_224.pth", help='local path to Swin-Small weights (.pth)')
    parser.add_argument('--texture_yolov8m_cls', type=str, default=r"C:\\Users\\HP\\Desktop\\Exp\\Model\\Pretrain\\Backbone\\Texture\\yolov8m-cls.pt", help='local path to yolov8m-cls.pt for initializing texture stream')
    parser.add_argument('--yolo_det', type=str, default=r"C:\\Users\\HP\\Desktop\\Exp\\Model\\Pretrain\\Detector\\YOLOv8\\yolov8m.pt", help='path to YOLOv8 yolov8m.pt weights')
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
    parser.add_argument('--ema_update_every', type=int, default=1, help='update EMA every N steps (default 1)')
    # 新增：仅当EMA更新次数达到该阈值后，才使用EMA进行评估/保存
    parser.add_argument('--ema_eval_min_updates', type=int, default=0, help='min EMA updates before using EMA for eval/save')
    parser.add_argument('--clip_grad', type=float, default=0.0, help='max grad norm (0 to disable)')
    # 新增：评估期收紧候选/阈值以提升精度
    parser.add_argument('--eval_max_dets', type=int, default=-1, help='eval-only: max detections per image (override)')
    parser.add_argument('--eval_nms', type=float, default=-1.0, help='eval-only: NMS threshold for ROI heads (override)')
    parser.add_argument('--eval_rpn_post_nms_topn', type=int, default=-1, help='eval-only: RPN post-NMS top-N (testing) override')
    parser.add_argument('--eval_tta_hflip', action='store_true', help='eval-only: enable horizontal flip TTA merge')
    # 新增：可禁用前期宽松阈值策略
    parser.add_argument('--no_early_relax', action='store_true', help='disable early relaxed thresholds for eval')
    # 开放集能量阈值（评估时基于ROI分类logits计算 energy=-logsumexp(logits/T) 并做过滤）
    parser.add_argument('--energy_open_set', action='store_true', help='enable ROI energy thresholding during evaluation')
    # 新增：评估期可关闭级联
    parser.add_argument('--eval_disable_cascade', action='store_true', help='eval-only: disable cascade classifier stage')
    parser.add_argument('--energy_gamma', type=float, default=8.0, help='energy threshold gamma; energy >= energy_gamma will be rejected')
    parser.add_argument('--energy_temp', type=float, default=1.0, help='temperature for energy computation')
    parser.add_argument('--energy_exclude_bg', action='store_true', help='compute energy on foreground classes only (exclude background)')
    # P0：不确定性感知打分/后处理
    parser.add_argument('--conf_temp', type=float, default=1.0, help='temperature scaling for classification logits when computing scores (eval)')
    parser.add_argument('--energy_rescore', action='store_true', help='re-score detections using energy during evaluation')
    parser.add_argument('--rescore_mode', type=str, default='sigmoid', choices=['sigmoid','linear'], help='energy re-scoring mode')
    parser.add_argument('--rescore_k', type=float, default=1.0, help='sigmoid mode: k in s * sigmoid(k*(gamma-energy))')
    parser.add_argument('--rescore_lambda', type=float, default=0.0, help='linear mode: s - lambda*energy')
    parser.add_argument('--soft_nms', action='store_true', help='use gaussian soft-NMS per class in eval (inside ROI filtering)')
    parser.add_argument('--soft_nms_sigma', type=float, default=0.5, help='sigma for gaussian soft-NMS')
    parser.add_argument('--soft_nms_iou', type=float, default=0.5, help='IoU threshold for soft-NMS')
    # 开放集未知集（检测）
    parser.add_argument('--unknown_val_json', type=str, default='', help='COCO json for unknown (open-set) validation')
    parser.add_argument('--unknown_images_dir', type=str, default='', help='images dir for unknown set, default to images_dir if empty')
    # P1：Outlier Exposure/干扰混入（能量正则）
    parser.add_argument('--oe_enable', action='store_true', help='enable OE energy regularization during training')
    parser.add_argument('--oe_weight', type=float, default=0.0, help='loss weight for OE energy regularization (0 disables)')
    parser.add_argument('--oe_gamma', type=float, default=8.0, help='target minimum energy for unknown proposals')
    parser.add_argument('--oe_num_props', type=int, default=64, help='max number of low-IoU proposals per image for OE')
    parser.add_argument('--oe_temp', type=float, default=1.0, help='temperature for energy in OE')
    parser.add_argument('--oe_exclude_bg', action='store_true', help='compute energy on foreground classes only in OE')
    parser.add_argument('--oe_start_epoch', type=int, default=5, help='start epoch to enable OE (disable OE before this epoch)')
    # 新增：ROI训练期参数（求精导向）
    parser.add_argument('--roi_fg_iou', type=float, default=0.6, help='ROI positive IoU threshold (train)')
    parser.add_argument('--roi_bg_iou', type=float, default=0.4, help='ROI negative IoU threshold (train)')
    parser.add_argument('--roi_pos_frac', type=float, default=0.20, help='ROI positive fraction in batch (train)')
    parser.add_argument('--roi_batch_size', type=int, default=512, help='ROI samples per image (train)')
    # 新增：ConvNeXt教师-特征级蒸馏参数
    parser.add_argument('--kd_feat', action='store_true', help='enable feature-level KD with ConvNeXt teacher')
    parser.add_argument('--kd_alpha', type=float, default=0.7, help='feature KD loss weight')
    parser.add_argument('--kd_feat_loss', type=str, default='cosine', choices=['cosine','l2'], help='feature KD loss type')
    parser.add_argument('--kd_max_rois', type=int, default=128, help='max positive ROIs per image for KD')
    parser.add_argument('--kd_teacher', type=str, default='convnext_large', help='teacher model name for KD (convnext_*)')
    parser.add_argument('--kd_teacher_ckpt', type=str, default=r"C:\\Users\\HP\\Desktop\\Exp\\Model\\Pretrain\\Basic\\ConvNeXt\\convnext_large.pth", help='teacher checkpoint path')
    parser.add_argument('--kd_warmup_epochs', type=int, default=3, help='linear warmup epochs for KD alpha')
    # 新增：评估专用模式与阈值扫描
    parser.add_argument('--eval_only', action='store_true', help='only run evaluation, no training')
    parser.add_argument('--eval_ckpt', type=str, default='', help='path to checkpoint to evaluate (default: best.pt in out_dir)')
    parser.add_argument('--thr_sweep', type=str, default='', help='score threshold sweep, e.g. "0.3:0.8:0.05" or "0.3,0.4,0.5"')
    # 新增：评估时优先加载 raw（若存在）
    parser.add_argument('--eval_load_raw', action='store_true', help='eval-only: prefer loading raw weights if available')
    # 新增：日志输出频率控制
    parser.add_argument('--log_interval', type=int, default=0, help='if >0, disable tqdm and print one log every N batches')
    # 新增：评估日志与调试参数（防止缺失导致 AttributeError）
    parser.add_argument('--eval_log_interval', type=int, default=0, help='eval logging: >0 periodic print, 0 tqdm, <0 silent')
    parser.add_argument('--debug_eval', action='store_true', help='print eval debug stats (pred/kept/avg IoU)')
    parser.add_argument('--debug_eval_interval', type=int, default=0, help='periodic eval debug every N steps (0 disables)')
    # 新增：续训开关
    parser.add_argument('--resume', action='store_true', help='resume training from checkpoint in out_dir')
    parser.add_argument('--resume_ckpt', type=str, default='', help='optional checkpoint path to resume from (overrides out_dir)')
    # 新增：KD与早停/调度控制
    parser.add_argument('--kd_stop_epoch', type=int, default=-1, help='stop applying KD after this epoch (1-based); <=0 disables')
    parser.add_argument('--early_stop', action='store_true', help='enable early stopping on F1')
    parser.add_argument('--early_stop_patience', type=int, default=15, help='early stopping patience (epochs)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.1, help='minimum F1 improvement (abs) to reset patience')
    parser.add_argument('--early_stop_warmup', type=int, default=15, help='do not early-stop before this epoch')
    parser.add_argument('--cosine_restarts', action='store_true', help='use CosineAnnealingWarmRestarts instead of plain cosine')
    parser.add_argument('--cosine_T0', type=int, default=10, help='T_0 for CosineAnnealingWarmRestarts')
    parser.add_argument('--cosine_Tmult', type=int, default=2, help='T_mult for CosineAnnealingWarmRestarts')
    # 解析参数（必须在首次使用 args 之前）
    args = parser.parse_args()
    # 兼容：若旧参数集中未包含 eval_log_interval，这里补一个默认值
    if not hasattr(args, 'eval_log_interval'):
        args.eval_log_interval = 0
    # 统一输出目录到项目根目录：Exp/Model/Output/50标签训练X（X 自增）
    if not args.out_dir or args.out_dir.strip() == '':
        project_root = Path(FILE_DIR).parent.parent  # .../TSFNet/src -> .../Exp
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

    # 固定 YOLOv8 权重路径为 yolov8m.pt（可传参覆盖）
    # 不再进行目录内自动挑选，确保一致性
    args.yolo_det = r"C:\\Users\\HP\\Desktop\\Exp\\Model\\Pretrain\\Detector\\YOLOv8\\yolov8m.pt"

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
        # 注意：为避免类别ID映射不一致，验证集强制引用训练集的categories
        ds_train_full = CocoRFDataset(images_dir=args.images_dir, annotations_json=args.annotations, augment=True)
        ds_val_full = CocoRFDataset(images_dir=args.images_dir, annotations_json=args.annotations, augment=False,
                                    categories_ref_json=args.annotations)
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

    # 构建KD教师与投影头占位
    kd_teacher_model = None
    kd_proj = None  # nn.Linear(student_dim -> teacher_dim)
    if args.kd_feat:
        kd_teacher_model = build_cascade_classifier(
            model_name=args.kd_teacher,
            ckpt=args.kd_teacher_ckpt,
            device=device
        )
        if kd_teacher_model is None:
            try:
                print('Warn: KD teacher build failed, disable kd_feat.')
            except Exception:
                pass
            args.kd_feat = False

    # eval-only：仅评估指定权重，支持阈值扫描与TTA
    if args.eval_only:
        ckpt_path = args.eval_ckpt if args.eval_ckpt else os.path.join(args.out_dir, 'best.pt')
        # 新增：找不到则回退 last.pt，或在输出根目录下搜索最近的 best/last.pt
        if not os.path.isfile(ckpt_path):
            alt = os.path.join(args.out_dir, 'last.pt')
            if os.path.isfile(alt):
                ckpt_path = alt
            else:
                try:
                    project_root = Path(FILE_DIR).parent.parent
                    base_out = project_root / 'Model' / 'Output'
                    cand = []
                    if base_out.exists():
                        for d in base_out.iterdir():
                            if d.is_dir():
                                p1 = d / 'best.pt'
                                p2 = d / 'last.pt'
                                if p1.exists():
                                    cand.append((p1.stat().st_mtime, str(p1)))
                                elif p2.exists():
                                    cand.append((p2.stat().st_mtime, str(p2)))
                    if cand:
                        cand.sort(reverse=True)
                        ckpt_path = cand[0][1]
                        print(f"[Eval-Only] Auto-picked ckpt: {ckpt_path}")
                    else:
                        print("[Eval-Only] No checkpoint found. Please provide --eval_ckpt explicitly.")
                        return
                except Exception:
                    print("[Eval-Only] No checkpoint found and search failed. Please provide --eval_ckpt explicitly.")
                    return
        sd = torch.load(ckpt_path, map_location=device)
        # 兼容新旧ckpt：优先按 --eval_load_raw 加载 raw，其次加载 model，再次加载 model_ema；最后回退整体
        state_to_load = None
        if isinstance(sd, dict):
            if args.eval_load_raw and ('model_raw' in sd):
                state_to_load = sd['model_raw']
            elif 'model' in sd:
                state_to_load = sd['model']
            elif 'model_ema' in sd:
                state_to_load = sd['model_ema']
        if state_to_load is None:
            state_to_load = sd
        # 使用宽松加载以跳过 RPN/ROI 形状不匹配
        load_state_dict_forgiving(model, state_to_load)
        # 构建级联模型（默认启用，使用本地权重）；可通过 --eval_disable_cascade 关闭
        cascade_model = None
        if not args.eval_disable_cascade:
            cascade_model = build_cascade_classifier(
                model_name='convnext_large',
                ckpt=r"C:\\Users\\HP\\Desktop\\Exp\\Model\\Pretrain\\Basic\\ConvNeXt\\convnext_large.pth",
                device=device
            )
        thr_list = _parse_thr_sweep(args.thr_sweep) if args.thr_sweep else [args.score_thr]
        best = None
        for thr in thr_list:
            stats = evaluate_simple(model, val_loader, device, score_thr=thr, iou_thr=args.iou_thr,
                                    log_interval=args.eval_log_interval,
                                    eval_max_dets=args.eval_max_dets if args.eval_max_dets > 0 else None,
                                    eval_nms=args.eval_nms if args.eval_nms > 0 else None,
                                    eval_rpn_post_nms_top_n=args.eval_rpn_post_nms_topn if args.eval_rpn_post_nms_topn > 0 else None,
                                    eval_tta_hflip=args.eval_tta_hflip,
                                    energy_open_set=args.energy_open_set, gamma=args.energy_gamma, energy_temp=args.energy_temp, energy_exclude_bg=args.energy_exclude_bg,
                                    conf_temp=args.conf_temp, energy_rescore=args.energy_rescore, rescore_mode=args.rescore_mode, rescore_k=args.rescore_k, rescore_lambda=args.rescore_lambda,
                                    soft_nms=args.soft_nms, soft_nms_sigma=args.soft_nms_sigma, soft_nms_iou=args.soft_nms_iou,
                                    cascade_enable=(not args.eval_disable_cascade), cascade_model=cascade_model,
                                    cascade_model_name='convnext_large', cascade_ckpt_path=r"C:\\Users\\HP\\Desktop\\Exp\\Model\\Pretrain\\Basic\\ConvNeXt\\convnext_large.pth",
                                    cascade_input_size=224, cascade_use_logitnorm=True, cascade_temp=1.0, cascade_gamma2=5.0)
            print(f"Eval thr={thr:.3f} | P {stats['precision@0.5']:.2f} R {stats['recall@0.5']:.2f} F1 {stats['f1@0.5']:.2f}")
            if (best is None) or (stats['f1@0.5'] > best['f1@0.5']):
                best = {**stats, 'thr': thr}
        if best is not None:
            print(f"Best@thr={best['thr']:.3f} -> P {best['precision@0.5']:.2f} R {best['recall@0.5']:.2f} F1 {best['f1@0.5']:.2f}")
        return

    # 在创建优化器前：若要求冻结前 N 个 epoch，则预冻结骨干，只保留 RPN/ROIHeads 训练
    if args.twostream and args.freeze_backbone > 0:
        for n, p in model.named_parameters():
            if n.startswith('rpn.') or n.startswith('roi_heads.'):
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            print(f"[Init] Freeze backbone for first {args.freeze_backbone} epochs (train heads only)")
        except Exception:
            pass

    # 参数组：两流骨干分组 LR；其它维持基准 LR（通过模块成员识别 backbone）
    if args.twostream:
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'parameters'):
            _backbone_ids = {id(p) for p in model.backbone.parameters()}
        else:
            _backbone_ids = set()
        backbone_params = [p for p in model.parameters() if id(p) in _backbone_ids]
        head_params = [p for p in model.parameters() if id(p) not in _backbone_ids]
        param_groups = [
            {'params': [p for p in backbone_params if p.requires_grad], 'lr': args.lr * args.backbone_lr_mult},
            {'params': [p for p in head_params if p.requires_grad], 'lr': args.lr},
        ]
    else:
        param_groups = [p for p in model.parameters() if p.requires_grad]
    # 统一创建优化器，避免未定义
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # 调度器：Warmup + 主调度器
    if args.cosine:
        if getattr(args, 'cosine_restarts', False):
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(1, int(getattr(args, 'cosine_T0', 10))),
                T_mult=max(1, int(getattr(args, 'cosine_Tmult', 2)))
            )
        else:
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

    # 新增：续训逻辑（从 last.pt 优先，否则 best.pt；也可 --resume_ckpt 指定）
    start_epoch = 0
    if getattr(args, 'resume', False):
        # 选择恢复文件
        ckpt_path = args.resume_ckpt.strip() if getattr(args, 'resume_ckpt', '') else ''
        if not ckpt_path:
            p_last = os.path.join(args.out_dir, 'last.pt')
            p_best = os.path.join(args.out_dir, 'best.pt')
            ckpt_path = p_last if os.path.isfile(p_last) else p_best
        if ckpt_path and os.path.isfile(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                # 恢复模型原始权重为训练用
                state_to_load = None
                if isinstance(ckpt, dict):
                    if 'model_raw' in ckpt:
                        state_to_load = ckpt['model_raw']
                    elif 'model' in ckpt:
                        state_to_load = ckpt['model']
                if state_to_load is None:
                    state_to_load = ckpt
                load_state_dict_forgiving(model, state_to_load)
                # 恢复 EMA（若启用且存在）
                try:
                    if ema is not None and isinstance(ckpt, dict) and ckpt.get('model_ema'):
                        load_state_dict_forgiving(ema.ema, ckpt['model_ema'])
                except Exception:
                    pass
                # 恢复优化器（参数组不匹配时忽略）
                try:
                    if isinstance(ckpt, dict) and 'optimizer' in ckpt:
                        optimizer.load_state_dict(ckpt['optimizer'])
                except Exception:
                    pass
                # 设置起始 epoch 与最佳 F1（若存在）
                if isinstance(ckpt, dict):
                    try:
                        start_epoch = int(ckpt.get('epoch', -1)) + 1
                    except Exception:
                        start_epoch = 0
                    try:
                        if isinstance(ckpt.get('stats', None), dict):
                            best_f1 = float(ckpt['stats'].get('f1@0.5', best_f1))
                    except Exception:
                        pass
                try:
                    print(f"[Resume] Loaded checkpoint: {ckpt_path} | start_epoch={start_epoch}")
                except Exception:
                    pass
            except Exception as e:
                try:
                    print(f"[Resume] Failed to load checkpoint {ckpt_path}: {e}")
                except Exception:
                    pass
        else:
            try:
                print(f"[Resume] Checkpoint not found: {ckpt_path}")
            except Exception:
                pass

    # 移除 results_csv，改为在 log.csv 中记录全部常用指标
    # if (not args.results_csv) or args.results_csv.strip() == '':
    #     args.results_csv = os.path.join(args.out_dir, 'results_det.csv')
    # try:
    #     if not os.path.exists(args.results_csv):
    #         with open(args.results_csv, 'w', encoding='utf-8') as f:
    #             f.write('epoch,lr,loss,precision@0.5,recall@0.5,f1@0.5,TP,FP,FN\n')
    # except Exception:
    #     pass

    # 冻结/解冻逻辑（两流骨干）
    def set_backbone_trainable(flag: bool):
        # flag=True: 全开训练；flag=False: 仅训练 RPN/ROIHeads（冻结其余，包括骨干）
        if flag:
            for p in model.parameters():
                p.requires_grad = True
        else:
            for n, p in model.named_parameters():
                if n.startswith('rpn.') or n.startswith('roi_heads.'):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        try:
            n_total = sum(p.numel() for p in model.parameters())
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[Freeze] trainable params: {n_train}/{n_total} ({100.0 * n_train / max(1, n_total):.2f}%) | flag={flag}")
        except Exception:
            pass

    if args.twostream and args.freeze_backbone > 0:
        set_backbone_trainable(False)

    for epoch in range(start_epoch, args.epochs):
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
            # 关键：训练期也做目标清洗，统一 dtype/shape/范围
            sanitize_targets(images, targets)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            if args.amp:
                with torch.autocast(
                    device_type=("cuda" if torch.cuda.is_available() else "cpu"),
                    enabled=True,
                    dtype=(torch.bfloat16 if (torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()) else torch.float16)
                ):
                    losses_dict = model(images, targets)
                    if isinstance(losses_dict, list):
                        model.train(); losses_dict = model(images, targets)
                    loss = sum(v for v in losses_dict.values())
                    # === Feature-level KD (AMP环境下，教师前向禁用autocast) ===
                    if args.kd_feat and kd_teacher_model is not None:
                        try:
                            # 生成 proposals 并筛选正样本
                            t_out2 = model.transform(images)
                            t_inp2 = t_out2[0] if isinstance(t_out2, (tuple, list)) else t_out2
                            feats2 = model.backbone(t_inp2.tensors)
                            if isinstance(feats2, torch.Tensor):
                                feats2 = {0: feats2}
                            props_list, _ = model.rpn(t_inp2, feats2)
                            kd_losses = []
                            for im_idx, props_i in enumerate(props_list):
                                gt_boxes = targets[im_idx]['boxes'] if 'boxes' in targets[im_idx] else torch.empty((0,4), device=props_i.device)
                                if props_i.numel() == 0 or gt_boxes.numel() == 0:
                                    continue
                                ious = box_iou(props_i, gt_boxes)
                                iou_max, _ = ious.max(dim=1)
                                pos_mask = iou_max >= float(args.roi_fg_iou)
                                sel = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
                                if sel.numel() == 0:
                                    continue
                                sel = sel[:max(1, int(args.kd_max_rois))]
                                props_sel = props_i[sel]
                                # 学生ROI特征（box_head输出）
                                box_feats_s = model.roi_heads.box_roi_pool(feats2, [props_sel], [t_inp2.image_sizes[im_idx]])
                                box_feats_s = model.roi_heads.box_head(box_feats_s).float()
                                # 教师pre-logits特征（ConvNeXt forward_features + GAP）
                                # 强制使用 float32，并裁剪到图像尺寸，避免 AMP 下 ROIAlign 数值不稳
                                h_i, w_i = t_inp2.image_sizes[im_idx]
                                props_sel = props_sel.to(torch.float32)
                                props_sel[:, 0::2] = props_sel[:, 0::2].clamp_(0, float(w_i - 1))
                                props_sel[:, 1::2] = props_sel[:, 1::2].clamp_(0, float(h_i - 1))
                                img_b = t_inp2.tensors[im_idx:im_idx+1].to(dtype=torch.float32)
                                idxs = torch.zeros((props_sel.size(0), 1), device=props_sel.device, dtype=torch.float32)
                                rois = torch.cat([idxs, props_sel], dim=1)
                                with torch.no_grad(), torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), enabled=False):
                                    crops = roi_align(img_b, [rois], output_size=(224, 224), spatial_scale=1.0, sampling_ratio=-1, aligned=True)
                                    try:
                                        feat_t = kd_teacher_model.forward_features(crops)
                                    except Exception:
                                        feat_t = kd_teacher_model.features(crops) if hasattr(kd_teacher_model, 'features') else kd_teacher_model(crops)
                                    if feat_t.ndim == 4:
                                        feat_t = feat_t.mean([-2, -1])
                                    feat_t = feat_t.float()
                                # 懒初始化投影头
                                nonlocal_kd = False
                                try:
                                    nonlocal_kd = True
                                except Exception:
                                    pass
                                if kd_proj is None:
                                    in_dim = box_feats_s.shape[1]
                                    out_dim = feat_t.shape[1]
                                    kd_proj = nn.Linear(in_dim, out_dim, bias=False).to(box_feats_s.device)
                                    # 将投影头加入优化器
                                    try:
                                        optimizer.add_param_group({'params': kd_proj.parameters(), 'lr': optimizer.param_groups[-1]['lr']})
                                    except Exception:
                                        pass
                                s = kd_proj(box_feats_s)
                                t = feat_t
                                if args.kd_feat_loss == 'cosine':
                                    s_n = F.normalize(s, dim=1, eps=1e-6); t_n = F.normalize(t, dim=1, eps=1e-6)
                                    kd_l = (1.0 - F.cosine_similarity(s_n, t_n, dim=1)).mean()
                                else:
                                    s_n = F.normalize(s, dim=1, eps=1e-6); t_n = F.normalize(t, dim=1, eps=1e-6)
                                    kd_l = F.mse_loss(s_n, t_n)
                                if torch.isfinite(kd_l):
                                    kd_losses.append(kd_l)
                            if len(kd_losses) > 0:
                                kd_term = (sum(kd_losses) / len(kd_losses))
                                if int(args.kd_warmup_epochs) > 0:
                                    ramp = min(1.0, float(epoch + 1) / float(max(1, int(args.kd_warmup_epochs))))
                                else:
                                    ramp = 1.0
                                # KD 停止开关：超过 kd_stop_epoch 后不再叠加KD
                                kd_weight = 0.0
                                if (int(getattr(args, 'kd_stop_epoch', -1)) <= 0) or ((epoch + 1) <= int(getattr(args, 'kd_stop_epoch', -1))):
                                    kd_weight = float(args.kd_alpha) * ramp
                                if torch.isfinite(kd_term) and kd_weight > 0:
                                    loss = loss + kd_weight * kd_term
                        except Exception:
                            pass
                # 非有限损失保护
                if not torch.isfinite(loss):
                    try:
                        print("Warn: non-finite loss detected (AMP). Skipping batch.")
                    except Exception:
                        pass
                    optimizer.zero_grad(set_to_none=True)
                    # 直接跳过本 batch，避免将 NaN/Inf 写入 running_loss
                    continue
                else:
                    scaler.scale(loss).backward()
                    if args.clip_grad and args.clip_grad > 0:
                        try:
                            scaler.unscale_(optimizer)
                        except Exception:
                            pass
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                losses_dict = model(images, targets)
                if isinstance(losses_dict, list):
                    model.train(); losses_dict = model(images, targets)
                loss = sum(v for v in losses_dict.values())
                # === Feature-level KD（非AMP） ===
                if args.kd_feat and kd_teacher_model is not None:
                    try:
                        # 生成 proposals 并筛选正样本
                        t_out2 = model.transform(images)
                        t_inp2 = t_out2[0] if isinstance(t_out2, (tuple, list)) else t_out2
                        feats2 = model.backbone(t_inp2.tensors)
                        if isinstance(feats2, torch.Tensor):
                            feats2 = {0: feats2}
                        props_list, _ = model.rpn(t_inp2, feats2)
                        kd_losses = []
                        for im_idx, props_i in enumerate(props_list):
                            gt_boxes = targets[im_idx]['boxes'] if 'boxes' in targets[im_idx] else torch.empty((0,4), device=props_i.device)
                            if props_i.numel() == 0 or gt_boxes.numel() == 0:
                                continue
                            ious = box_iou(props_i, gt_boxes)
                            iou_max, _ = ious.max(dim=1)
                            pos_mask = iou_max >= float(args.roi_fg_iou)
                            sel = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
                            if sel.numel() == 0:
                                continue
                            sel = sel[:max(1, int(args.kd_max_rois))]
                            props_sel = props_i[sel]
                            # 学生ROI特征（box_head输出）
                            box_feats_s = model.roi_heads.box_roi_pool(feats2, [props_sel], [t_inp2.image_sizes[im_idx]])
                            box_feats_s = model.roi_heads.box_head(box_feats_s).float()
                            # 教师pre-logits特征（ConvNeXt forward_features + GAP）
                            # 强制使用 float32，并裁剪到图像尺寸，避免 ROIAlign 数值不稳
                            h_i, w_i = t_inp2.image_sizes[im_idx]
                            props_sel = props_sel.to(torch.float32)
                            props_sel[:, 0::2] = props_sel[:, 0::2].clamp_(0, float(w_i - 1))
                            props_sel[:, 1::2] = props_sel[:, 1::2].clamp_(0, float(h_i - 1))
                            img_b = t_inp2.tensors[im_idx:im_idx+1].to(dtype=torch.float32)
                            idxs = torch.zeros((props_sel.size(0), 1), device=props_sel.device, dtype=torch.float32)
                            rois = torch.cat([idxs, props_sel], dim=1)
                            with torch.no_grad():
                                crops = roi_align(img_b, [rois], output_size=(224, 224), spatial_scale=1.0, sampling_ratio=-1, aligned=True)
                                try:
                                    feat_t = kd_teacher_model.forward_features(crops)
                                except Exception:
                                    feat_t = kd_teacher_model.features(crops) if hasattr(kd_teacher_model, 'features') else kd_teacher_model(crops)
                                if feat_t.ndim == 4:
                                    feat_t = feat_t.mean([-2, -1])
                                feat_t = feat_t.float()
                            if kd_proj is None:
                                in_dim = box_feats_s.shape[1]
                                out_dim = feat_t.shape[1]
                                kd_proj = nn.Linear(in_dim, out_dim, bias=False).to(box_feats_s.device)
                                try:
                                    optimizer.add_param_group({'params': kd_proj.parameters(), 'lr': optimizer.param_groups[-1]['lr']})
                                except Exception:
                                    pass
                            s = kd_proj(box_feats_s)
                            t = feat_t
                            if args.kd_feat_loss == 'cosine':
                                s_n = F.normalize(s, dim=1, eps=1e-6); t_n = F.normalize(t, dim=1, eps=1e-6)
                                kd_l = (1.0 - F.cosine_similarity(s_n, t_n, dim=1)).mean()
                            else:
                                s_n = F.normalize(s, dim=1, eps=1e-6); t_n = F.normalize(t, dim=1, eps=1e-6)
                                kd_l = F.mse_loss(s_n, t_n)
                            if torch.isfinite(kd_l):
                                kd_losses.append(kd_l)
                        if len(kd_losses) > 0:
                            kd_term = (sum(kd_losses) / len(kd_losses))
                            if int(args.kd_warmup_epochs) > 0:
                                ramp = min(1.0, float(epoch + 1) / float(max(1, int(args.kd_warmup_epochs))))
                            else:
                                ramp = 1.0
                            kd_weight = 0.0
                            if (int(getattr(args, 'kd_stop_epoch', -1)) <= 0) or ((epoch + 1) <= int(getattr(args, 'kd_stop_epoch', -1))):
                                kd_weight = float(args.kd_alpha) * ramp
                            if torch.isfinite(kd_term) and kd_weight > 0:
                                loss = loss + kd_weight * kd_term
                    except Exception:
                        pass
                # 反传
                if not torch.isfinite(loss):
                    try:
                        print("Warn: non-finite loss detected. Skipping batch.")
                    except Exception:
                        pass
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                if args.clip_grad and args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

            if ema is not None:
                if (steps % max(1, int(args.ema_update_every))) == 0:
                    ema.update(model)
            running_loss += float(loss.detach().item()) * len(images)
            n_images += len(images)
            steps += 1
            if use_periodic_log and (args.log_interval > 0) and (steps % args.log_interval == 0 or steps == total_steps):
                print(f"Train {epoch+1}/{args.epochs}: {steps}/{total_steps} | loss {(running_loss/max(n_images,1)):.4f}")
            if args.dry_run > 0 and steps >= args.dry_run:
                break
        if tqdm is not None and hasattr(it, 'close') and not use_periodic_log:
            it.close()

        if args.dry_run > 0:
            print(f"Dry-run done. Seen batches: {steps}. Avg loss: {(running_loss / max(n_images, 1)):.4f}")
            return

        # 调度器步进
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        # 验证（优先使用 EMA）
        # 仅当EMA已至少更新过一次时，才使用EMA进行评估；否则回退到当前模型
        eval_model = model
        used_ema_for_eval = False
        if ema is not None and getattr(ema, 'updates', 0) >= int(getattr(args, 'ema_eval_min_updates', 0)):
            eval_model = ema.ema
            used_ema_for_eval = True
        # 本轮稳定性检查：关闭能量/级联评估，不构建级联模型
        cascade_model = None
        # 动态早期阈值：<10轮 0.05，<20轮 0.1，其后用用户阈值
        _thr_now = args.score_thr
        _iou_now = args.iou_thr
        if not getattr(args, 'no_early_relax', False):
            if epoch < 10:
                _thr_now = min(args.score_thr, 0.05)
                _iou_now = min(args.iou_thr, 0.3)
            elif epoch < 20:
                _thr_now = min(args.score_thr, 0.1)
                _iou_now = min(args.iou_thr, 0.4)
        # 标记本轮评估是否最终采用 raw（用于保存逻辑）
        used_raw_for_eval = (not used_ema_for_eval)
        stats = evaluate_simple(
            eval_model,
            val_loader,
            device,
            score_thr=_thr_now,
            iou_thr=_iou_now,
            log_interval=args.eval_log_interval,
            eval_max_dets=args.eval_max_dets if args.eval_max_dets > 0 else None,
            eval_nms=args.eval_nms if args.eval_nms > 0 else None,
            eval_rpn_post_nms_top_n=args.eval_rpn_post_nms_topn if args.eval_rpn_post_nms_topn > 0 else None,
            eval_tta_hflip=args.eval_tta_hflip,
            energy_open_set=False,  # 关闭能量
            conf_temp=args.conf_temp,
            energy_rescore=False,  # 关闭能量重打分
            rescore_mode=args.rescore_mode, rescore_k=args.rescore_k, rescore_lambda=args.rescore_lambda,
            soft_nms=args.soft_nms, soft_nms_sigma=args.soft_nms_sigma, soft_nms_iou=args.soft_nms_iou,
            cascade_enable=False,  # 关闭级联
            cascade_model=None,
            debug_eval=args.debug_eval,
            debug_eval_interval=args.debug_eval_interval,
        )

        # 若使用EMA评估且出现P/R/F1全为0，回退用当前模型再评估一次
        try:
            if used_ema_for_eval and (stats.get('precision@0.5', 0.0) == 0.0) and (stats.get('recall@0.5', 0.0) == 0.0):
                stats_base = evaluate_simple(
                    model,
                    val_loader,
                    device,
                    score_thr=_thr_now,
                    iou_thr=_iou_now,
                    log_interval=args.eval_log_interval,
                    eval_max_dets=args.eval_max_dets if args.eval_max_dets > 0 else None,
                    eval_nms=args.eval_nms if args.eval_nms > 0 else None,
                    eval_rpn_post_nms_top_n=args.eval_rpn_post_nms_topn if args.eval_rpn_post_nms_topn > 0 else None,
                    eval_tta_hflip=args.eval_tta_hflip,
                    energy_open_set=False,
                    conf_temp=args.conf_temp,
                    energy_rescore=False,
                    rescore_mode=args.rescore_mode, rescore_k=args.rescore_k, rescore_lambda=args.rescore_lambda,
                    soft_nms=args.soft_nms, soft_nms_sigma=args.soft_nms_sigma, soft_nms_iou=args.soft_nms_iou,
                    cascade_enable=False,
                    cascade_model=None,
                    debug_eval=args.debug_eval,
                    debug_eval_interval=args.debug_eval_interval,
                )
                # 仅当基础模型效果更好时替换
                if stats_base.get('f1@0.5', 0.0) > stats.get('f1@0.5', 0.0):
                    stats = stats_base
                    used_raw_for_eval = True
        except Exception:
            pass
        avg_loss = (running_loss / max(n_images, 1))
        # 当前学习率（取各 param group 平均）
        try:
            lrs = [pg.get('lr', 0.0) for pg in optimizer.param_groups]
            lr_now = float(sum(lrs) / max(1, len(lrs)))
        except Exception:
            lr_now = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} | loss {avg_loss:.4f} | P {stats['precision@0.5']:.2f} R {stats['recall@0.5']:.2f} F1 {stats['f1@0.5']:.2f}")

        # 统一写 log.csv（包含 lr）
        # 防止未定义开放集指标变量导致写入失败
        os_TKR = None
        os_TUR = None
        os_KP = None
        try:
            if not os.path.exists(log_csv):
                with open(log_csv, 'w', encoding='utf-8', newline='') as f:
                    f.write('epoch,lr,loss,precision@0.5,recall@0.5,f1@0.5,TP,FP,FN,energy_gamma,energy_temp,TKR,TUR,KP\n')
            with open(log_csv, 'a', encoding='utf-8', newline='') as f:
                tkr = f"{os_TKR:.6f}" if os_TKR is not None else ''
                tur = f"{os_TUR:.6f}" if os_TUR is not None else ''
                kp = f"{os_KP:.6f}" if os_KP is not None else ''
                f.write(f"{epoch+1},{lr_now:.6g},{avg_loss:.6f},{stats['precision@0.5']:.6f},{stats['recall@0.5']:.6f},{stats['f1@0.5']:.6f},{int(stats['TP'])},{int(stats['FP'])},{int(stats['FN'])},{args.energy_gamma:.6g},{args.energy_temp:.6g},{tkr},{tur},{kp}\n")
        except Exception:
            pass

        # 保存 last.pt（使用 EMA 权重以便推理稳定）
        raw_sd = model.state_dict()
        ema_sd = (ema.ema.state_dict() if ema is not None else {})
        ckpt_last = {
            'epoch': epoch,
            # 兼容字段：model 指向本轮用于评估/保存的权重
            'model': (ema_sd if (ema is not None and not used_raw_for_eval) else raw_sd),
            # 明确保存两份，便于后续选择
            'model_raw': raw_sd,
            'model_ema': ema_sd,
            'optimizer': optimizer.state_dict(),
            'scheduler': (main_scheduler.state_dict() if hasattr(main_scheduler, 'state_dict') else {}),
            'args': vars(args),
            'stats': stats,
            'kd_proj': (kd_proj.state_dict() if kd_proj is not None else {}),
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

        # 早停：在 warmup 轮数之后监控 F1，超过耐心值则停止
        if getattr(args, 'early_stop', False) and (epoch + 1) >= int(getattr(args, 'early_stop_warmup', 15)):
            # 初始化计数器（绑定到函数闭包外部变量时使用 nonlocal 复杂，这里用属性挂载在函数上）
            if not hasattr(main, '_es_bad_epochs'):
                setattr(main, '_es_bad_epochs', 0)
                setattr(main, '_es_best', best_f1)
            es_bad = getattr(main, '_es_bad_epochs')
            es_best = getattr(main, '_es_best')
            min_delta = float(getattr(args, 'early_stop_min_delta', 0.1))
            if (best_f1 - es_best) > min_delta:
                setattr(main, '_es_bad_epochs', 0)
                setattr(main, '_es_best', best_f1)
            else:
                setattr(main, '_es_bad_epochs', es_bad + 1)
            if getattr(main, '_es_bad_epochs') >= int(getattr(args, 'early_stop_patience', 15)):
                try:
                    print(f"[EarlyStop] No F1 improvement > {min_delta} for {args.early_stop_patience} epochs. Stopping at epoch {epoch+1}.")
                except Exception:
                    pass
                break

    print(f"Done. Best F1@0.5: {best_f1:.2f}. Checkpoint: {best_path}")


if __name__ == '__main__':
    main()
