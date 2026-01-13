import argparse
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
# 新增：调度器
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

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
from torchvision.ops import box_iou

from det_coco_dataset import CocoRFDataset
from dataset import seed_worker


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

    def _clone_model(self, model: nn.Module) -> nn.Module:
        import copy
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + msd[k].to(v.device, dtype=v.dtype) * (1.0 - d))


def build_model(num_classes: int, imagenet_backbone: bool = True, no_pretrained: bool = False, backbone_ckpt: str = "") -> FasterRCNN:
    # 使用 ResNet50-FPN 作为骨干（速度/精度均衡）；默认不加载任何在线权重
    use_weights = (imagenet_backbone and not no_pretrained)
    weights = "DEFAULT" if use_weights else None
    # 关键：若不使用预训练或未启用 imagenet_backbone，则显式禁用 backbone 预训练权重，避免隐式下载
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


def evaluate_simple(model: nn.Module, loader: DataLoader, device: torch.device, score_thr: float = 0.5, iou_thr: float = 0.5) -> Dict[str, float]:
    model.eval()
    TP = FP = FN = 0
    with torch.no_grad():
        it = loader if tqdm is None else tqdm(loader, desc="Eval", ncols=100, ascii=True)
        for images, targets in it:
            images, targets = to_device(list(images), list(targets), device)
            sanitize_targets(images, targets)
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                scores = out['scores']
                keep = scores >= score_thr
                boxes_p = out['boxes'][keep]
                labels_p = out['labels'][keep]
                boxes_g = tgt['boxes']
                labels_g = tgt['labels']
                # 按分数降序
                if boxes_p.numel() > 0:
                    order = torch.argsort(scores[keep], descending=True)
                    boxes_p = boxes_p[order]
                    labels_p = labels_p[order]
                tp, fp, fn = greedy_match_iou(boxes_p, labels_p, boxes_g, labels_g, iou_thr=iou_thr)
                TP += tp; FP += fp; FN += fn
        if tqdm is not None and hasattr(it, 'close'):
            it.close()
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision@0.5": 100.0 * precision, "recall@0.5": 100.0 * recall, "f1@0.5": 100.0 * f1,
            "TP": float(TP), "FP": float(FP), "FN": float(FN)}


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
    parser.add_argument('--backbone_imagenet', action='store_true', help='use ImageNet-pretrained backbone')
    parser.add_argument('--no_pretrained', action='store_true', help='disable any pretrained loading (default for offline)')
    parser.add_argument('--backbone_ckpt', type=str, default='', help='path to local fasterrcnn checkpoint (optional)')
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

    args = parser.parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    # 统一输出目录到项目根目录，避免在 src 下误建子目录
    if not args.out_dir or args.out_dir.strip() == '':
        args.out_dir = str((Path(FILE_DIR).parent / 'checkpoints_det').resolve())
    os.makedirs(args.out_dir, exist_ok=True)

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
        # 验证集显式引用训练用 annotations 作为 categories 参考，避免 label 映射错位
        ds_val_full = CocoRFDataset(images_dir=args.images_dir, annotations_json=args.annotations, augment=False,
                                     categories_ref_json=args.annotations)
        N = len(ds_train_full)
        idx_all = list(range(N))
        # 先按 seed 打乱索引再划分，避免验证集偶然集中为无目标样本
        rnd = random.Random(args.seed)
        rnd.shuffle(idx_all)
        n_val = int(N * max(0.0, min(0.9, args.val_split)))
        val_idx = idx_all[:n_val]
        train_idx = idx_all[n_val:]
        from torch.utils.data import Subset
        ds_train = Subset(ds_train_full, train_idx)
        ds_val = Subset(ds_val_full, val_idx)

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

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=False, worker_init_fn=seed_worker, generator=g,
                              persistent_workers=(args.workers > 0), collate_fn=collate_fn)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, drop_last=False, worker_init_fn=seed_worker, generator=g,
                            persistent_workers=(args.workers > 0), collate_fn=collate_fn)

    # 启动时对验证集做一次GT统计自检
    try:
        pos = tot = boxes = 0
        for i in range(len(ds_val)):
            _, tgt = ds_val[i]
            tot += 1
            b = tgt.get('boxes', None)
            if b is not None and getattr(b, 'numel', lambda: 0)() > 0:
                pos += 1
                boxes += int(b.size(0)) if hasattr(b, 'size') else 0
        print(f"[Sanity] Val GT images with boxes: {pos}/{tot} | total boxes: {boxes}")
    except Exception:
        pass

    # 类别数：从数据集中读取映射（背景为 0）
    num_classes = _infer_num_classes(ds_train)

    # 不下载预训练：默认 no_pretrained=True；如明确传入 --backbone_imagenet 且未禁用，则使用本地缓存权重
    model = build_model(num_classes=num_classes,
                        imagenet_backbone=args.backbone_imagenet,
                        no_pretrained=args.no_pretrained,
                        backbone_ckpt=args.backbone_ckpt).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
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

    scaler = GradScaler(enabled=args.amp)
    ema = ModelEMA(model, decay=args.ema_decay, device=device) if args.ema else None

    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, 'det_best.pt')

    for epoch in range(args.epochs):
        model.train()
        it = train_loader if tqdm is None else tqdm(train_loader, desc=f"Train {epoch+1}/{args.epochs}", ncols=100, ascii=True)
        running_loss = 0.0
        n_images = 0
        steps = 0
        for images, targets in it:
            images, targets = to_device(list(images), list(targets), device)
            sanitize_targets(images, targets)
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with autocast():
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
            if tqdm is not None:
                it.set_postfix({"loss": f"{(running_loss/max(n_images,1)):.4f}"})
            if args.dry_run > 0 and steps >= args.dry_run:
                break
        if tqdm is not None and hasattr(it, 'close'):
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
        stats = evaluate_simple(eval_model, val_loader, device, score_thr=args.score_thr, iou_thr=args.iou_thr)
        print(f"Epoch {epoch+1}/{args.epochs} | loss {(running_loss/max(n_images,1)):.4f} | P {stats['precision@0.5']:.2f} R {stats['recall@0.5']:.2f} F1 {stats['f1@0.5']:.2f}")

        if stats['f1@0.5'] > best_f1:
            best_f1 = stats['f1@0.5']
            ckpt = {
                'epoch': epoch,
                'model': (ema.ema.state_dict() if ema is not None else model.state_dict()),
                'optimizer': optimizer.state_dict(),
                'scheduler': (main_scheduler.state_dict() if hasattr(main_scheduler, 'state_dict') else {}),
                'args': vars(args),
                'stats': stats,
            }
            torch.save(ckpt, best_path)

    print(f"Done. Best F1@0.5: {best_f1:.2f}. Checkpoint: {best_path}")


if __name__ == '__main__':
    main()
