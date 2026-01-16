import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from PIL import Image

# ensure project root (parent of scripts/) is on sys.path so 'src' can be imported when running from any CWD
_PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

# local imports
from src.data.dataset_npz import NPZDataset
from src.models.smnet_backbone import SMNetBackbone
from src.models.smnet_head import AnchorHead, MultiScaleHead, decode_head_output, decode_multi_head_output, generate_anchors
from src.models.smnet_loss import DetectionLoss, bbox_iou_xywh


# ---------- helpers ----------
def collate_fn(batch):
    xs, ys = [], []
    for X, y in batch:
        xs.append(torch.from_numpy(X))  # [3,512,512]
        ys.append(torch.from_numpy(y))  # [N,5] [cls,cx,cy,w,h] normalized
    Xb = torch.stack(xs, dim=0)
    return Xb, ys


def parse_anchor_sizes(s: str) -> List[Tuple[int, int]]:
    # format: "w1x h1,w2xh2,..." examples: "8x8,12x12,16x16,24x24"
    out: List[Tuple[int, int]] = []
    for part in s.split(','):
        part = part.strip().lower()
        if not part:
            continue
        if 'x' not in part:
            raise ValueError(f"Bad anchor size: {part}")
        a, b = part.split('x')
        # robust cleanup: remove any leading non-digit artifacts (e.g. stray '\\' from PowerShell escaping)
        import re
        a_clean = re.sub(r"[^0-9]", "", a)
        b_clean = re.sub(r"[^0-9]", "", b)
        if not a_clean or not b_clean:
            raise ValueError(f"Bad anchor size token after cleaning: '{a}' 'x' '{b}'")
        out.append((int(a_clean), int(b_clean)))
    if not out:
        raise ValueError("No anchor sizes parsed")
    return out


def sort_anchor_sizes(anchor_sizes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return sorted(anchor_sizes, key=lambda wh: (wh[0] * wh[1], wh[0], wh[1]))


def _peek_first_feature(dataset) -> int:
    from torch.utils.data import Subset, ConcatDataset

    if isinstance(dataset, Subset):
        return _peek_first_feature(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            try:
                return _peek_first_feature(ds)
            except Exception:
                continue
        raise RuntimeError("ConcatDataset has no accessible child samples")
    sample = dataset[0]
    feats = sample[0] if isinstance(sample, tuple) else sample
    return int(getattr(feats, 'shape', [3])[0])


def nms_xywh(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> List[int]:
    # boxes: [N,4] cx,cy,w,h -> convert to xyxy
    if boxes.numel() == 0:
        return []
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    order = scores.argsort(descending=True)
    keep: List[int] = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_i = (x2[i] - x1[i]).clamp(min=0) * (y2[i] - y1[i]).clamp(min=0)
        area_j = (x2[order[1:]] - x1[order[1:]]).clamp(min=0) * (y2[order[1:]] - y1[order[1:]]).clamp(min=0)
        iou = inter / (area_i + area_j - inter + 1e-6)
        order = order[1:][iou <= iou_thres]
    return keep


def nms_xywh_classwise(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_thres: float) -> List[int]:
    if boxes.numel() == 0:
        return []
    keep_all: List[int] = []
    unique_labels = labels.unique()
    for c in unique_labels:
        idx = (labels == c).nonzero(as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue
        keep_c = nms_xywh(boxes[idx], scores[idx], iou_thres)
        keep_all.extend(idx[torch.tensor(keep_c, device=idx.device)].tolist())
    if not keep_all:
        return []
    keep_all = list(set(keep_all))
    # sort by score descending for stable evaluation
    keep_scores = scores[keep_all]
    order = torch.argsort(keep_scores, descending=True)
    return [keep_all[i] for i in order.tolist()]


def evaluate(model: nn.Module,
             anchors_pix: torch.Tensor,
             stride: int,
             loader: DataLoader,
             num_classes: int,
             conf_thres: float = 0.05,
             nms_iou: float = 0.5,
             eval_conf_thres: List[float] | None = None,
             agnostic_nms: bool = False,
             device: str | torch.device = 'cuda',
             max_batches: int = 0,
             eval_topk: int = 300,
             return_aps: bool = False,
             dump_vis: int = 0,
             vis_dir: Path | None = None,
             per_class_ap: bool = False):
    model.eval()
    # per-class lists of (score, is_tp)
    preds_per_class: List[List[Tuple[float, int]]] = [[] for _ in range(num_classes)]
    gt_count_per_class = [0 for _ in range(num_classes)]
    eval_conf_thres = eval_conf_thres or [0.05, 0.25, 0.5]
    prefilter_thres = min(eval_conf_thres) if eval_conf_thres else conf_thres

    dumped = 0
    with torch.no_grad():
        for bi_loader, (Xb, ys) in enumerate(loader, start=1):
            Xb = Xb.to(device, non_blocking=True)
            # build targets (pixels)
            targets: List[Dict[str, torch.Tensor]] = []
            for y in ys:
                if y.numel() == 0:
                    targets.append({"boxes": torch.zeros((0, 4), device=device),
                                    "labels": torch.zeros((0,), dtype=torch.long, device=device)})
                else:
                    y = y.to(device)
                    cls = y[:, 0].long()
                    cxcywh = y[:, 1:] * 512.0
                    targets.append({"boxes": cxcywh, "labels": cls})
                    for c in cls.tolist():
                        if 0 <= c < num_classes:
                            gt_count_per_class[c] += 1

            # forward
            out = model(Xb)
            if model.multi_scale:
                conf_logits, pred_boxes, cls_logits = decode_multi_head_output(out, model.anchor_sizes_levels, model.strides)
            else:
                conf_logits, pred_boxes, cls_logits = decode_head_output(out, model.anchor_sizes, stride)
            conf = torch.sigmoid(conf_logits)
            # per-image decode
            for bi in range(Xb.size(0)):
                scores = conf[bi]
                boxes = pred_boxes[bi]
                cls_scores = torch.softmax(cls_logits[bi], dim=-1)
                # optional pre-filter by confidence threshold
                if prefilter_thres is not None and prefilter_thres > 0:
                    mask = scores > prefilter_thres
                    if mask.sum() == 0:
                        continue
                    scores = scores[mask]
                    boxes = boxes[mask]
                    cls_scores = cls_scores[mask]
                # keep top-K before NMS to avoid over-pruning weak but useful small-object preds
                if eval_topk and scores.numel() > eval_topk:
                    topk = torch.topk(scores, k=eval_topk)
                    idx = topk.indices
                    scores = scores[idx]
                    boxes = boxes[idx]
                    cls_scores = cls_scores[idx]
                # pick best class per anchor
                probs, labels = cls_scores.max(dim=-1)
                scores = scores * probs
                if scores.numel() == 0:
                    continue
                if agnostic_nms:
                    keep = nms_xywh(boxes, scores, nms_iou)
                else:
                    keep = nms_xywh_classwise(boxes, scores, labels, nms_iou)
                if len(keep) == 0:
                    continue
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                # match to GT for TP/FP at IoU=0.5
                gt = targets[bi]
                gt_boxes = gt["boxes"]
                gt_labels = gt["labels"]
                used = torch.zeros((gt_boxes.size(0),), dtype=torch.bool, device=device)
                for j in range(boxes.size(0)):
                    c = int(labels[j].item())
                    if c < 0 or c >= num_classes:
                        continue
                    if gt_boxes.numel() == 0:
                        preds_per_class[c].append((float(scores[j].item()), 0))
                        continue
                    # compute IoU to same-class GT
                    same = (gt_labels == c).nonzero(as_tuple=False).reshape(-1)
                    if same.numel() == 0:
                        preds_per_class[c].append((float(scores[j].item()), 0))
                        continue
                    iou = bbox_iou_xywh(boxes[j].unsqueeze(0), gt_boxes[same])[0]  # [K]
                    i = torch.argmax(iou)
                    if iou[i] >= 0.5 and not used[same[i]]:
                        used[same[i]] = True
                        preds_per_class[c].append((float(scores[j].item()), 1))
                    else:
                        preds_per_class[c].append((float(scores[j].item()), 0))

                # optional dump of predictions to JSON for first few images
                if dump_vis and dumped < dump_vis:
                    try:
                        import json as _json
                        dd = []
                        for j in range(boxes.size(0)):
                            dd.append({
                                "score": float(scores[j].item()),
                                "label": int(labels[j].item()),
                                "box": [float(v) for v in boxes[j].tolist()],  # cx,cy,w,h in pixels
                            })
                        meta = {
                            "preds": dd,
                            "gt": [{
                                "label": int(gt_labels[k].item()),
                                "box": [float(v) for v in gt_boxes[k].tolist()]
                            } for k in range(gt_boxes.size(0))]
                        }
                        vis_root = vis_dir or Path("vis")
                        vis_root.mkdir(parents=True, exist_ok=True)
                        out_fp = vis_root / f"eval_{bi_loader:05d}_{bi:02d}.json"
                        with open(out_fp, 'w', encoding='utf-8') as f:
                            _json.dump(meta, f, ensure_ascii=False)
                        # pseudo visualization treating golden triplet as RGB
                        feat = Xb[bi][:3].detach().cpu().clamp(0.0, 1.0).numpy()
                        vis_img = np.transpose(feat, (1, 2, 0)) * 255.0
                        vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
                        Image.fromarray(vis_img).save(out_fp.with_suffix('.png'))
                        dumped += 1
                    except Exception:
                        pass
            if max_batches and bi_loader >= max_batches:
                break

    # compute AP per class (VOC07 11-point) + recall/precision at best F1
    def ap_from_scores(scores_tp: List[Tuple[float, int]], gt_count: int) -> Tuple[float, float, float]:
        """Returns (AP, recall@F1, precision@F1)"""
        if gt_count == 0:
            return 0.0, 0.0, 0.0
        scores_tp.sort(key=lambda x: x[0], reverse=True)
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        for _, is_tp in scores_tp:
            if is_tp:
                tp += 1
            else:
                fp += 1
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (gt_count + 1e-6)
            precisions.append(prec)
            recalls.append(rec)
        
        # Compute AP (11-point)
        ap = 0.0
        for t in [i / 10 for i in range(11)]:
            p = 0.0
            for r, pr in zip(recalls, precisions):
                if r >= t:
                    p = max(p, pr)
            ap += p / 11.0
        
        # Find best F1 point
        best_f1 = 0.0
        best_recall = 0.0
        best_precision = 0.0
        for rec, prec in zip(recalls, precisions):
            f1 = 2 * prec * rec / (prec + rec + 1e-6)
            if f1 > best_f1:
                best_f1 = f1
                best_recall = rec
                best_precision = prec
        
        return ap, best_recall, best_precision

    results = [ap_from_scores(preds_per_class[c], gt_count_per_class[c]) for c in range(num_classes)]
    aps = [r[0] for r in results]
    recalls = [r[1] for r in results]
    precisions = [r[2] for r in results]
    
    valid = [a for a, g in zip(aps, gt_count_per_class) if g > 0]
    if not valid:
        return (0.0, aps, gt_count_per_class, recalls, precisions) if return_aps else 0.0
    m = float(np.mean(valid))
    # fixed-threshold PR (per-class TP/FP/FN + micro/macro)
    fixed_reports: Dict[str, Dict[str, object]] = {}
    for th in eval_conf_thres:
        tp_list = []
        fp_list = []
        fn_list = []
        per_class = []
        for c in range(num_classes):
            gt = gt_count_per_class[c]
            if gt <= 0:
                continue
            entries = preds_per_class[c]
            tp = sum(1 for s, is_tp in entries if s >= th and is_tp == 1)
            fp = sum(1 for s, is_tp in entries if s >= th and is_tp == 0)
            fn = max(0, gt - tp)
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            per_class.append((c, tp, fp, fn, prec, rec, gt))
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
        if tp_list:
            micro_tp = sum(tp_list)
            micro_fp = sum(fp_list)
            micro_fn = sum(fn_list)
            micro_p = micro_tp / (micro_tp + micro_fp + 1e-6)
            micro_r = micro_tp / (micro_tp + micro_fn + 1e-6)
            macro_p = sum(p for _, _, _, _, p, _, _ in per_class) / max(1, len(per_class))
            macro_r = sum(r for _, _, _, _, _, r, _ in per_class) / max(1, len(per_class))
        else:
            micro_p = micro_r = macro_p = macro_r = 0.0
        fixed_reports[f"{th:.2f}"] = {
            "per_class": per_class,
            "micro": (micro_p, micro_r),
            "macro": (macro_p, macro_r),
        }

    if return_aps:
        return m, aps, gt_count_per_class, recalls, precisions, fixed_reports
    if per_class_ap:
        # print per-class metrics
        present = [(ci, aps[ci], recalls[ci], precisions[ci], gt_count_per_class[ci]) 
                   for ci in range(num_classes) if gt_count_per_class[ci] > 0]
        print("  per-class metrics: (class_id, AP, Recall@F1, Prec@F1, #GT)")
        for ci, ap, rec, prec, gt in present:
            print(f"    cls{ci}: AP={ap:.4f}, R={rec:.4f}, P={prec:.4f}, GT={gt}")
    return m


class ModelWrap(nn.Module):
    def __init__(
        self,
        num_classes: int,
        anchor_sizes: List[Tuple[int, int]] | None = None,
        anchor_sizes_levels: List[List[Tuple[int, int]]] | None = None,
        in_ch: int = 3,
        multi_scale: bool = False,
    ):
        super().__init__()
        self.backbone = SMNetBackbone(in_ch=in_ch, base=64)
        self.multi_scale = bool(multi_scale)
        if self.multi_scale:
            if not anchor_sizes_levels:
                raise ValueError("anchor_sizes_levels is required when multi_scale=True")
            self.head = MultiScaleHead(ch=64, num_classes=num_classes, anchor_sizes_levels=anchor_sizes_levels)
            self.anchor_sizes_levels = anchor_sizes_levels
            self.strides = [8, 16, 32]
            self.anchor_sizes = None
        else:
            if not anchor_sizes:
                raise ValueError("anchor_sizes is required when multi_scale=False")
            self.head = AnchorHead(ch=64, num_classes=num_classes, anchor_sizes=anchor_sizes)
            self.anchor_sizes = anchor_sizes
            self.anchor_sizes_levels = None
            self.strides = [16]

    def forward(self, x: torch.Tensor):
        if self.multi_scale:
            feats = self.backbone(x, return_pyramid=True)
            return self.head(feats)
        fused = self.backbone(x)
        return self.head(fused)


def train_one_epoch(model: ModelWrap,
                    loader: DataLoader,
                    criterion: DetectionLoss,
                    anchors_pix: torch.Tensor,
                    stride: int,
                    optimizer: torch.optim.Optimizer,
                    scaler: GradScaler,
                    log_interval: int = 50,
                    max_steps: int = 0,
                    device: str | torch.device = 'cuda',
                    accumulation_steps: int = 1) -> Dict[str, float]:
    """One epoch training with optional gradient accumulation.

    accumulation_steps >= 2 reduces GPU memory at the cost of more steps per epoch.
    """
    model.train()
    total_losses: List[float] = []
    pos_sum = 0
    neg_sum = 0
    conf_sum = 0.0
    reg_sum = 0.0
    cls_sum = 0.0
    pbar = tqdm(loader, total=len(loader), desc="train", leave=False)
    optimizer.zero_grad(set_to_none=True)
    device_is_cuda = (device.type == 'cuda') if not isinstance(device, str) else (device == 'cuda')
    for bi, (Xb, ys) in enumerate(pbar, start=1):
        Xb = Xb.to(device, non_blocking=True)
        # build targets per sample in pixels
        targets = []
        for y in ys:
            if y.numel() == 0:
                targets.append({"boxes": torch.zeros((0, 4), device=device),
                                "labels": torch.zeros((0,), dtype=torch.long, device=device)})
            else:
                y = y.to(device)
                cls = y[:, 0].long()
                cxcywh = y[:, 1:] * 512.0
                targets.append({"boxes": cxcywh, "labels": cls})

        with autocast(device_type='cuda', enabled=device_is_cuda):
            out = model(Xb)
            if model.multi_scale:
                conf_logits, pred_boxes, cls_logits = decode_multi_head_output(out, model.anchor_sizes_levels, model.strides)
            else:
                conf_logits, pred_boxes, cls_logits = decode_head_output(out, model.anchor_sizes, stride)
            loss, stat = criterion(conf_logits, pred_boxes, cls_logits, anchors_pix, targets)
            loss = loss / float(max(1, accumulation_steps))

        scaler.scale(loss).backward()

        do_step = (bi % max(1, accumulation_steps) == 0) or (bi == len(loader))
        if do_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_losses.append(float(loss.detach().item()))
        pos_sum += stat["pos"]
        neg_sum += stat["neg"]
        # accumulate per-component stats if available
        conf_sum += float(stat.get("loss_conf", 0.0))
        reg_sum += float(stat.get("loss_reg", 0.0))
        cls_sum += float(stat.get("loss_cls", 0.0))
        if log_interval > 0 and (bi % log_interval == 0 or bi == len(loader)):
            avg_loss = float(np.mean(total_losses)) if total_losses else float(loss.detach().item())
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "pos": int(pos_sum), "neg": int(neg_sum),
                              "c": f"{conf_sum/bi:.4f}", "r": f"{reg_sum/bi:.4f}", "k": f"{cls_sum/bi:.4f}"})

        if max_steps and bi >= max_steps:
            break

    return {"loss": float(np.mean(total_losses) if total_losses else 0.0),
        "pos": float(pos_sum), "neg": float(neg_sum),
        "loss_conf": float(conf_sum / max(1, len(total_losses))),
        "loss_reg": float(reg_sum / max(1, len(total_losses))),
        "loss_cls": float(cls_sum / max(1, len(total_losses)))}


def main():
    ap = argparse.ArgumentParser(description="Train SMNet on NPZ dataset (S/E/C)")
    ap.add_argument("--train-json", type=str, required=True, help="Path to train.json (list of .npz)")
    ap.add_argument("--val-json", type=str, required=True, help="Path to val.json (list of .npz)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=None, help="Alias of --batch")
    ap.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps (default 1)")
    ap.add_argument("--num-workers", type=int, default=2, help="DataLoader num_workers (default 2). Use 0 on Windows if issues arise")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=3, help="epochs for linear warmup")
    ap.add_argument("--warmup-epochs", type=int, default=None, help="Alias of --warmup")
    ap.add_argument("--num-classes", type=int, default=24)
    ap.add_argument(
        "--anchor-sizes",
        type=str,
        default="6x5,7x6,10x5,12x6,12x6,12x7,8x19,7x23,7x23,8x22,7x29,18x16,19x20,34x22",
        help="Comma-separated WxH list for the single 32x32 grid (e.g. '6x5,7x6,...'). We'll auto-sort by area (small->large).",
    )
    ap.add_argument("--multi-scale", action="store_true", help="Enable true multi-scale detection (P2/P3/P4)")
    ap.add_argument("--anchor-sizes-p2", type=str, default="", help="Comma-separated WxH list for P2 (64x64, stride=8)")
    ap.add_argument("--anchor-sizes-p3", type=str, default="", help="Comma-separated WxH list for P3 (32x32, stride=16)")
    ap.add_argument("--anchor-sizes-p4", type=str, default="", help="Comma-separated WxH list for P4 (16x16, stride=32)")
    ap.add_argument("--conf-thres", type=float, default=0.05)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--agnostic-nms", action="store_true", help="Use class-agnostic NMS (default is class-wise)")
    ap.add_argument("--eval-conf-thres", type=str, default="0.05,0.25,0.5",
                    help="Comma-separated conf thresholds for fixed-PR evaluation")
    ap.add_argument("--pos-iou", type=float, default=0.5, help="Positive IOU threshold for anchor assignment")
    ap.add_argument("--neg-iou", type=float, default=0.4, help="Negative IOU threshold for anchor assignment")
    ap.add_argument("--conf-loss", type=str, default="bce", choices=["bce","focal"], help="Confidence loss type")
    ap.add_argument("--focal-alpha", type=float, default=0.25, help="Focal loss alpha (pos weighting)")
    ap.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma")
    ap.add_argument("--out", type=str, default=str(Path("Model/SMNet-Output").resolve()), help="output directory for logs and checkpoints")
    ap.add_argument("--log-interval", type=int, default=50, help="batches between progress prints")
    ap.add_argument("--val-interval", type=int, default=2, help="Epoch interval between validations (default 2)")
    ap.add_argument("--per-class-ap", action="store_true", help="Print per-class AP during evaluation")
    ap.add_argument("--dump-vis", type=int, default=0, help="During evaluation, dump predictions for N images to JSON (in out/vis)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--neg-topk-ratio", type=int, default=0, help="Hard negative mining: keep at most pos*ratio negatives (0 to disable)")
    # 兼容学生脚本常用参数（作为占位或轻量影响）
    ap.add_argument("--fast-mode", action="store_true", help="Reduce evaluation frequency/IO; bumps --val-interval to >=3")
    ap.add_argument("--resize", type=int, default=512, help="Input resize (only 512 supported; others will be coerced)")
    ap.add_argument("--input-mode", type=str, default=None, help="Compatibility placeholder (ignored)")
    ap.add_argument("--loss-type", type=str, default="paper", help="Compatibility placeholder (ignored)")
    ap.add_argument("--subset-frac", type=float, default=1.0, help="Use a fraction of the training set for quick experiments (0,1]")
    ap.add_argument("--max-train-steps", type=int, default=0, help="Limit number of training batches per epoch (for quick sanity runs)")
    ap.add_argument("--max-val-steps", type=int, default=0, help="Limit number of validation batches during mAP eval (for quick sanity runs)")
    ap.add_argument("--hflip", action="store_true", help="Enable random horizontal flip in training")
    ap.add_argument("--use-extra", action="store_true", help="Append preprocessed feature maps (edges,corners,gray) as extra channels")
    ap.add_argument("--extra-keys", type=str, default="", help="Optional comma list of extra feature keys to append (subset of edges,corners,gray); overrides --use-extra if non-empty")
    # background (pure negative) samples integration
    ap.add_argument("--background-json", type=str, default=None, help="Optional JSON index of background (no-object) images to include as additional negative-only samples")
    ap.add_argument("--background-frac", type=float, default=0.15, help="Fraction (0-1] of labeled train count to cap number of background samples (default 0.15). Ignored if --background-json not set")
    ap.add_argument("--background-max", type=int, default=0, help="Hard cap on number of background samples (0 = disable cap except fraction rule)")
    # mixup augmentation
    ap.add_argument("--mixup-prob", type=float, default=0.0, help="Probability of applying Mixup augmentation (0.0 to disable)")
    ap.add_argument("--mixup-alpha", type=float, default=0.5, help="Beta distribution parameter for Mixup")
    # resume / initialize from checkpoint
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pth) to resume training from (loads model, optimizer, scheduler, epoch)")
    ap.add_argument("--init-weights", type=str, default=None, help="Path to checkpoint (.pth) to load model weights only (no optimizer/scheduler state)")
    args = ap.parse_args()

    if args.resume and args.init_weights:
        raise ValueError("Cannot use --resume and --init-weights together. Choose one.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set seeds
    try:
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass
    out_dir = Path(args.out)
    (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")
    # persist hyper-params
    with open(out_dir / 'hparams.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    log_path = out_dir / 'log.jsonl'

    # datasets & loaders
    # 限制输入尺寸到 512
    if args.resize != 512:
        print(f"[warn] resize={args.resize} not supported; forcing 512.")
        args.resize = 512
    # decide extra feature maps
    if args.extra_keys.strip():
        raw_keys = [k.strip() for k in args.extra_keys.split(',') if k.strip()]
        extra_keys = [k for k in raw_keys if k in ('edges','corners','gray')]
    else:
        extra_keys = ['edges','corners','gray'] if args.use_extra else []
    if extra_keys:
        print(f"[extra] using extra feature maps: {extra_keys}")
    feature_key = 'golden_triplet'
    labeled_train_ds = NPZDataset(
        args.train_json,
        require_label=True,
        extra_keys=extra_keys,
        mixup_prob=args.mixup_prob,
        mixup_alpha=args.mixup_alpha,
        enable_hflip=bool(args.hflip),
        enable_mixup=(args.mixup_prob > 0.0),
        feature_key=feature_key,
    )
    val_ds = NPZDataset(
        args.val_json,
        require_label=True,
        extra_keys=extra_keys,
        enable_hflip=False,
        enable_mixup=False,
        feature_key=feature_key,
    )

    # optional subset on labeled set ONLY (before mixing background)
    if 0.0 < float(args.subset_frac) < 1.0:
        from torch.utils.data import Subset
        n_all_lab = len(labeled_train_ds)
        k_lab = max(1, int(n_all_lab * float(args.subset_frac)))
        rng_lab = np.random.RandomState(42)
        idxs_lab = rng_lab.choice(n_all_lab, size=k_lab, replace=False).tolist()
        labeled_train_ds = Subset(labeled_train_ds, idxs_lab)
        print(f"[subset-frac] labeled subset {k_lab}/{n_all_lab}")

    # integrate background dataset if provided
    train_ds = labeled_train_ds
    if args.background_json:
        try:
            from torch.utils.data import ConcatDataset, Subset
            bg_raw = NPZDataset(
                args.background_json,
                require_label=False,
                extra_keys=extra_keys,
                enable_hflip=False,
                enable_mixup=False,
                feature_key=feature_key,
            )
            bg_n_total = len(bg_raw)
            # decide how many to sample
            target_bg = int(len(labeled_train_ds) * float(args.background_frac)) if hasattr(labeled_train_ds, '__len__') else int(bg_n_total * float(args.background_frac))
            if args.background_max > 0:
                target_bg = min(target_bg, int(args.background_max))
            target_bg = max(0, min(target_bg, bg_n_total))
            if target_bg > 0 and target_bg < bg_n_total:
                rng_bg = np.random.RandomState(123)
                sel = rng_bg.choice(bg_n_total, size=target_bg, replace=False).tolist()
                bg_ds = Subset(bg_raw, sel)
            else:
                bg_ds = bg_raw
            train_ds = ConcatDataset([labeled_train_ds, bg_ds])
            print(f"[background] using {len(bg_ds)}/{bg_n_total} background samples (frac={args.background_frac}, max={args.background_max})")
        except Exception as e:
            print(f"[background][warn] failed to integrate background dataset: {e}")

    # optional infer num_classes by scanning train labels (only from labeled subset)
    if args.num_classes <= 0:
        max_cls = 0
        # unwrap if ConcatDataset
        scan_source = labeled_train_ds
        scan_len = len(scan_source)
        for i in range(min(scan_len, 2000)):
            _, y = scan_source[i]
            if y.size > 0:
                max_cls = max(max_cls, int(np.max(y[:, 0])))
        args.num_classes = max_cls + 1
        print(f"[infer] num_classes -> {args.num_classes}")

    # apply batch-size alias if provided
    if args.batch_size is not None:
        args.batch = int(args.batch_size)

    # apply warmup-epochs alias if provided
    if args.warmup_epochs is not None:
        args.warmup = int(args.warmup_epochs)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=max(0, args.num_workers), pin_memory=True, collate_fn=collate_fn)

    # model & loss
    anchor_sizes = sort_anchor_sizes(parse_anchor_sizes(args.anchor_sizes))
    print(f"[anchors] sorted (small->large by area): {anchor_sizes}")
    # infer input channel count from a sample (robust to Subset/Concat wrappers)
    try:
        in_ch = _peek_first_feature(train_ds)
    except Exception:
        in_ch = 3 + len(extra_keys)
    if in_ch == 3:
        print("[model] input channels = 3 (golden triplet order: Log-Spec, Gray-Norm, Corner-Mask)")
    else:
        print(f"[model] input channels = {in_ch} (golden triplet + {in_ch - 3} extra maps)")
    if args.multi_scale:
        def _split_sizes(sizes: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
            n = len(sizes)
            if n <= 2:
                return sizes, sizes, sizes
            k1 = max(1, n // 3)
            k2 = max(k1 + 1, (2 * n) // 3)
            p2 = sizes[:k1]
            p3 = sizes[k1:k2]
            p4 = sizes[k2:]
            return p2 or sizes, p3 or sizes, p4 or sizes

        p2_sizes = sort_anchor_sizes(parse_anchor_sizes(args.anchor_sizes_p2)) if args.anchor_sizes_p2.strip() else None
        p3_sizes = sort_anchor_sizes(parse_anchor_sizes(args.anchor_sizes_p3)) if args.anchor_sizes_p3.strip() else None
        p4_sizes = sort_anchor_sizes(parse_anchor_sizes(args.anchor_sizes_p4)) if args.anchor_sizes_p4.strip() else None
        if p2_sizes is None or p3_sizes is None or p4_sizes is None:
            sp2, sp3, sp4 = _split_sizes(anchor_sizes)
            p2_sizes = p2_sizes or sp2
            p3_sizes = p3_sizes or sp3
            p4_sizes = p4_sizes or sp4
        anchor_sizes_levels = [p2_sizes, p3_sizes, p4_sizes]
        print(f"[anchors][p2] {p2_sizes}")
        print(f"[anchors][p3] {p3_sizes}")
        print(f"[anchors][p4] {p4_sizes}")
        model = ModelWrap(num_classes=args.num_classes,
                          anchor_sizes_levels=anchor_sizes_levels,
                          in_ch=in_ch,
                          multi_scale=True).to(device)
    else:
        model = ModelWrap(num_classes=args.num_classes,
                          anchor_sizes=anchor_sizes,
                          in_ch=in_ch,
                          multi_scale=False).to(device)

    if args.init_weights:
        ckpt_path = Path(args.init_weights)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {ckpt_path}")
        print(f"[init-weights] loading model weights from {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device)
        state_dict = ckpt.get('model', ckpt)
        model.load_state_dict(state_dict, strict=True)

    criterion = DetectionLoss(num_classes=args.num_classes, w_conf=1.0, w_reg=2.0, w_cls=1.0,
                              pos_iou_th=args.pos_iou, neg_iou_th=args.neg_iou,
                              conf_loss=args.conf_loss, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
                              neg_topk_ratio=args.neg_topk_ratio)

    # optimizer & sched
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler(enabled=(device.type=='cuda'))
    
    # load checkpoint if resuming
    start_epoch = 1
    best_map = 0.0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f"[resume] loading checkpoint from {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location=device)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', 1) + 1
            best_map = ckpt.get('best_map', 0.0)
            print(f"[resume] continuing from epoch {start_epoch}, best_map={best_map:.4f}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    if start_epoch > args.epochs:
        print(f"[resume][warn] start_epoch ({start_epoch}) exceeds target --epochs ({args.epochs}). Resetting to 1 with fresh optimizer. Use --init-weights for cross-stage warm starts.")
        start_epoch = 1
        best_map = 0.0
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # cosine schedule excluding warmup
    t_max = max(1, args.epochs - args.warmup)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.lr * 0.1)
    # step scheduler to match resumed epoch if needed
    if start_epoch > args.warmup + 1:
        for _ in range(start_epoch - args.warmup - 1):
            scheduler.step()

    # anchors tensor (pixels)
    if args.multi_scale:
        anchors_p2 = generate_anchors(64, 64, 8, anchor_sizes_levels[0])
        anchors_p3 = generate_anchors(32, 32, 16, anchor_sizes_levels[1])
        anchors_p4 = generate_anchors(16, 16, 32, anchor_sizes_levels[2])
        anchors_pix = torch.cat([anchors_p2, anchors_p3, anchors_p4], dim=0).to(device)
        stride = 16
    else:
        stride, grid_h, grid_w = 16, 32, 32
        anchors_pix = generate_anchors(grid_h, grid_w, stride, anchor_sizes).to(device)

    # training loop (start from start_epoch if resumed)
    # fast-mode: 提高验证间隔，减少评估开销
    if args.fast_mode and args.val_interval < 3:
        args.val_interval = 3
    for epoch in range(start_epoch, args.epochs + 1):
        # basic warmup (epoch-level)
        if epoch <= args.warmup:
            for g in optimizer.param_groups:
                g['lr'] = args.lr * epoch / max(1, args.warmup)
        else:
            scheduler.step(epoch - args.warmup)
        print(f"\nEpoch {epoch}/{args.epochs} - lr={optimizer.param_groups[0]['lr']:.6f}")
        stat = train_one_epoch(model, train_loader, criterion, anchors_pix, stride, optimizer, scaler, log_interval=args.log_interval, max_steps=args.max_train_steps, device=device, accumulation_steps=args.accumulation_steps)
        print(f"  train: loss={stat['loss']:.4f} pos={stat['pos']:.0f} neg={stat['neg']:.0f} conf={stat.get('loss_conf',0.0):.4f} reg={stat.get('loss_reg',0.0):.4f} cls={stat.get('loss_cls',0.0):.4f}")
        # evaluate mAP@0.5 according to val interval
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            vis_dir = out_dir / 'vis'
            eval_conf_thres = [float(x) for x in args.eval_conf_thres.split(',') if x.strip()]
            map50, aps, gtc, recalls, precisions, fixed_reports = evaluate(
                model, anchors_pix, stride, val_loader, args.num_classes,
                conf_thres=args.conf_thres, nms_iou=args.nms_iou, eval_conf_thres=eval_conf_thres,
                agnostic_nms=args.agnostic_nms, device=device, max_batches=args.max_val_steps,
                eval_topk=300, return_aps=True, dump_vis=args.dump_vis, vis_dir=vis_dir,
                per_class_ap=args.per_class_ap
            )
            print(f"  val: AP50(VOC07 11-pt)={map50:.4f}")
            if args.per_class_ap:
                present = [(i, float(aps[i]), float(recalls[i]), float(precisions[i]), int(gtc[i])) 
                           for i in range(len(aps)) if gtc[i] > 0]
                print("  per-class AP: [(class_id, AP, Recall@F1, Prec@F1, #GT)]")
                print(f"  {present}")
            # fixed-threshold PR outputs
            for th_key, rep in fixed_reports.items():
                micro_p, micro_r = rep["micro"]
                macro_p, macro_r = rep["macro"]
                print(f"  PR@conf>={th_key} (IoU=0.5): micro P={micro_p:.4f} R={micro_r:.4f} | macro P={macro_p:.4f} R={macro_r:.4f}")
                if args.per_class_ap:
                    print("    per-class (cls, TP, FP, FN, P, R, #GT):")
                    print(f"    {rep['per_class']}")
            # epoch log -> jsonl
            log_dict = {
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'loss': stat['loss'],
                'pos': stat['pos'],
                'neg': stat['neg'],
                'map50': map50,
            }
            # Add per-class metrics to log
            for i in range(len(aps)):
                if gtc[i] > 0:
                    log_dict[f'ap_cls{i}'] = float(aps[i])
                    log_dict[f'recall_cls{i}'] = float(recalls[i])
                    log_dict[f'precision_cls{i}'] = float(precisions[i])
            
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")
            # checkpoints
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
                'best_map50': best_map,
            }
            torch.save(ckpt, out_dir / 'ckpts' / f'epoch_{epoch:03d}.pth')
            if map50 > best_map:
                best_map = map50
                torch.save(ckpt, out_dir / 'best.pth')
                print("  ✓ saved best checkpoint")

    print(f"\nTraining done. Best mAP@0.5={best_map:.4f}")


if __name__ == "__main__":
    main()
