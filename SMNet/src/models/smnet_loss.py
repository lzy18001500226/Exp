from __future__ import annotations

from typing import Dict, List

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_iou_xywh(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute IoU between two sets of (cx, cy, w, h) boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    b1 = boxes1.unsqueeze(1)
    b2 = boxes2.unsqueeze(0).to(boxes1.device)

    x1_1 = b1[..., 0] - b1[..., 2] * 0.5
    y1_1 = b1[..., 1] - b1[..., 3] * 0.5
    x2_1 = b1[..., 0] + b1[..., 2] * 0.5
    y2_1 = b1[..., 1] + b1[..., 3] * 0.5

    x1_2 = b2[..., 0] - b2[..., 2] * 0.5
    y1_2 = b2[..., 1] - b2[..., 3] * 0.5
    x2_2 = b2[..., 0] + b2[..., 2] * 0.5
    y2_2 = b2[..., 1] + b2[..., 3] * 0.5

    inter_x1 = torch.maximum(x1_1, x1_2)
    inter_y1 = torch.maximum(y1_1, y1_2)
    inter_x2 = torch.minimum(x2_1, x2_2)
    inter_y2 = torch.minimum(y2_1, y2_2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (x2_1 - x1_1).clamp(min=0) * (y2_1 - y1_1).clamp(min=0)
    area2 = (x2_2 - x1_2).clamp(min=0) * (y2_2 - y1_2).clamp(min=0)
    union = (area1 + area2 - inter).clamp(min=eps)
    return inter / union


def bbox_ciou_xywh(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute CIoU between two sets of (cx, cy, w, h) boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    b1 = boxes1.unsqueeze(1)
    b2 = boxes2.unsqueeze(0).to(boxes1.device)

    x1_1 = b1[..., 0] - b1[..., 2] * 0.5
    y1_1 = b1[..., 1] - b1[..., 3] * 0.5
    x2_1 = b1[..., 0] + b1[..., 2] * 0.5
    y2_1 = b1[..., 1] + b1[..., 3] * 0.5

    x1_2 = b2[..., 0] - b2[..., 2] * 0.5
    y1_2 = b2[..., 1] - b2[..., 3] * 0.5
    x2_2 = b2[..., 0] + b2[..., 2] * 0.5
    y2_2 = b2[..., 1] + b2[..., 3] * 0.5

    inter_x1 = torch.maximum(x1_1, x1_2)
    inter_y1 = torch.maximum(y1_1, y1_2)
    inter_x2 = torch.minimum(x2_1, x2_2)
    inter_y2 = torch.minimum(y2_1, y2_2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (x2_1 - x1_1).clamp(min=0) * (y2_1 - y1_1).clamp(min=0)
    area2 = (x2_2 - x1_2).clamp(min=0) * (y2_2 - y1_2).clamp(min=0)
    union = (area1 + area2 - inter).clamp(min=eps)
    iou = inter / union

    # center distance
    rho2 = (b1[..., 0] - b2[..., 0]).pow(2) + (b1[..., 1] - b2[..., 1]).pow(2)
    # enclosing box diagonal
    c_x1 = torch.minimum(x1_1, x1_2)
    c_y1 = torch.minimum(y1_1, y1_2)
    c_x2 = torch.maximum(x2_1, x2_2)
    c_y2 = torch.maximum(y2_1, y2_2)
    c2 = (c_x2 - c_x1).pow(2) + (c_y2 - c_y1).pow(2)
    c2 = c2.clamp(min=eps)

    # aspect ratio term
    v = (4 / (torch.pi ** 2)) * (torch.atan(b1[..., 2] / (b1[..., 3] + eps)) - torch.atan(b2[..., 2] / (b2[..., 3] + eps))).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2) - alpha * v
    return ciou


def bbox_iou_xywh_aligned(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute IoU for aligned (cx, cy, w, h) boxes (one-to-one)."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0],))
    b1 = boxes1
    b2 = boxes2
    x1_1 = b1[:, 0] - b1[:, 2] * 0.5
    y1_1 = b1[:, 1] - b1[:, 3] * 0.5
    x2_1 = b1[:, 0] + b1[:, 2] * 0.5
    y2_1 = b1[:, 1] + b1[:, 3] * 0.5
    x1_2 = b2[:, 0] - b2[:, 2] * 0.5
    y1_2 = b2[:, 1] - b2[:, 3] * 0.5
    x2_2 = b2[:, 0] + b2[:, 2] * 0.5
    y2_2 = b2[:, 1] + b2[:, 3] * 0.5

    inter_x1 = torch.maximum(x1_1, x1_2)
    inter_y1 = torch.maximum(y1_1, y1_2)
    inter_x2 = torch.minimum(x2_1, x2_2)
    inter_y2 = torch.minimum(y2_1, y2_2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (x2_1 - x1_1).clamp(min=0) * (y2_1 - y1_1).clamp(min=0)
    area2 = (x2_2 - x1_2).clamp(min=0) * (y2_2 - y1_2).clamp(min=0)
    union = (area1 + area2 - inter).clamp(min=eps)
    return inter / union


class DetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        w_conf: float = 1.0,
        w_reg: float = 2.0,
        w_cls: float = 1.0,
        pos_iou_th: float = 0.5,
        neg_iou_th: float = 0.4,
        conf_loss: str = "bce",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        neg_topk_ratio: int = 0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.w_conf = w_conf
        self.w_reg = w_reg
        self.w_cls = w_cls
        self.pos_iou_th = pos_iou_th
        self.neg_iou_th = neg_iou_th
        self.conf_loss = conf_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.neg_topk_ratio = neg_topk_ratio
        self._assign_stats = _AssignStats.from_env()

    def forward(
        self,
        conf_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        cls_logits: torch.Tensor,
        anchors_pix: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        device = conf_logits.device
        anchors = anchors_pix.to(device)
        batch_size, num_anchors = conf_logits.shape

        total_conf = conf_logits.new_tensor(0.0)
        total_reg = conf_logits.new_tensor(0.0)
        total_cls = conf_logits.new_tensor(0.0)
        pos_total = 0
        neg_total = 0

        for b in range(batch_size):
            conf_b = conf_logits[b]
            bbox_b = pred_boxes[b]
            cls_b = cls_logits[b]

            tgt = targets[b]
            boxes = tgt["boxes"].to(device)
            labels = tgt["labels"].to(device).long()

            if boxes.numel() == 0:
                pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
                neg_mask = torch.ones(num_anchors, dtype=torch.bool, device=device)
                assigned_boxes = boxes.new_zeros((0, 4))
                assigned_labels = labels.new_zeros((0,), dtype=torch.long)
            else:
                ious = bbox_iou_xywh(anchors, boxes)
                best_iou, best_idx = ious.max(dim=1)
                best_anchor_for_gt = ious.argmax(dim=0)
                pos_mask = best_iou >= self.pos_iou_th
                pos_mask[best_anchor_for_gt] = True
                neg_mask = best_iou <= self.neg_iou_th
                assigned_boxes = boxes[best_idx[pos_mask]]
                assigned_labels = labels[best_idx[pos_mask]]

            pos_count = int(pos_mask.sum().item())
            neg_mask = self._select_negatives(conf_b, neg_mask.clone(), pos_count)
            neg_count = int(neg_mask.sum().item())

            if self._assign_stats.enabled:
                self._assign_stats.update(
                    anchors=anchors,
                    pos_mask=pos_mask,
                    neg_mask=neg_mask,
                    best_iou=best_iou if boxes.numel() > 0 else None,
                    best_idx=best_idx if boxes.numel() > 0 else None,
                    gt_count=int(boxes.shape[0]),
                )

            target_conf = torch.zeros_like(conf_b)
            if pos_count > 0:
                iou_pos = bbox_iou_xywh_aligned(bbox_b[pos_mask], assigned_boxes).clamp(0.0, 1.0)
                target_conf[pos_mask] = iou_pos.to(target_conf.dtype)
            mask = pos_mask | neg_mask
            if mask.any():
                conf_loss = self._conf_loss(conf_b[mask], target_conf[mask])
                total_conf += conf_loss
            if pos_count > 0:
                ciou = bbox_ciou_xywh(bbox_b[pos_mask], assigned_boxes)
                reg_loss = (1.0 - ciou).sum()
                cls_loss = F.cross_entropy(cls_b[pos_mask], assigned_labels, reduction="sum")
                total_reg += reg_loss
                total_cls += cls_loss

            pos_total += pos_count
            neg_total += neg_count

        denom = max(1, batch_size)
        loss_conf = total_conf / denom
        loss_reg = total_reg / denom
        loss_cls = total_cls / denom
        loss = self.w_conf * loss_conf + self.w_reg * loss_reg + self.w_cls * loss_cls

        stats = {
            "loss_conf": float(loss_conf.detach().item()),
            "loss_reg": float(loss_reg.detach().item()),
            "loss_cls": float(loss_cls.detach().item()),
            "pos": float(pos_total),
            "neg": float(neg_total),
        }
        return loss, stats

    def _conf_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.conf_loss == "focal":
            # 【Fix 1】限制 logits 范围，防止 sigmoid 下溢或上溢
            logits = logits.clamp(-10.0, 10.0)

            prob = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            pt = prob * targets + (1 - prob) * (1 - targets)

            # 【Fix 2】给 pt 加一个极小值，防止 (1-pt) 甚至 log(pt) 出现数值问题
            pt = pt.clamp(min=1e-6, max=1.0 - 1e-6)

            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            loss = alpha_t * (1 - pt).pow(self.focal_gamma) * ce
            return loss.sum()
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")

    def _select_negatives(self, logits: torch.Tensor, neg_mask: torch.Tensor, pos_count: int) -> torch.Tensor:
        if self.neg_topk_ratio <= 0 or neg_mask.sum() == 0:
            return neg_mask
        max_neg = int(pos_count * self.neg_topk_ratio)
        if pos_count == 0:
            max_neg = max(int(self.neg_topk_ratio), 1)
        max_neg = max(max_neg, 1)
        neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)
        if neg_indices.numel() <= max_neg:
            return neg_mask
        scores = torch.sigmoid(logits[neg_indices].detach())
        topk = torch.topk(scores, k=max_neg, largest=True)
        keep = neg_indices[topk.indices]
        new_mask = torch.zeros_like(neg_mask)
        new_mask[keep] = True
        return new_mask


class _AssignStats:
    def __init__(self, enabled: bool, max_batches: int, out_path: Path, bins: int = 20) -> None:
        self.enabled = bool(enabled)
        self.max_batches = int(max_batches)
        self.out_path = Path(out_path)
        self.bins = int(bins)
        self.batch_count = 0
        self.image_pos_counts: List[int] = []
        self.image_neg_counts: List[int] = []
        self.gt_pos_counts: List[int] = []
        self.best_iou_min: List[float] = []
        self.best_iou_mean: List[float] = []
        self.best_iou_p50: List[float] = []
        self.best_iou_p90: List[float] = []
        self.best_iou_hist = torch.zeros(self.bins, dtype=torch.long)
        self.best_iou_total = 0
        self.anchor_pos_counts: Dict[str, int] = {}
        self._finalized = False

    @staticmethod
    def from_env() -> "_AssignStats":
        enabled = os.getenv("SMNET_ASSIGN_STATS", "0").strip() in ("1", "true", "True")
        max_batches = int(os.getenv("SMNET_ASSIGN_STATS_MAX_BATCHES", "300"))
        out_path = Path(os.getenv("SMNET_ASSIGN_STATS_OUT", "analysis_output/smnet_assign_stats.json"))
        return _AssignStats(enabled=enabled, max_batches=max_batches, out_path=out_path)

    def update(
        self,
        anchors: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
        best_iou: torch.Tensor | None,
        best_idx: torch.Tensor | None,
        gt_count: int,
    ) -> None:
        if not self.enabled or self._finalized:
            return

        self.image_pos_counts.append(int(pos_mask.sum().item()))
        self.image_neg_counts.append(int(neg_mask.sum().item()))

        if gt_count > 0 and best_idx is not None:
            pos_gt = best_idx[pos_mask]
            if pos_gt.numel() > 0:
                counts = torch.bincount(pos_gt, minlength=gt_count)
                self.gt_pos_counts.extend([int(x) for x in counts.tolist()])

        if best_iou is not None and best_iou.numel() > 0:
            bi = best_iou.detach()
            self.best_iou_min.append(float(bi.min().item()))
            self.best_iou_mean.append(float(bi.mean().item()))
            self.best_iou_p50.append(float(torch.quantile(bi, 0.5).item()))
            self.best_iou_p90.append(float(torch.quantile(bi, 0.9).item()))
            hist = torch.histc(bi, bins=self.bins, min=0.0, max=1.0).to(torch.long)
            self.best_iou_hist += hist.cpu()
            self.best_iou_total += int(bi.numel())

        if anchors.numel() > 0 and pos_mask.any():
            sizes = anchors[pos_mask][:, 2:4].detach().round().to(torch.int64)
            if sizes.numel() > 0:
                uniq, counts = torch.unique(sizes, dim=0, return_counts=True)
                for (w, h), c in zip(uniq.tolist(), counts.tolist()):
                    key = f"{int(w)}x{int(h)}"
                    self.anchor_pos_counts[key] = self.anchor_pos_counts.get(key, 0) + int(c)

        self.batch_count += 1
        if self.batch_count >= self.max_batches:
            self._finalize()

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        def _summarize(vals: List[float | int]) -> Dict[str, float]:
            if not vals:
                return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
            t = torch.tensor(vals, dtype=torch.float32)
            return {
                "min": float(t.min().item()),
                "mean": float(t.mean().item()),
                "p50": float(torch.quantile(t, 0.5).item()),
                "p90": float(torch.quantile(t, 0.9).item()),
            }

        hist = self.best_iou_hist.tolist()
        frac_lt_02 = 0.0
        if self.best_iou_total > 0:
            bins_lt_02 = int(self.bins * 0.2)
            frac_lt_02 = float(sum(hist[:bins_lt_02]) / max(1, self.best_iou_total))

        payload = {
            "batches": self.batch_count,
            "image_pos_counts": _summarize(self.image_pos_counts),
            "image_neg_counts": _summarize(self.image_neg_counts),
            "gt_pos_counts": _summarize(self.gt_pos_counts),
            "best_iou_stats": {
                "min": _summarize(self.best_iou_min),
                "mean": _summarize(self.best_iou_mean),
                "p50": _summarize(self.best_iou_p50),
                "p90": _summarize(self.best_iou_p90),
                "hist": hist,
                "bins": self.bins,
                "total": self.best_iou_total,
                "frac_lt_0.2": frac_lt_02,
            },
            "anchor_pos_counts": dict(sorted(self.anchor_pos_counts.items(), key=lambda kv: kv[0])),
        }
        self.out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

