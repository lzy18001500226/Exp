from __future__ import annotations

from typing import Dict, List

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

            target_conf = torch.zeros_like(conf_b)
            target_conf[pos_mask] = 1.0
            mask = pos_mask | neg_mask
            if mask.any():
                conf_loss = self._conf_loss(conf_b[mask], target_conf[mask])
                total_conf += conf_loss
            if pos_count > 0:
                reg_loss = F.smooth_l1_loss(bbox_b[pos_mask], assigned_boxes, reduction="sum")
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
