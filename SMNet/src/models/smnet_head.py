from typing import List, Tuple

import torch
import torch.nn as nn


def generate_anchors(grid_h: int, grid_w: int, strides: int, sizes: List[Tuple[int, int]]):
    """Generate anchor boxes in pixel units on a grid.
    Returns tensor [A,4] in (cx,cy,w,h) pixel format.
    """
    device = torch.device("cpu")
    ys, xs = torch.meshgrid(
        torch.arange(grid_h, device=device), torch.arange(grid_w, device=device), indexing="ij"
    )
    xs = (xs + 0.5) * strides
    ys = (ys + 0.5) * strides
    base = []
    for (aw, ah) in sizes:
        cx = xs.reshape(-1)
        cy = ys.reshape(-1)
        w = torch.full_like(cx, float(aw))
        h = torch.full_like(cy, float(ah))
        base.append(torch.stack([cx, cy, w, h], dim=1))
    return torch.cat(base, dim=0)  # [(H*W*B),4]


class AnchorHead(nn.Module):
    def __init__(self, ch: int, num_classes: int, anchor_sizes: List[Tuple[int, int]]):
        super().__init__()
        self.num_classes = num_classes
        self.anchor_sizes = anchor_sizes
        B = len(anchor_sizes)
        # conv head
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True)
        )
        self.pred = nn.Conv2d(ch, B * (1 + 4 + num_classes), 1, 1, 0)

    def forward(self, p: torch.Tensor):
        x = self.conv(p)
        out = self.pred(x)  # [B, B*(1+4+C), H, W]
        return out


def decode_head_output(out: torch.Tensor, anchor_sizes: List[Tuple[int, int]], stride: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode raw head output to conf, boxes, cls_logits.
    out: [N, B*(1+4+C), H, W] ->
      conf: [N, A]
      boxes: [N, A, 4] (cx,cy,w,h) in pixel units
      cls: [N, A, C]
    """
    n, c, h, w = out.shape
    B = len(anchor_sizes)
    # basic shape checks
    if c % B != 0:
        raise ValueError(f"Channel dim {c} is not divisible by anchors-per-cell {B}")
    C = c // B - 5
    if C <= 0:
        raise ValueError(f"Decoded num_classes <= 0 from channels={c}, B={B}")
    out = out.view(n, B, 1 + 4 + C, h, w)
    conf = out[:, :, 0]
    reg = out[:, :, 1:5]
    cls = out[:, :, 5:]
    conf = conf.permute(0, 1, 2, 3).reshape(n, -1)
    cls = cls.permute(0, 1, 3, 4, 2).reshape(n, -1, C)

    # build anchors
    anchors = generate_anchors(h, w, stride, anchor_sizes).to(out.device)  # [A,4]
    A = anchors.shape[0]
    reg = reg.permute(0, 1, 3, 4, 2).reshape(n, A, 4)
    # decode offsets: cx,cy = a_c + dxy*aw/ah; w,h = aw*exp(dw), ah*exp(dh)
    acx, acy, aw, ah = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    dxy = reg[:, :, 0:2]
    dwh = reg[:, :, 2:4]
    cx = acx + dxy[..., 0] * aw
    cy = acy + dxy[..., 1] * ah
    # clamp for stability to avoid huge exp at init
    ww = aw * torch.exp(dwh[..., 0].clamp(-4.0, 4.0))
    hh = ah * torch.exp(dwh[..., 1].clamp(-4.0, 4.0))
    boxes = torch.stack([cx, cy, ww, hh], dim=-1)
    return conf, boxes, cls


class MultiScaleHead(nn.Module):
    """Multi-level detection head wrapper with one AnchorHead per FPN level."""
    def __init__(self, ch: int, num_classes: int, anchor_sizes_levels: List[List[Tuple[int, int]]]):
        super().__init__()
        self.heads = nn.ModuleList([AnchorHead(ch=ch, num_classes=num_classes, anchor_sizes=sizes)
                                     for sizes in anchor_sizes_levels])
        self.anchor_sizes_levels = anchor_sizes_levels

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # features must align with heads order
        assert len(features) == len(self.heads), f"features({len(features)}) != heads({len(self.heads)})"
        outs = []
        for head, feat in zip(self.heads, features):
            outs.append(head(feat))
        return outs


def decode_multi_head_output(outs: List[torch.Tensor], anchor_sizes_levels: List[List[Tuple[int, int]]],
                             strides: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode and concatenate multi-level outputs.
    Returns:
      conf: [N, A_all]
      boxes: [N, A_all, 4]
      cls: [N, A_all, C]
    """
    assert len(outs) == len(anchor_sizes_levels) == len(strides)
    confs, boxes_list, clss = [], [], []
    for out, sizes, s in zip(outs, anchor_sizes_levels, strides):
        c, b, cl = decode_head_output(out, sizes, s)
        confs.append(c)
        boxes_list.append(b)
        clss.append(cl)
    conf = torch.cat(confs, dim=1)
    boxes = torch.cat(boxes_list, dim=1)
    cls = torch.cat(clss, dim=1)
    return conf, boxes, cls
