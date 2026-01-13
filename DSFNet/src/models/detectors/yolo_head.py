import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOHeadSingleScale(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, num_anchors: int = 3):
        super().__init__()
        self.na = num_anchors
        self.nc = num_classes
        self.cv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.cv2 = nn.Conv2d(in_ch, self.na * (5 + self.nc), 1)

    def forward(self, x):
        x = F.silu(self.bn1(self.cv1(x)))
        return self.cv2(x)

class DetectorModel(nn.Module):
    def __init__(self, backbone, head: YOLOHeadSingleScale):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        p3, p4 = self.backbone(x)
        # 使用更语义的 p4（下采样 1/32）作为单尺度头
        return self.head(p4)
