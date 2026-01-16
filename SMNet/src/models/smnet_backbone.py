from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class TFBlock(nn.Module):
    """Texture-focused block: depthwise separable conv + residual."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(ch, ch)
        self.conv2 = DepthwiseSeparableConv(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + identity


class AxialAttention(nn.Module):
    """Lightweight dual-axis attention proxy: channel mixing with average pooling over each axis."""

    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        mid = max(4, ch // reduction)
        self.fx = nn.Sequential(
            nn.Conv1d(ch, mid, 1, bias=False), nn.ReLU(inplace=True), nn.Conv1d(mid, ch, 1, bias=False)
        )
        self.fy = nn.Sequential(
            nn.Conv1d(ch, mid, 1, bias=False), nn.ReLU(inplace=True), nn.Conv1d(mid, ch, 1, bias=False)
        )
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # time-axis context (along width)
        ctx_x = x.mean(dim=2)  # [b,c,w]
        ctx_x = self.fx(ctx_x)  # [b,c,w]
        ctx_x = ctx_x.unsqueeze(2).expand(-1, -1, h, -1)
        # freq-axis context (along height)
        ctx_y = x.mean(dim=3)  # [b,c,h]
        ctx_y = self.fy(ctx_y)  # [b,c,h]
        ctx_y = ctx_y.unsqueeze(3).expand(-1, -1, -1, w)
        out = x + 0.5 * (ctx_x + ctx_y)
        return self.bn(out)


class PFBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.ax = AxialAttention(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ax(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(4, ch // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, ch // reduction), ch, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SMNetBackbone(nn.Module):
    """SMNet backbone with always-on channel attention over the leading triplet."""

    def __init__(self, in_ch: int = 3, base: int = 64):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        # RGB/golden stem (always assume first 3 channels reflect the golden triplet order)
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, 2, 1, bias=False),  # -> 256x256
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.rgb_ca = ChannelAttention(c1)
        self.use_extra = in_ch > 3
        if self.use_extra:
            extra_ch = in_ch - 3
            # Extra branch: preserve spatial resolution parity with rgb_stem output (256x256)
            self.extra_branch = nn.Sequential(
                nn.Conv2d(extra_ch, c1, 3, 2, 1, bias=False),  # -> 256x256
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c1, c1, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
            )
            # Fuse concat([rgb, extra]) -> base
            self.fuse = nn.Sequential(
                nn.Conv2d(c1 * 2, c1, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
            )
        else:
            self.extra_branch = None
            self.fuse = None
        self._attn_notice_printed = False
        # stage2 -> 128x128 then 64x64
        self.down2 = nn.Conv2d(c1, c2, 3, 2, 1, bias=False)
        self.tf2 = TFBlock(c2)
        self.pf2 = PFBlock(c2)
        self.down2b = nn.Conv2d(c2, c2, 3, 2, 1, bias=False)

        # stage3 -> 32x32
        self.down3 = nn.Conv2d(c2, c3, 3, 2, 1, bias=False)
        self.tf3 = TFBlock(c3)
        self.pf3 = PFBlock(c3)

        # Feature projections for fusion (P2=64x64, P3=32x32, P4=16x16)
        self.lateral2 = nn.Conv2d(c2, base, 1, 1, 0)
        self.lateral3 = nn.Conv2d(c3, base, 1, 1, 0)
        self.smooth2 = nn.Conv2d(base, base, 3, 1, 1)
        self.smooth3 = nn.Conv2d(base, base, 3, 1, 1)
        self.p4 = nn.Conv2d(base, base, 3, 2, 1)  # 32->16

        # Fusion layers following Eq.(20)
        self.conv_mid = nn.Sequential(
            nn.Conv2d(base, base, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.down_from_p2 = nn.Sequential(
            nn.Conv2d(base, base, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.up_from_p4 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(base, base, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, return_pyramid: bool = False):
        if self.use_extra:
            rgb = x[:, :3]
            extra = x[:, 3:]
            rgb_feat = self.rgb_ca(self.rgb_stem(rgb))
            extra_feat = self.extra_branch(extra)
            x = self.fuse(torch.cat([rgb_feat, extra_feat], dim=1))
        else:
            x = self.rgb_ca(self.rgb_stem(x))
        self._maybe_log_attention()
        x2 = self.down2(x)
        x2 = self.tf2(x2); x2 = self.pf2(x2)
        x2 = self.down2b(x2)  # 64x64

        x3 = self.down3(x2)  # 32x32
        x3 = self.tf3(x3); x3 = self.pf3(x3)

        p3 = self.lateral3(x3)
        p2 = self.lateral2(x2) + F.interpolate(p3, scale_factor=2.0, mode="nearest")
        p2 = self.smooth2(p2)
        p3 = self.smooth3(p3)
        p4 = self.p4(p3)

        fused = self.conv_mid(p3) + self.down_from_p2(p2) + self.up_from_p4(p4)
        if return_pyramid:
            return [p2, p3, p4]
        return fused

    def _maybe_log_attention(self) -> None:
        if not self._attn_notice_printed:
            print(
                "[SMNetBackbone] ChannelAttention active on golden triplet inputs (R=Log, G=Gray, B=Corner)."
            )
            self._attn_notice_printed = True
