from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    """Explicit position encoding P(n,m) with sin/cos over width index (paper-aligned)."""

    def __init__(self):
        super().__init__()
        self._cache: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        key = (h, w, x.device, x.dtype)
        if key in self._cache:
            return x + self._cache[key]

        m = torch.arange(w, device=x.device, dtype=x.dtype)
        n = torch.arange(h, device=x.device, dtype=x.dtype).unsqueeze(1)
        omega = torch.pow(10000.0, -m / max(1.0, float(w)))
        phase = n * omega.view(1, w)
        mask = (m % 2 == 1).view(1, w)
        P = torch.where(mask, torch.sin(phase), torch.cos(phase))
        P = P.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        self._cache[key] = P
        return x + P


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
        ctx_x = x.mean(dim=2)  # [b,c,w]
        ctx_x = self.fx(ctx_x)
        ctx_x = ctx_x.unsqueeze(2).expand(-1, -1, h, -1)
        ctx_y = x.mean(dim=3)  # [b,c,h]
        ctx_y = self.fy(ctx_y)
        ctx_y = ctx_y.unsqueeze(3).expand(-1, -1, -1, w)
        out = x + 0.5 * (ctx_x + ctx_y)
        return self.bn(out)


class DualAxisAttention(nn.Module):
    """Paper-style dual-direction attention (original + transposed)."""

    def __init__(self, ch: int):
        super().__init__()
        self.ax = AxialAttention(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.ax(x)
        y2 = self.ax(x.transpose(2, 3)).transpose(2, 3)
        return 0.5 * (y1 + y2)


class PsiBlock(nn.Module):
    """Standardized conv unit: Conv(3x3) + BN + ReLU + MaxPool(stride=2)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.pool(x)


class SMNetBackbone(nn.Module):
    """SMNet backbone aligned to paper: explicit PE + dual attention + Psi pyramid + fused 32x32 output."""

    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        self.use_extra = in_ch > 3
        self.input_proj = nn.Conv2d(in_ch, 3, 1, 1, 0, bias=False) if self.use_extra else None
        self.pos = PositionalEncoding2D()
        self.attn = DualAxisAttention(3)

        # Psi stack to produce 64/32/16 feature maps with paper channels
        self.psi1 = PsiBlock(3, base)          # 512 -> 256
        self.psi2 = PsiBlock(base, base * 2)   # 256 -> 128
        self.psi3 = PsiBlock(base * 2, base * 4)   # 128 -> 64
        self.psi4 = PsiBlock(base * 4, base * 8)   # 64 -> 32
        self.psi5 = PsiBlock(base * 8, base * 16)  # 32 -> 16

        # FPN-style fusion (paper Eq.19/20)
        self.up_from_16 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(base * 16, base * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )
        self.up_from_32 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(base * 8, base * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )
        self.proj_64 = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )
        self.down_from_64 = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )
        self._attn_notice_printed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_extra and self.input_proj is not None:
            x = self.input_proj(x)
        x = self.pos(x)
        x = self.attn(x)
        self._maybe_log_attention()

        x256 = self.psi1(x)
        x128 = self.psi2(x256)
        x64 = self.psi3(x128)
        x32 = self.psi4(x64)
        x16 = self.psi5(x32)

        xhat32 = x32 + self.up_from_16(x16)
        xhat64 = self.proj_64(x64) + self.up_from_32(xhat32)
        fused = self.conv_mid(xhat32) + self.down_from_64(xhat64)
        return fused

    def _maybe_log_attention(self) -> None:
        if not self._attn_notice_printed:
            print("[SMNetBackbone] PositionalEncoding + DualAxisAttention active.")
            self._attn_notice_printed = True
