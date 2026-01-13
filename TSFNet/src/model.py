import math
from typing import Tuple, Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
# 新增：引入 torchvision 用于 MobileNetV3
try:
    import torchvision.models as tvm
except Exception:
    tvm = None

try:
    import timm  # 可选，仅用于从完整 Swin 模型中提取 stage4 权重
except Exception:
    timm = None


# === 新增：安全选择多头数，保证可被 dim 整除 ===
def _safe_nheads(dim: int, prefer: int) -> int:
    prefer = max(1, int(prefer))
    for h in range(min(prefer, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


# ----------------------
# Basic building blocks
# ----------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(c // r, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class DWConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1):
        super().__init__()
        p = k // 2
        self.dw = ConvBNAct(in_c, in_c, k, s, p, g=in_c)
        self.pw = ConvBNAct(in_c, out_c, 1, 1, 0)

    def forward(self, x):
        return self.pw(self.dw(x))


class C2f(nn.Module):
    """Simplified C2f block: 1x1 reduce -> 3x3 conv -> 1x1 expand with residual"""
    def __init__(self, in_c, out_c, hid_c=None, n=1):
        super().__init__()
        hid_c = hid_c or out_c // 2
        self.cv1 = ConvBNAct(in_c, hid_c, 1, 1, 0)
        self.m = nn.Sequential(*[ConvBNAct(hid_c, hid_c, 3, 1) for _ in range(n)])
        self.cv2 = ConvBNAct(hid_c, out_c, 1, 1, 0)
        self.short = (in_c == out_c)

    def forward(self, x):
        y = self.cv2(self.m(self.cv1(x)))
        return y + x if self.short else y


# ----------------------
# Backbones (Texture / Position streams)
# ----------------------
class TextureBackbone(nn.Module):
    """A light YOLOv8-cls style backbone producing P3/P4/P5"""
    def __init__(self, in_c=3, base_c=64):
        super().__init__()
        c1, c2, c3, c4 = base_c, base_c * 2, base_c * 4, base_c * 8
        self.stem = ConvBNAct(in_c, c1, 3, 2)           # 256x256
        self.stage2 = nn.Sequential(                     # 128x128
            ConvBNAct(c1, c2, 3, 2),
            C2f(c2, c2)
        )
        self.stage3 = nn.Sequential(                     # 64x64  -> P3
            ConvBNAct(c2, c3, 3, 2),
            C2f(c3, c3)
        )
        self.stage4 = nn.Sequential(                     # 32x32  -> P4
            ConvBNAct(c3, c4, 3, 2),
            C2f(c4, c4)
        )
        self.stage5 = nn.Sequential(                     # 16x16  -> P5
            ConvBNAct(c4, c4, 3, 2),
            C2f(c4, c4)
        )
        self.out_channels = (c3, c4, c4)  # P3, P4, P5 channels

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5


class PositionBackbone(nn.Module):
    """Lightweight position stream with DWConv + SE."""
    def __init__(self, in_c=3, base_c=32):
        super().__init__()
        c1, c2, c3, c4 = base_c, base_c * 2, base_c * 4, base_c * 8
        self.stem = DWConvBlock(in_c, c1, 3, 2)         # 256x256
        self.stage2 = nn.Sequential(                    # 128x128
            DWConvBlock(c1, c2, 3, 2),
            SEBlock(c2)
        )
        self.stage3 = nn.Sequential(                    # 64x64 -> P3
            DWConvBlock(c2, c3, 3, 2),
            SEBlock(c3)
        )
        self.stage4 = nn.Sequential(                    # 32x32 -> P4
            DWConvBlock(c3, c4, 3, 2),
            SEBlock(c4)
        )
        self.stage5 = nn.Sequential(                    # 16x16 -> P5
            DWConvBlock(c4, c4, 3, 2),
            SEBlock(c4)
        )
        self.out_channels = (c3, c4, c4)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5


# 新增：基于 torchvision 的 MobileNetV3-Large 位置流骨干
class PositionBackboneMobileNetV3(nn.Module):
    """MobileNetV3-Large features as position stream, exporting P3/P4/P5 at strides 8/16/32.
    将 torchvision 的 mobilenet_v3_large.features 作为特征提取器，动态检测下采样尺度。
    """
    def __init__(self):
        super().__init__()
        if tvm is None:
            raise ImportError('torchvision is required for PositionBackboneMobileNetV3')
        m = tvm.mobilenet_v3_large(weights=None)
        self.m = m.features  # nn.Sequential
        # 通过一次 dummy 前向推断出各尺度通道数
        with torch.no_grad():
            x = torch.zeros(1, 3, 256, 256)
            p3, p4, p5 = self.forward(x)
            self.out_channels = (p3.shape[1], p4.shape[1], p5.shape[1])

    def _collect_multiscale(self, x: torch.Tensor):
        h0, w0 = x.shape[-2:]
        got8 = got16 = got32 = False
        p3 = p4 = p5 = None
        cur = x
        for layer in self.m:
            cur = layer(cur)
            h, w = cur.shape[-2:]
            # 计算相对 stride（整除安全起见用 round）
            stride_h = max(1, round(h0 / max(h, 1)))
            stride_w = max(1, round(w0 / max(w, 1)))
            stride = max(stride_h, stride_w)
            if (not got8) and stride >= 8:
                p3 = cur
                got8 = True
            if (not got16) and stride >= 16:
                p4 = cur
                got16 = True
            if (not got32) and stride >= 32:
                p5 = cur
                got32 = True
        # 兜底：若极端输入导致未到达目标 stride，则用最后特征
        last = cur
        p3 = p3 if p3 is not None else last
        p4 = p4 if p4 is not None else last
        p5 = p5 if p5 is not None else last
        return p3, p4, p5

    def forward(self, x):
        return self._collect_multiscale(x)


# ----------------------
# Neck (Fusion + FPN/PANet-lite)
# ----------------------
class FusionPANet(nn.Module):
    def __init__(self, tex_chs: Tuple[int, int, int], pos_chs: Tuple[int, int, int], out_chs: Tuple[int, int, int] = (192, 384, 768), align_resize: bool = False):
        super().__init__()
        c3, c4, c5 = out_chs
        self.out_chs = out_chs
        self.align_resize = align_resize
        # 同尺度拼接对齐
        self.align3 = ConvBNAct(tex_chs[0] + pos_chs[0], c3, 1, 1, 0)
        self.align4 = ConvBNAct(tex_chs[1] + pos_chs[1], c4, 1, 1, 0)
        self.align5 = ConvBNAct(tex_chs[2] + pos_chs[2], c5, 1, 1, 0)
        # 局部细化
        self.refine3 = C2f(c3, c3)
        self.refine4 = C2f(c4, c4)
        self.refine5 = C2f(c5, c5)
        # 自顶向下
        self.td4 = C2f(c4 + c5, c4)
        self.td3 = C2f(c3 + c4, c3)
        # 自底向上
        self.bu4 = C2f(c4 + c3, c4)
        self.bu5 = C2f(c5 + c4, c5)

    def _cat_align(self, a: torch.Tensor, b: torch.Tensor, name: str) -> torch.Tensor:
        if a.shape[-2:] != b.shape[-2:]:
            if not self.align_resize:
                raise RuntimeError(f"FusionPANet: spatial mismatch at {name}: tex {tuple(a.shape[-2:])} vs pos {tuple(b.shape[-2:])}. Enable align_resize or ensure same strides.")
            b = F.interpolate(b, size=a.shape[-2:], mode='nearest')
        return torch.cat([a, b], dim=1)

    def forward(self, p3_tex, p4_tex, p5_tex, p3_pos, p4_pos, p5_pos):
        # 融合同尺度（必要时做紧急最近邻对齐）
        f3 = self.refine3(self.align3(self._cat_align(p3_tex, p3_pos, 'P3')))
        f4 = self.refine4(self.align4(self._cat_align(p4_tex, p4_pos, 'P4')))
        f5 = self.refine5(self.align5(self._cat_align(p5_tex, p5_pos, 'P5')))
        # top-down
        up5 = F.interpolate(f5, size=f4.shape[-2:], mode='nearest')
        p4_td = self.td4(torch.cat([f4, up5], dim=1))
        up4 = F.interpolate(p4_td, size=f3.shape[-2:], mode='nearest')
        p3_td = self.td3(torch.cat([f3, up4], dim=1))
        # bottom-up
        down3 = F.max_pool2d(p3_td, kernel_size=2)
        p4_out = self.bu4(torch.cat([p4_td, down3], dim=1))
        down4 = F.max_pool2d(p4_out, kernel_size=2)
        p5_out = self.bu5(torch.cat([f5, down4], dim=1))
        return p3_td, p4_out, p5_out  # 通道 [c3,c4,c5]


# 新增：timm 的 Swin BasicLayer（可选）
try:
    # timm>=1.0 中没有 BasicLayer，使用 SwinTransformerStage
    from timm.models.swin_transformer import SwinTransformerStage as TimmSwinStage
except Exception:
    TimmSwinStage = None


class SwinStage4Adapter(nn.Module):
    """Swin-Small/Base Stage4 适配器，保持空间尺寸不变，仅做窗口注意力堆叠。
    - 输入/输出：BCHW，C 应与目标 stage4 通道一致（Small=768，Base=1024）。
    - 内部：使用 timm SwinTransformerStage（blocks + norm），输入格式 BHWC，需要提供输入分辨率。
    """
    def __init__(self, dim: int = 768, depth: int = 2, num_heads: int = 24, window_size: int = 7, drop_path: float = 0.0, input_resolution: Tuple[int,int]=(8,8)):
        super().__init__()
        self.input_resolution = input_resolution
        if TimmSwinStage is None:
            # 回退：使用轻量 TransformerEncoder（全局注意力），确保 nhead 整除 dim
            self.stage = None
            nhead = _safe_nheads(dim, num_heads)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=nhead,
                dim_feedforward=int(dim * 4),
                dropout=0.1,
                batch_first=True,
                activation='gelu'
            )
            self.fallback = nn.ModuleList([nn.TransformerEncoder(encoder_layer, num_layers=1) for _ in range(depth)])
        else:
            dprs = [drop_path for _ in range(depth)] if isinstance(drop_path, float) else drop_path
            self.stage = TimmSwinStage(
                dim=dim,
                out_dim=dim,  # 不改变通道
                input_resolution=input_resolution,
                depth=depth,
                downsample=False,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=4.0,
                qkv_bias=True,
                proj_drop=0.0,
                attn_drop=0.0,
                drop_path=dprs,
                norm_layer=nn.LayerNorm,
            )
            self.fallback = None
        self.is_stage4 = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.stage is None:
            # 回退路径（batch_first）：BCHW -> BLC -> BCHW
            x_seq = x.flatten(2).transpose(1, 2)  # (B, L, C)
            for enc in self.fallback:
                x_seq = enc(x_seq)
            return x_seq.transpose(1, 2).reshape(b, c, h, w).contiguous()
        # timm SwinTransformerStage 路径：BCHW -> BHWC -> BCHW
        if (h, w) != self.input_resolution:
            # 根据实际分辨率重建 stage，并迁移已有权重与设备
            old = self.stage
            old_sd = old.state_dict()
            dim = x.shape[1]
            depth = len(old.blocks)
            heads = old.blocks[0].attn.num_heads
            window = old.blocks[0].attn.window_size
            # drop_path 可能是 ModuleList，提取概率
            dprs = []
            for bl in old.blocks:
                dp = getattr(bl, 'drop_path', None)
                prob = float(getattr(dp, 'drop_prob', 0.0)) if dp is not None else 0.0
                dprs.append(prob)
            new_stage = TimmSwinStage(
                dim=dim, out_dim=dim, input_resolution=(h, w), depth=depth, downsample=False,
                num_heads=heads, window_size=window, mlp_ratio=4.0, qkv_bias=True,
                proj_drop=0.0, attn_drop=0.0, drop_path=dprs, norm_layer=nn.LayerNorm
            )
            new_stage = new_stage.to(x.device)
            new_sd = new_stage.state_dict()
            for k, v in old_sd.items():
                if k in new_sd and new_sd[k].shape == v.shape:
                    new_sd[k].copy_(v.detach())
            new_stage.load_state_dict(new_sd)
            self.stage = new_stage
            self.input_resolution = (h, w)
        x_bhwc = x.permute(0, 2, 3, 1).contiguous()
        x_bhwc = self.stage(x_bhwc)
        return x_bhwc.permute(0, 3, 1, 2).contiguous()


# ----------------------
# Swin-lite (optional)
# ----------------------
class SwinOrTransformerLite(nn.Module):
    """轻量回退实现：当不使用专用 Stage4 或 timm 不可用时使用。"""
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        nhead = _safe_nheads(dim, num_heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.blocks = nn.ModuleList([nn.TransformerEncoder(encoder_layer, num_layers=1) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, L, C)
        for enc in self.blocks:
            x_seq = enc(x_seq)
        out = x_seq.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return out


# ----------------------
# Full Model
# ----------------------
class TwoStreamClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        # 输出通道按尺度配置，默认对齐 YOLOv8m 惯例
        out_chs: Tuple[int, int, int] = (192, 384, 768),
        use_swin: bool = False,
        swin_heads: int = 24,  # 对齐 Swin-Small stage4 heads=24
        swin_layers: int = 2,  # 对齐 Swin-Small stage4 深度=2
        pos_backbone: str = 'dwconv',
        align_resize: bool = False,
    ):
        super().__init__()
        self.out_chs = out_chs
        # streams
        # 将纹理流输入从3通道改为6通道（[S,t,f] + 伪彩色C[3]）
        self.tex = TextureBackbone(in_c=6, base_c=64)
        if pos_backbone == 'mobilenetv3':
            self.pos = PositionBackboneMobileNetV3()
        else:
            self.pos = PositionBackbone(in_c=3, base_c=32)
        # 在模型内实现可学习的 1x1 伪彩色映射：C = Conv1x1(S)
        self.pseudo_color = nn.Conv2d(1, 3, kernel_size=1, bias=True)
        # neck
        self.neck = FusionPANet(self.tex.out_channels, self.pos.out_channels, out_chs=out_chs, align_resize=align_resize)
        # optional Swin on F5（通道=out_chs[2]，默认 768）
        self.use_swin = use_swin
        if use_swin:
            if TimmSwinStage is not None and out_chs[2] in (768, 1024):
                self.swin = SwinStage4Adapter(dim=out_chs[2], depth=swin_layers, num_heads=swin_heads, window_size=7, drop_path=0.0, input_resolution=(8,8))
            else:
                self.swin = SwinOrTransformerLite(dim=out_chs[2], num_heads=max(1, swin_heads//3), num_layers=swin_layers)
        else:
            self.swin = nn.Identity()
        # head: GAP -> FC（暂用分类头，后续可替换为检测头）
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_chs[2], num_classes)

    @staticmethod
    def energy(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        # E(x; f) = -T * log( sum_i exp(f_i / T) )
        return -T * torch.logsumexp(logits / T, dim=1)

    def forward(self, x_tex: torch.Tensor, x_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 构造伪彩色增强通道 C：从纹理输入中取 S 通道 (假定顺序为 [S, t, f])
        # 将 C 与原始 [S, t, f] 堆叠，形成 6 通道送入纹理骨干
        s = x_tex[:, 0:1, :, :]
        c = self.pseudo_color(s)  # (B,3,H,W)
        x_tex = torch.cat([x_tex, c], dim=1)  # (B,6,H,W)
        # streams
        p3_tex, p4_tex, p5_tex = self.tex(x_tex)
        p3_pos, p4_pos, p5_pos = self.pos(x_pos)

        # neck fusion -> F3,F4,F5
        f3, f4, f5 = self.neck(p3_tex, p4_tex, p5_tex, p3_pos, p4_pos, p5_pos)

        # optional Swin on F5
        f5 = self.swin(f5)

        # head
        feat = self.pool(f5).flatten(1)
        logits = self.fc(feat)
        energy = self.energy(logits)
        return logits, energy


def build_model(num_classes: int, use_swin: bool = False, out_chs: Tuple[int, int, int] = (192, 384, 768), swin_heads: int = 8, swin_layers: int = 2, pos_backbone: str = 'dwconv', align_resize: bool = False) -> TwoStreamClassifier:
    return TwoStreamClassifier(num_classes=num_classes, out_chs=out_chs, use_swin=use_swin, swin_heads=swin_heads, swin_layers=swin_layers, pos_backbone=pos_backbone, align_resize=align_resize)


# ----------------------
# Two-stream Det Backbone (for Faster R-CNN)
# ----------------------
class TwoStreamDetBackbone(nn.Module):
    """两流(CSP/C2f 纹理流 + MobileNetV3-Large 位置流) + PANet(out [192,384,768]) + 可选 Swin(S)@P5。
    作为检测骨干(backbone)供 torchvision FasterRCNN 使用：
    - forward(x[B,3,H,W]) -> OrderedDict{"0": P3, "1": P4, "2": P5}，各层通道统一投影为 256。
    - 设置属性 out_channels=256 以便 RPN/ROI 头构建。
    - 提供 load_local_pretrained(...) 用本地权重初始化各子模块。
    """
    def __init__(self, out_chs: Tuple[int,int,int] = (192,384,768), use_swin: bool = True, align_resize: bool = False, pos_backbone: str = 'mobilenetv3'):
        super().__init__()
        self.use_swin = use_swin
        # 纹理流：输入6通道（[RGB] + 伪彩色C[3]，C由首通道生成）
        self.tex = TextureBackbone(in_c=6, base_c=64)
        # 位置流：默认 MobileNetV3-Large
        if pos_backbone == 'mobilenetv3':
            self.pos = PositionBackboneMobileNetV3()
        else:
            self.pos = PositionBackbone(in_c=3, base_c=32)
        self.pseudo_color = nn.Conv2d(1, 3, kernel_size=1, bias=True)
        # 颈部：PANet 融合
        self.neck = FusionPANet(self.tex.out_channels, self.pos.out_channels, out_chs=out_chs, align_resize=align_resize)
        # Swin stage4 仅作用于 P5
        if use_swin:
            if TimmSwinStage is not None and out_chs[2] in (768, 1024):
                self.swin = SwinStage4Adapter(dim=out_chs[2], depth=2, num_heads=24, window_size=7, drop_path=0.0, input_resolution=(8,8))
            else:
                self.swin = SwinOrTransformerLite(dim=out_chs[2], num_heads=8, num_layers=2)
        else:
            self.swin = nn.Identity()
        # 将三层统一投影为 256 通道，供 RPN/ROI 使用
        self.lateral3 = ConvBNAct(out_chs[0], 256, k=1, s=1, p=0)
        self.lateral4 = ConvBNAct(out_chs[1], 256, k=1, s=1, p=0)
        self.lateral5 = ConvBNAct(out_chs[2], 256, k=1, s=1, p=0)
        self.out_channels = 256

    def forward(self, x: torch.Tensor):
        # x: (B,3,H,W)
        s = x[:, 0:1, :, :]
        c = self.pseudo_color(s)  # (B,3,H,W)
        x_tex = torch.cat([x, c], dim=1)  # (B,6,H,W)
        p3_tex, p4_tex, p5_tex = self.tex(x_tex)
        p3_pos, p4_pos, p5_pos = self.pos(x)
        f3, f4, f5 = self.neck(p3_tex, p4_tex, p5_tex, p3_pos, p4_pos, p5_pos)
        f5 = self.swin(f5)
        o3 = self.lateral3(f3)
        o4 = self.lateral4(f4)
        o5 = self.lateral5(f5)
        from collections import OrderedDict
        return OrderedDict({"0": o3, "1": o4, "2": o5})

    @torch.no_grad()
    def load_local_pretrained(self, pos_weights: Optional[str] = None, swin_weights: Optional[str] = None,
                              texture_yolov8m_cls: Optional[str] = None, yolo_det: Optional[str] = None):
        # Position: 从本地 MobileNetV3-Large ckpt 加载
        if pos_weights and os.path.isfile(pos_weights) and tvm is not None:
            try:
                mdl = tvm.mobilenet_v3_large(weights=None)
                sd = torch.load(pos_weights, map_location='cpu', weights_only=True)
                if isinstance(sd, dict) and 'state_dict' in sd:
                    sd = sd['state_dict']
                mdl.load_state_dict(sd, strict=False)
                self.pos.m.load_state_dict(mdl.features.state_dict(), strict=False)
                print(f"Loaded MobileNetV3-Large features from {pos_weights}")
            except Exception as e:
                print(f"Warn: failed to load MobileNetV3 weights: {e}")
        # Swin: 从完整 SwinSmall 模型提取 stage4 权重，尽量匹配
        if self.use_swin and swin_weights and os.path.isfile(swin_weights) and timm is not None:
            try:
                full = timm.create_model('swin_small_patch4_window7_224', pretrained=False)
                sd = torch.load(swin_weights, map_location='cpu', weights_only=True)
                if isinstance(sd, dict) and 'state_dict' in sd:
                    sd = sd['state_dict']
                full.load_state_dict(sd, strict=False)
                if hasattr(self.swin, 'stage') and self.swin.stage is not None:
                    stage_sd = self.swin.stage.state_dict()
                    src = full.layers[3].state_dict() if hasattr(full, 'layers') else full.stages[3].state_dict()
                    for k, v in stage_sd.items():
                        if k in src and src[k].shape == v.shape:
                            stage_sd[k].copy_(src[k])
                    print(f"Loaded Swin-S stage4 weights from {swin_weights}")
            except Exception as e:
                print(f"Warn: failed to load Swin stage4 weights: {e}")
        # Texture: 尝试从 YOLOv8m 分类或检测权重迁移卷积参数（按形状顺序匹配，启发式）
        def _load_ultralytics_conv_only(path: str):
            try:
                sd = torch.load(path, map_location='cpu', weights_only=True)
                if isinstance(sd, dict):
                    if 'model' in sd and hasattr(sd['model'], 'state_dict'):
                        sds = sd['model'].state_dict()
                    elif 'state_dict' in sd:
                        sds = sd['state_dict']
                    else:
                        sds = sd
                else:
                    sds = sd
                # 收集源中的卷积权重（4D）按键排序
                src_convs = [(k, v) for k, v in sorted(sds.items()) if isinstance(v, torch.Tensor) and v.ndim == 4]
                # 收集目标纹理流中的 Conv 权重按照拓扑顺序
                tgt_convs = []
                for name, m in self.tex.named_modules():
                    if isinstance(m, nn.Conv2d):
                        tgt_convs.append((name, m))
                copied = 0
                si = 0
                for tn, tm in tgt_convs:
                    # 向前查找形状匹配的源卷积
                    while si < len(src_convs):
                        sk, sv = src_convs[si]
                        si += 1
                        if sv.shape == tm.weight.shape:
                            tm.weight.copy_(sv)
                            copied += 1
                            break
                print(f"Heuristically copied {copied} conv layers from {os.path.basename(path)} to TextureBackbone")
            except Exception as e:
                print(f"Warn: failed to map YOLOv8 convs from {path}: {e}")
        if texture_yolov8m_cls and os.path.isfile(texture_yolov8m_cls):
            _load_ultralytics_conv_only(texture_yolov8m_cls)
        elif yolo_det and os.path.isfile(yolo_det):
            _load_ultralytics_conv_only(yolo_det)


def build_two_stream_det_backbone(out_chs: Tuple[int,int,int]=(192,384,768), use_swin: bool=True, align_resize: bool=False, pos_backbone: str='mobilenetv3') -> TwoStreamDetBackbone:
    return TwoStreamDetBackbone(out_chs=out_chs, use_swin=use_swin, align_resize=align_resize, pos_backbone=pos_backbone)
