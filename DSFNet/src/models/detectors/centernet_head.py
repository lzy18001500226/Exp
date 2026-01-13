import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入 PF-Aware 模块
try:
    from ..modules.pfaware import PFAwareModule, LightweightPFAware
    _has_pfaware = True
except ImportError:
    _has_pfaware = False
    print("Warning: PFAware module not found. PF-aware mechanism will be disabled.")

class CenterHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        ch = max(64, in_ch // 4)
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        )
        self.head_hm = nn.Conv2d(ch, num_classes, 1)
        self.head_wh = nn.Conv2d(ch, 2, 1)
        self.head_off = nn.Conv2d(ch, 2, 1)
        # heatmap 初始偏置为负，避免一开始全为正样本
        nn.init.constant_(self.head_hm.bias, -2.19)
        # 输出可选上采样（默认为恒等）
        self.out_upsample = 1
        self.out_up = nn.Identity()

    def forward(self, x):
        # 强制 FP32，避免 AMP 与小 batch 的 BatchNorm 数值不稳定
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x = x.float()
            f = self.shared(x)
            f = self.out_up(f)
            hm = torch.sigmoid(self.head_hm(f))
            wh = self.head_wh(f)
            off = self.head_off(f)
        return {'hm': hm, 'wh': wh, 'off': off}


def _gaussian_radius(w, h, min_overlap=0.7):
    a1 = 1
    b1 = (h + w)
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(torch.clamp(b1 ** 2 - 4 * a1 * c1, min=0.0))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = torch.sqrt(torch.clamp(b2 ** 2 - 4 * a2 * c2, min=0.0))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = torch.sqrt(torch.clamp(b3 ** 2 - 4 * a3 * c3, min=0.0))
    r3 = (b3 + sq3) / (2 * a3 + 1e-6)
    return torch.min(torch.min(r1, r2), r3)


def _draw_gaussian(hm, center, radius):
    x, y = int(center[0]), int(center[1])
    radius = int(max(0, radius))
    height, width = hm.shape[-2:]
    diameter = 2 * radius + 1
    gaussian = torch.exp(-((torch.arange(diameter, device=hm.device).float() - radius) ** 2) / (2 * (radius / 3 + 1e-6) ** 2))
    gaussian2d = torch.ger(gaussian, gaussian)

    left, right = min(x, radius), min(width - x - 1, radius)
    top, bottom = min(y, radius), min(height - y - 1, radius)

    masked_hm = hm[..., y - top:y + bottom + 1, x - left:x + right + 1]
    masked_gauss = gaussian2d[radius - top:radius + bottom + 1, radius - left:radius + right + 1]
    torch.maximum(masked_hm, masked_gauss, out=masked_hm)


class CenterLoss(nn.Module):
    """多尺度 CenterNet Loss
    
    改进:
    1. 支持 P2/P3/P4 三尺度联合训练
    2. wh 使用 sigmoid 归一化 + log-L1 loss (稳定小目标梯度)
    3. 自动将 GT 分配到合适的尺度 (小目标→P2, 中目标→P3, 大目标→P4)
    """
    def __init__(self, num_classes: int, hm_weight: float = 1.0, wh_weight: float = 0.3, off_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.off_weight = off_weight
        # ✅ 多尺度高斯半径 (P2 更小适应FCS, P4 更大适应VTS)
        self.min_radius_p2 = 1.5  # FCS 小目标友好 (6-8px)
        self.min_radius_p3 = 2.5
        self.min_radius_p4 = 3.5
        # 当一个 batch 内没有正样本时，降低纯负样本图的负项权重
        self.neg_only_factor = 0.25
        # ✅ 尺度分配阈值 (根据512尺度,更精细)
        self.scale_thresh_small = 24   # <24px → P2 (FCS 小目标)
        self.scale_thresh_medium = 96  # <96px → P3 (中等目标)
        # >96px → P4 (VTS 大目标)

    @staticmethod
    def _focal_core(pred, gt, alpha=2.0, beta=4.0):
        # 稳定化：清理 NaN/Inf，并 clamp 防止 log(0)
        pred = torch.nan_to_num(pred, nan=0.5, posinf=0.999, neginf=0.001)
        gt = torch.nan_to_num(gt, nan=0.0, posinf=1.0, neginf=0.0)
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        pos_loss = -torch.log(pred + 1e-8) * torch.pow(1 - pred, alpha) * pos_inds
        neg_loss = -torch.log(1 - pred + 1e-8) * torch.pow(pred, alpha) * torch.pow(1 - gt, beta) * neg_inds
        pos_sum = torch.nan_to_num(pos_loss.sum(), nan=0.0)
        neg_sum = torch.nan_to_num(neg_loss.sum(), nan=0.0)
        pos_cnt = torch.clamp(pos_inds.sum(), min=0.0)
        return pos_sum, neg_sum, pos_cnt

    def focal_loss(self, pred, gt, alpha=2.0, beta=4.0):
        pos_sum, neg_sum, pos_cnt = self._focal_core(pred, gt, alpha, beta)
        # 无正样本时，降低负项权重，避免背景主导
        if float(pos_cnt.item()) < 1.0:
            neg_sum = neg_sum * self.neg_only_factor
        denom = torch.clamp(pos_cnt, min=1.0)
        return (pos_sum + neg_sum) / denom
    
    def _assign_to_scale(self, box_w_px, box_h_px):
        """根据框的像素尺寸分配到合适的尺度
        
        策略优化:
        1. 使用短边判断 (避免细长目标被误判到大尺度)
        2. 更精细的阈值 (24/96 vs 30/80)
        3. FCS小目标(<24px) → P2, 中目标(<96px) → P3, VTS大目标(≥96px) → P4
        """
        # ✅ 使用短边或面积根号,避免细长条被误判到P4
        min_side = min(box_w_px, box_h_px)
        area_sqrt = (box_w_px * box_h_px) ** 0.5
        
        # 优先看短边,兼顾面积
        if min_side < self.scale_thresh_small or area_sqrt < self.scale_thresh_small:
            return 'p2'
        elif min_side < self.scale_thresh_medium * 0.6 or area_sqrt < self.scale_thresh_medium:
            return 'p3'
        else:
            return 'p4'

    def forward(self, preds, targets):
        """
        preds: {'p2': {'hm', 'wh', 'off', 'stride'}, 'p3': {...}, 'p4': {...}}
        targets: list of [B] tensors, each [N, 6] = [cls, cx_norm, cy_norm, w_norm, h_norm, conf]
        """
        # ✅ 修复: 直接从预测张量获取设备,避免targets为空或CPU张量导致错误
        device = preds['p2']['hm'].device
        B = len(targets)
        
        # ===== 初始化三尺度的 GT =====
        scales = ['p2', 'p3', 'p4']
        gt_dict = {}
        for scale in scales:
            pred_scale = preds[scale]
            _, C, H, W = pred_scale['hm'].shape
            gt_dict[scale] = {
                'hm': torch.zeros((B, C, H, W), device=device),
                'wh': torch.zeros((B, 2, H, W), device=device),
                'off': torch.zeros((B, 2, H, W), device=device),
                'mask_wh': torch.zeros((B, 1, H, W), device=device),
                'mask_off': torch.zeros((B, 1, H, W), device=device),
            }
        
        # ===== 构造 GT (自动分配到合适尺度) =====
        for b in range(B):
            t = targets[b]
            if t.numel() == 0:
                continue
            
            for gt in t:
                gcls = int(gt[0].item())
                if not (0 <= gcls < self.num_classes):
                    continue
                
                # targets: [cls, cx_norm, cy_norm, w_norm, h_norm, conf]
                cx_norm, cy_norm = float(gt[1].item()), float(gt[2].item())
                w_norm, h_norm = float(gt[3].item()), float(gt[4].item())
                
                # 转为512像素坐标判断尺度
                w_px_512 = w_norm * 512
                h_px_512 = h_norm * 512
                
                # 分配到合适尺度
                assigned_scale = self._assign_to_scale(w_px_512, h_px_512)
                
                # 获取该尺度的特征图尺寸
                _, C, H, W = preds[assigned_scale]['hm'].shape
                
                # 转为特征图坐标
                cx, cy = cx_norm * W, cy_norm * H
                gw, gh = w_norm * W, h_norm * H
                
                if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(gw) and math.isfinite(gh)):
                    continue
                
                gi = int(torch.clamp(torch.tensor(cx, device=device), 0, W - 1).item())
                gj = int(torch.clamp(torch.tensor(cy, device=device), 0, H - 1).item())
                
                # ✅ 高斯半径: 根据尺度调整最小值,同时让半径与真实框比例相关
                if assigned_scale == 'p2':
                    min_r = self.min_radius_p2
                elif assigned_scale == 'p3':
                    min_r = self.min_radius_p3
                else:
                    min_r = self.min_radius_p4
                
                radius = _gaussian_radius(torch.tensor(gw, device=device), torch.tensor(gh, device=device))
                if torch.isfinite(radius):
                    # ✅ 让半径与框尺寸挂钩,避免极小框用固定大半径
                    adaptive_min = max(min_r, 0.4 * min(gw, gh))
                    radius = torch.clamp(radius, min=adaptive_min)
                else:
                    radius = torch.tensor(min_r, device=device)
                
                _draw_gaussian(gt_dict[assigned_scale]['hm'][b, gcls], (gi, gj), float(radius.item()))
                
                # ✅ 直接赋值避免频繁构造tensor (性能优化)
                gt_dict[assigned_scale]['wh'][b, 0, gj, gi] = max(0.0, min(1.0, w_norm))
                gt_dict[assigned_scale]['wh'][b, 1, gj, gi] = max(0.0, min(1.0, h_norm))
                gt_dict[assigned_scale]['off'][b, 0, gj, gi] = cx - gi
                gt_dict[assigned_scale]['off'][b, 1, gj, gi] = cy - gj
                gt_dict[assigned_scale]['mask_wh'][b, 0, gj, gi] = 1.0
                gt_dict[assigned_scale]['mask_off'][b, 0, gj, gi] = 1.0
        
        # ===== 计算三尺度的 Loss =====
        # use tensor on device to accumulate losses
        total_loss = torch.tensor(0.0, device=device)
        loss_parts = {}
        eps = 1e-6
        # per-scale loss cap to avoid single-scale exploding gradients
        loss_cap = 1e3

        for scale in scales:
            pred_scale = preds[scale]
            gt_scale = gt_dict[scale]

            hm_pred = torch.nan_to_num(pred_scale['hm'], nan=0.0, posinf=1.0, neginf=0.0)
            wh_pred_raw = torch.nan_to_num(pred_scale['wh'], nan=0.0, posinf=1.0, neginf=0.0)
            off_pred = torch.nan_to_num(pred_scale['off'], nan=0.0, posinf=1.0, neginf=-1.0)

            # 🔥 改进1: WH在loss计算前强制约束 (防止训练时回归失控)
            # 根据尺度设定物理上限 (相对512尺度的归一化值)
            if scale == 'p2':
                max_wh_norm = 96.0 / 512.0  # FCS小目标上限
            elif scale == 'p3':
                max_wh_norm = 256.0 / 512.0
            else:  # p4
                max_wh_norm = 512.0 / 512.0  # VTS大目标上限
            
            # 强制clamp,梯度会被截断,防止无界回归
            wh_pred = torch.clamp(wh_pred_raw, eps, max_wh_norm)

            loss_hm = self.focal_loss(hm_pred, gt_scale['hm'])

            # ✅ log-L1 loss for wh (稳定小目标梯度,简化重复eps) + NaN保护
            wh_pred_clamped = torch.clamp(wh_pred, eps, max_wh_norm)
            wh_gt_clamped = torch.clamp(gt_scale['wh'], eps, 1.0)
            log_pred = torch.nan_to_num(torch.log(wh_pred_clamped + eps), nan=0.0, posinf=0.0, neginf=-10.0)
            log_gt = torch.nan_to_num(torch.log(wh_gt_clamped + eps), nan=0.0, posinf=0.0, neginf=-10.0)
            loss_wh = F.l1_loss(
                log_pred * gt_scale['mask_wh'],
                log_gt * gt_scale['mask_wh'],
                reduction='sum'
            ) / torch.clamp(gt_scale['mask_wh'].sum(), min=1.0)

            loss_off = F.l1_loss(
                off_pred * gt_scale['mask_off'],
                gt_scale['off'] * gt_scale['mask_off'],
                reduction='sum'
            ) / torch.clamp(gt_scale['mask_off'].sum(), min=1.0)

            # 检查有限性
            if not torch.isfinite(loss_hm):
                loss_hm = torch.tensor(0.0, device=device)
            if not torch.isfinite(loss_wh):
                loss_wh = torch.tensor(0.0, device=device)
            if not torch.isfinite(loss_off):
                loss_off = torch.tensor(0.0, device=device)

            # 🔥 改进3: Per-scale自适应权重 (考虑样本难度和分布)
            num_pos = float(gt_scale['mask_wh'].sum().item())
            
            # 计算全局正样本数 (用于自适应权重)
            total_pos = sum(float(gt_dict[s]['mask_wh'].sum().item()) for s in scales)
            
            # 每尺度损失先除以num_pos,再乘权重,避免样本数不平衡主导loss
            if num_pos > 0:
                scale_loss_normalized = (self.hm_weight * loss_hm + self.wh_weight * loss_wh + self.off_weight * loss_off) / (num_pos + eps)
            else:
                # 无正样本时只保留hm的负项损失,且降权
                scale_loss_normalized = self.hm_weight * loss_hm * 0.1
            
            # 🔥 自适应尺度权重: 样本少的尺度给更高权重 (平方根衰减)
            if total_pos > 0:
                # 比例越小,权重越高 (用sqrt平滑)
                ratio = num_pos / (total_pos + eps)
                adaptive_weight = (1.0 / (ratio + 0.1)) ** 0.5  # sqrt衰减
                adaptive_weight = torch.clamp(torch.tensor(adaptive_weight, device=device), 0.5, 3.0)
            else:
                adaptive_weight = torch.tensor(1.0, device=device)
            
            scale_loss = adaptive_weight * scale_loss_normalized
            # cap per-scale loss safely for both tensors and python floats
            if isinstance(scale_loss, torch.Tensor):
                scale_loss = torch.clamp(scale_loss, max=loss_cap)
            else:
                try:
                    sval = float(scale_loss)
                except Exception:
                    sval = 0.0
                if sval > loss_cap:
                    scale_loss = torch.tensor(loss_cap, device=device)
                else:
                    scale_loss = torch.tensor(sval, device=device)
            total_loss = total_loss + scale_loss

            loss_parts[f'loss_hm_{scale}'] = float(loss_hm.detach())
            loss_parts[f'loss_wh_{scale}'] = float(loss_wh.detach())
            loss_parts[f'loss_off_{scale}'] = float(loss_off.detach())
            loss_parts[f'num_pos_{scale}'] = num_pos  # 诊断用

        # 汇总分量 (向后兼容)
        loss_parts['loss_hm'] = sum(loss_parts[f'loss_hm_{s}'] for s in scales) / len(scales)
        loss_parts['loss_wh'] = sum(loss_parts[f'loss_wh_{s}'] for s in scales) / len(scales)
        loss_parts['loss_off'] = sum(loss_parts[f'loss_off_{s}'] for s in scales) / len(scales)

        # final safety clamp
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=loss_cap * len(scales), neginf=-loss_cap * len(scales))
        return total_loss, loss_parts


class ImprovedCenterHead(nn.Module):
    """多尺度 CenterNet 头（P2/P3/P4 三尺度独立预测）+ PF-Aware 位置感知机制。
    
    架构改进:
    1. 增加 P2 (stride=4) 高分辨率输出,覆盖 6-25px 小目标
    2. 每个尺度独立预测 hm/wh/off,避免单一输出的感知盲区
    3. 轻量 FPN 融合 (P4→P3→P2 自顶向下 + 横向连接)
    4. wh 输出 sigmoid 归一化到 [0,1],配合 log-L1 loss
    """
    def __init__(self, in_ch_p2: int, in_ch_p3: int, in_ch_p4: int, num_classes: int, out_upsample: int = 1, use_pfaware: bool = False, pfaware_lite: bool = False):
        super().__init__()
        self.num_classes = int(num_classes)
        self.out_upsample = int(max(1, out_upsample))
        self.use_pfaware = bool(use_pfaware) and _has_pfaware
        
        # 🔥 改进2: PF-Aware渐进式预热门控 (初始为0,训练中逐步开启)
        self.pf_warmup_gate = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
        # 🔥 改进4: Backbone输出预处理 (LayerNorm稳定特征分布)
        self.p2_prenorm = nn.LayerNorm([in_ch_p2])  # 会在forward中reshape
        self.p3_prenorm = nn.LayerNorm([in_ch_p3])
        self.p4_prenorm = nn.LayerNorm([in_ch_p4])
        
        # ===== 轻量 FPN: 自顶向下融合 =====
        # P4 → P3
        self.p4_lateral = nn.Conv2d(in_ch_p4, 256, 1)
        self.p4_fuse = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.p3_lateral = nn.Conv2d(in_ch_p3, 256, 1)
        self.p3_fuse = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # P3 → P2
        self.p2_lateral = nn.Conv2d(in_ch_p2, 128, 1)
        self.p2_fuse = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # ===== PF-Aware 模块 (应用在 P3 融合后的特征上) =====
        if self.use_pfaware:
            if pfaware_lite:
                self.pfaware = LightweightPFAware(in_channels=256, downsample_ratio=4, reduction=8)
                print("✅ Using Lightweight PF-Aware Module on P3")
            else:
                self.pfaware = PFAwareModule(in_channels=256, reduction=8, num_heads=4)
                print("✅ Using Standard PF-Aware Module on P3")
        else:
            self.pfaware = nn.Identity()
        
        # ===== 三尺度独立预测头 =====
        # P2 head (stride=4, 关注小目标 6-25px)
        self.p2_head = self._make_head(128, num_classes, name='P2')
        
        # P3 head (stride=8, 关注中目标 20-60px)
        self.p3_head = self._make_head(256, num_classes, name='P3')
        
        # P4 head (stride=16, 关注大目标 60+px)
        self.p4_head = self._make_head(256, num_classes, name='P4')
        
        # ✅ 输出上采样 (默认 identity,可选 2x/4x)
        self.out_up = nn.Identity() if self.out_upsample == 1 else nn.Upsample(scale_factor=self.out_upsample, mode='bilinear', align_corners=False)
    
    def _make_head(self, in_ch: int, num_classes: int, name: str = ''):
        """构建单个尺度的预测头"""
        shared = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch//2, 3, padding=1), nn.BatchNorm2d(in_ch//2), nn.ReLU(inplace=True),
        )
        head_hm = nn.Conv2d(in_ch//2, num_classes, 1)
        head_wh = nn.Conv2d(in_ch//2, 2, 1)
        head_off = nn.Conv2d(in_ch//2, 2, 1)
        
        # heatmap 初始化为负偏置
        nn.init.constant_(head_hm.bias, -2.19)
        
        return nn.ModuleDict({
            'shared': shared,
            'hm': head_hm,
            'wh': head_wh,
            'off': head_off,
        })
    
    def _forward_head(self, x, head_dict):
        """单个尺度的前向传播 + 多层NaN保护"""
        feat = head_dict['shared'](x)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
        
        hm = torch.sigmoid(head_dict['hm'](feat))
        hm = torch.nan_to_num(hm, nan=0.0, posinf=1.0, neginf=0.0)
        hm = torch.clamp(hm, 0.0, 1.0)
        
        wh = torch.sigmoid(head_dict['wh'](feat))
        wh = torch.nan_to_num(wh, nan=0.5, posinf=1.0, neginf=0.0)
        wh = torch.clamp(wh, 0.0, 1.0)
        
        off = torch.tanh(head_dict['off'](feat))
        off = torch.nan_to_num(off, nan=0.0, posinf=1.0, neginf=-1.0)
        off = torch.clamp(off, -1.0, 1.0)
        
        return hm, wh, off

    def forward(self, p2, p3, p4):
        # 强制 FP32 避免 AMP + 小 batch + BN 的不稳定
        with torch.amp.autocast(device_type='cuda', enabled=False):
            p2, p3, p4 = p2.float(), p3.float(), p4.float()
            
            # 🔥 改进4: Backbone输出LayerNorm预处理 (稳定分布)
            B, C2, H2, W2 = p2.shape
            _, C3, H3, W3 = p3.shape
            _, C4, H4, W4 = p4.shape
            
            # LayerNorm: (B, C, H, W) → (B, H, W, C) → norm → (B, C, H, W)
            p2 = p2.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
            p2 = self.p2_prenorm(p2)
            p2 = p2.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            
            p3 = p3.permute(0, 2, 3, 1).contiguous()
            p3 = self.p3_prenorm(p3)
            p3 = p3.permute(0, 3, 1, 2).contiguous()
            
            p4 = p4.permute(0, 2, 3, 1).contiguous()
            p4 = self.p4_prenorm(p4)
            p4 = p4.permute(0, 3, 1, 2).contiguous()
            
            # ✅ 输入保护: 防止backbone输出异常 (LayerNorm后更严格的clamp)
            p2 = torch.nan_to_num(p2, nan=0.0, posinf=3.0, neginf=-3.0)
            p3 = torch.nan_to_num(p3, nan=0.0, posinf=3.0, neginf=-3.0)
            p4 = torch.nan_to_num(p4, nan=0.0, posinf=3.0, neginf=-3.0)
            p2 = torch.clamp(p2, -5.0, 5.0)
            p3 = torch.clamp(p3, -5.0, 5.0)
            p4 = torch.clamp(p4, -5.0, 5.0)
            
            # ===== FPN 融合 =====
            # P4 lateral + fuse
            p4_lat = self.p4_lateral(p4)
            p4_lat = torch.nan_to_num(p4_lat, nan=0.0, posinf=1.0, neginf=-1.0)
            p4_fused = self.p4_fuse(p4_lat)
            p4_fused = torch.nan_to_num(p4_fused, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # P4 → P3 融合
            p4_up = F.interpolate(p4_fused, size=p3.shape[-2:], mode='bilinear', align_corners=False)
            p4_up = torch.nan_to_num(p4_up, nan=0.0, posinf=1.0, neginf=-1.0)
            p3_lat = self.p3_lateral(p3)
            p3_lat = torch.nan_to_num(p3_lat, nan=0.0, posinf=1.0, neginf=-1.0)
            p3_fused = self.p3_fuse(p4_up + p3_lat)
            p3_fused = torch.nan_to_num(p3_fused, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # PF-Aware 应用在 P3 (最具代表性的中间尺度) + 🔥 渐进式门控
            if self.use_pfaware:
                p3_raw = p3_fused
                p3_pfaware = self.pfaware(p3_fused)
                p3_pfaware = torch.nan_to_num(p3_pfaware, nan=0.0, posinf=1.0, neginf=-1.0)
                # 门控混合: p3 = (1-gate)*raw + gate*pfaware
                gate = torch.clamp(self.pf_warmup_gate, 0.0, 1.0)
                p3_fused = (1.0 - gate) * p3_raw + gate * p3_pfaware
            p3_fused = torch.nan_to_num(p3_fused, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # P3 → P2 融合
            p3_up = F.interpolate(p3_fused, size=p2.shape[-2:], mode='bilinear', align_corners=False)
            p3_up = torch.nan_to_num(p3_up, nan=0.0, posinf=1.0, neginf=-1.0)
            p2_lat = self.p2_lateral(p2)
            p2_lat = torch.nan_to_num(p2_lat, nan=0.0, posinf=1.0, neginf=-1.0)
            p2_fused = self.p2_fuse(torch.cat([p3_up, p2_lat], dim=1))
            p2_fused = torch.nan_to_num(p2_fused, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # ===== 三尺度独立预测 =====
            hm_p2, wh_p2, off_p2 = self._forward_head(p2_fused, self.p2_head)
            hm_p3, wh_p3, off_p3 = self._forward_head(p3_fused, self.p3_head)
            hm_p4, wh_p4, off_p4 = self._forward_head(p4_fused, self.p4_head)
            
            # ✅ 可选上采样 + stride 元数据同步更新
            stride_p2, stride_p3, stride_p4 = 4, 8, 16
            if self.out_upsample > 1:
                hm_p2 = self.out_up(hm_p2)
                wh_p2 = self.out_up(wh_p2)
                off_p2 = self.out_up(off_p2)
                hm_p3 = self.out_up(hm_p3)
                wh_p3 = self.out_up(wh_p3)
                off_p3 = self.out_up(off_p3)
                hm_p4 = self.out_up(hm_p4)
                wh_p4 = self.out_up(wh_p4)
                off_p4 = self.out_up(off_p4)
                # ✅ stride 同步除以上采样倍数
                stride_p2 = stride_p2 // self.out_upsample
                stride_p3 = stride_p3 // self.out_upsample
                stride_p4 = stride_p4 // self.out_upsample
            
            return {
                'p2': {'hm': hm_p2, 'wh': wh_p2, 'off': off_p2, 'stride': stride_p2},
                'p3': {'hm': hm_p3, 'wh': wh_p3, 'off': off_p3, 'stride': stride_p3},
                'p4': {'hm': hm_p4, 'wh': wh_p4, 'off': off_p4, 'stride': stride_p4},
            }

