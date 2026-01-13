import argparse
import json
from pathlib import Path
import sys
import time
import csv
import yaml
import warnings
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import numpy as np

# 静默 NumPy 空集警告（验证早期可能触发）
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')

# 可选绘图
try:
    import matplotlib.pyplot as plt
    _has_plt = True
except Exception:
    _has_plt = False

# 进度条
try:
    from tqdm.auto import tqdm
except Exception:
    # 安全降级：无 tqdm 时使用原生迭代器
    tqdm = None

# 路径
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
WORKSPACE_ROOT = ROOT_DIR.parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.detectors.backbone_swin import SwinBackbone
from models.detectors.backbone_convnext import ConvNeXtBackbone
from models.detectors.centernet_head import ImprovedCenterHead, CenterLoss
# 动态导入数据集，避免静态分析误报
import importlib as _importlib
try:
    _det_mod = _importlib.import_module('data.dataset_fcs_vts')
except Exception:
    _det_mod = _importlib.import_module('dataset_fcs_vts')
FCSVTSDetDataset = getattr(_det_mod, 'FCSVTSDetDataset')
det_collate_fn = getattr(_det_mod, 'det_collate_fn')


def set_seed(seed: int = 3407):
    try:
        import random as _random, numpy as _np, torch as _torch
        _random.seed(seed); _np.random.seed(seed); _torch.manual_seed(seed)
        if _torch.cuda.is_available(): _torch.cuda.manual_seed_all(seed)
    except Exception: pass


def parse_args():
    p = argparse.ArgumentParser(description='B0/B1 频谱检测（CenterNet 多尺度，SMNet→ImageNet 预处理）')
    p.add_argument('--backbone_type', type=str, default='swin', choices=['swin', 'convnext'])
    p.add_argument('--train_list', type=str, required=True)
    p.add_argument('--val_list', type=str, default=None)
    p.add_argument('--run_dir', type=str, default='')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_classes', type=int, default=22)
    p.add_argument('--num_anchors', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--save_interval', type=int, default=1)
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--checkpoint', type=str, default='')
    # 检测专用
    p.add_argument('--exclude_classes', type=str, default='13,14')
    p.add_argument('--conf_thresh', type=float, default=0.05, help='Confidence threshold for validation metrics (lower = more detections for mAP)')
    p.add_argument('--head_out_up', type=int, default=3, help='Output upsample factor for detection head (RECOMMENDED: 3 to match multi-scale decode; 1=no upsample, 2/4 for higher resolution)')
    # 解码/评估控制
    p.add_argument('--decode_scales', type=str, default='all', help='Which scales to decode: "all" or a comma list like "p3" / "p2,p3"')
    p.add_argument('--disable_curriculum', action='store_true', help='Disable curriculum-based decoding tightening; use static thresholds below')
    p.add_argument('--decode_conf', type=float, default=None, help='Static decode confidence threshold (overrides curriculum when set)')
    p.add_argument('--decode_topk', type=int, default=None, help='Static top-k per class per scale (overrides curriculum when set)')
    p.add_argument('--decode_min_wh', type=float, default=None, help='Static minimum box width/height in pixels (overrides curriculum when set)')
    p.add_argument('--decode_nms', type=float, default=0.5, help='NMS IoU threshold in decode')
    # 检测数据增强总开关（默认启用；设置 --det_no_aug 关闭）
    p.add_argument('--det_no_aug', action='store_true', help='Disable detection data augmentations (Albumentations) for a stable baseline')
    
    # 📋 模块2: Per-Scale WH上限约束
    p.add_argument('--max_wh_p2', type=float, default=96.0, help='Max box width/height (pixels) for P2 scale (FCS-friendly, default 96)')
    p.add_argument('--max_wh_p3', type=float, default=256.0, help='Max box width/height (pixels) for P3 scale (default 256)')
    p.add_argument('--max_wh_p4', type=float, default=512.0, help='Max box width/height (pixels) for P4 scale (VTS large objects, default 512)')
    
    # 📋 模块3: 梯度/Loss监控
    p.add_argument('--enable_grad_monitor', action='store_true', help='Enable gradient norm monitoring and save stats to run_dir/grad_stats.json')
    # CenterLoss 权重
    p.add_argument('--loss_hm', type=float, default=1.0)
    p.add_argument('--loss_wh', type=float, default=0.3)
    p.add_argument('--loss_off', type=float, default=1.0)
    
    # 📋 模块4: 可训练PosEncScale
    p.add_argument('--trainable_posenc', action='store_true', help='Make PosEncScale a trainable parameter (auto clamp to [0,1])')
    # SMNet 三通道预处理与位置编码
    p.add_argument('--smnet_preproc', action='store_true', help='Use [S,S_edge,S_corner] channels with optional positional encoding')
    # 边缘/角点与归一化：可复现实验的参数化
    p.add_argument('--edge_high', type=float, default=None, help='High threshold for Sobel magnitude (absolute in [0,1]; None=use percentile)')
    p.add_argument('--edge_low', type=float, default=None, help='Low threshold for Sobel magnitude (absolute; default=0.5*high when pct not provided)')
    p.add_argument('--edge_high_pct', type=float, default=None, help='Percentile for Sobel magnitude high threshold (e.g., 90 for 90th percentile)')
    p.add_argument('--edge_low_pct', type=float, default=None, help='Percentile for Sobel magnitude low threshold (e.g., 50)')
    p.add_argument('--corner_high', type=float, default=None, help='Absolute high threshold for Harris response after [0,1] normalization')
    p.add_argument('--corner_low', type=float, default=None, help='Absolute low threshold for Harris response after [0,1] normalization')
    p.add_argument('--corner_high_pct', type=float, default=None, help='Percentile for Harris high threshold (e.g., 90)')
    p.add_argument('--corner_low_pct', type=float, default=None, help='Percentile for Harris low threshold (e.g., 50)')
    p.add_argument('--harris_k', type=float, default=0.04, help='Harris corner constant k in R=det(M)-k*(trace(M))^2')
    p.add_argument('--harris_gamma', type=float, default=None, help='Deprecated: previous gamma-like parameter (ignored if harris_k provided)')
    p.add_argument('--spec_norm_mode', type=str, default='fixed-128', choices=['fixed-128','sample-minmax'], help='Spec normalization: fixed-128 (recommended) or sample-minmax')
    p.add_argument('--posenc_scale', type=float, default=0.0, help='Amplitude of sinusoidal 2D positional encoding added to S channel only (0=off)')
    
    # PF-Aware 位置感知模块 (SMNet 论文核心创新)
    p.add_argument('--use_pfaware', action='store_true', help='Enable PF-Aware module for time-frequency position feature extraction')
    p.add_argument('--pfaware_lite', action='store_true', help='Use lightweight PF-Aware module (reduces computation)')
    p.add_argument('--pfaware_gate_target', type=float, default=1.0, help='Final gate value for PF-Aware warmup (0..1, decoupled from posenc_scale)')
    
    # 严格验证开关（仅保留有≥1 有效框的样本用于 mAP）
    p.add_argument('--strict_val', action='store_true', help='Use strict validation set that only contains images with at least one valid GT box')

    # 数据子集选择（按文件名自然数前缀过滤，例如 0-100 做冒烟验证）
    p.add_argument('--stem_min', type=int, default=None, help='Filter dataset items whose filename natural-number prefix >= stem_min')
    p.add_argument('--stem_max', type=int, default=None, help='Filter dataset items whose filename natural-number prefix <= stem_max')

    # 早停
    p.add_argument('--early_stop', action='store_true')
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--min_delta', type=float, default=1e-4)

    args = p.parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return args


def build_model(args):
    if args.backbone_type == 'swin':
        backbone = SwinBackbone(pretrained=args.pretrained, checkpoint=args.checkpoint, img_size=512)
    else:
        backbone = ConvNeXtBackbone(pretrained=args.pretrained, checkpoint=args.checkpoint, img_size=512)
    class Model(torch.nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.bb = bb
            # ✅ 适配新的三尺度架构: P2(128ch) + P3(256ch) + P4(512ch)
            self.head = ImprovedCenterHead(
                in_ch_p2=128,
                in_ch_p3=256, 
                in_ch_p4=512, 
                num_classes=args.num_classes, 
                out_upsample=int(max(1, args.head_out_up)),
                use_pfaware=args.use_pfaware,
                pfaware_lite=args.pfaware_lite
            )
            # 📋 模块4: 可训练PosEncScale
            if args.trainable_posenc and args.posenc_scale > 0:
                self.posenc_gain = nn.Parameter(torch.tensor(args.posenc_scale))
                print(f"✅ PosEncScale is trainable: init={args.posenc_scale:.3f}")
            else:
                self.posenc_gain = None
        def forward(self, x):
            p2, p3, p4 = self.bb(x)  # ✅ 现在返回三个特征层
            return self.head(p2, p3, p4)
    model = Model(backbone)
    
    # 📋 模块1: 验证HeadOutUp与decode一致性
    expected_strides = {1: [4,8,16], 2: [2,4,8], 3: [1,2,4], 4: [1,1,2]}
    out_up = int(max(1, args.head_out_up))
    if out_up in expected_strides:
        print(f"✅ HeadOutUp={out_up} → Expected output strides: P2={expected_strides[out_up][0]}, P3={expected_strides[out_up][1]}, P4={expected_strides[out_up][2]}")
    else:
        print(f"⚠️  HeadOutUp={out_up} is non-standard; decode may need adjustment")
    
    return model


def main():
    args = parse_args()

    # 数据（列表应为 FCSLabel/VTSLabel 下 JSON 路径的 txt）
    # 解析排除类别
    exc = tuple(int(s) for s in str(args.exclude_classes).replace(';', ',').split(',') if s.strip())
    train_ds = FCSVTSDetDataset(
        args.train_list, img_size=512, augment=(not args.det_no_aug), exclude_classes=exc,
        smnet_preproc=args.smnet_preproc,
        edge_high=args.edge_high, edge_low=args.edge_low,
        edge_high_pct=args.edge_high_pct, edge_low_pct=args.edge_low_pct,
        corner_high=args.corner_high, corner_low=args.corner_low,
        corner_high_pct=args.corner_high_pct, corner_low_pct=args.corner_low_pct,
        harris_k=(args.harris_k if args.harris_k is not None else (args.harris_gamma if args.harris_gamma is not None else 0.04)),
        posenc_scale=args.posenc_scale,
        spec_norm_mode=args.spec_norm_mode,
        strict_only=False,
        stem_min=args.stem_min,
        stem_max=args.stem_max
    )
    val_ds = FCSVTSDetDataset(
        args.val_list, img_size=512, augment=False, exclude_classes=exc,
        smnet_preproc=args.smnet_preproc,
        edge_high=args.edge_high, edge_low=args.edge_low,
        edge_high_pct=args.edge_high_pct, edge_low_pct=args.edge_low_pct,
        corner_high=args.corner_high, corner_low=args.corner_low,
        corner_high_pct=args.corner_high_pct, corner_low_pct=args.corner_low_pct,
        harris_k=(args.harris_k if args.harris_k is not None else (args.harris_gamma if args.harris_gamma is not None else 0.04)),
        posenc_scale=args.posenc_scale,
        spec_norm_mode=args.spec_norm_mode,
        strict_only=bool(args.strict_val),
        stem_min=args.stem_min,
        stem_max=args.stem_max
    ) if args.val_list else None

    # ===== 标注覆盖率自检（验证集） =====
    try:
        if args.val_list and Path(args.val_list).is_file():
            json_from_npz = getattr(_det_mod, '_npz_to_label_json_path', None)
            boxes_from_json = getattr(_det_mod, '_json_to_boxes_norm', None)
            if boxes_from_json is not None:
                gt_cnt = [0] * int(args.num_classes)
                total_files, missing_json = 0, 0
                empty_json = 0
                missing_samples = []  # 缺失 JSON 的示例（前10条）
                empty_samples = []    # JSON 存在但无 shapes 的示例（前10条）
                map_preview = []      # NPZ→JSON 映射预览（前3对）
                with open(args.val_list, 'r', encoding='utf-8') as f:
                    for line in f:
                        p = line.strip()
                        if not p:
                            continue
                        total_files += 1
                        jp = None
                        if p.lower().endswith('.npz'):
                            if json_from_npz is not None:
                                jp = json_from_npz(p)
                                if len(map_preview) < 3:
                                    map_preview.append((p, jp))
                        else:
                            jp = p
                        if not jp or not Path(jp).is_file():
                            missing_json += 1
                            if len(missing_samples) < 10:
                                missing_samples.append(jp if jp else f"<no-map> from {p}")
                            continue
                        try:
                            boxes = boxes_from_json(jp, 512, 512)
                        except Exception:
                            boxes = []
                        if not boxes:
                            empty_json += 1
                            if len(empty_samples) < 10:
                                empty_samples.append(jp)
                            # 为空则不会增加任何类的 GT 计数
                            continue
                        for b in boxes:
                            c = int(b[0])
                            if c in exc:
                                continue
                            mapped = c - 2 if c > 14 else c
                            if 0 <= mapped < int(args.num_classes):
                                gt_cnt[mapped] += 1
                nonzero = sum(1 for v in gt_cnt if v > 0)
                total_gt = sum(gt_cnt)
                print(f"[COVERAGE] Val files={total_files}, missing_json={missing_json}, empty_json={empty_json}, total_gt={total_gt}, classes_with_gt={nonzero}/{int(args.num_classes)}")
                print(f"[COVERAGE] min_box_px=2 (json解析阈值,极限优化:8px→4px→2px)")
                # 映射预览
                if map_preview:
                    for a, b in map_preview:
                        print(f"[COVERAGE] Map preview: {a} -> {b}")
                # 打印示例：缺失/空标注
                if missing_samples:
                    print("[COVERAGE] Missing JSON samples (first 10):")
                    for s in missing_samples:
                        print(f"  - {s}")
                if empty_samples:
                    print("[COVERAGE] Empty JSON (no shapes) samples (first 10):")
                    for s in empty_samples:
                        print(f"  - {s}")
                # 可选：打印前若干类的 GT 数
                nz_pairs = [(i, v) for i, v in enumerate(gt_cnt) if v > 0]
                nz_pairs.sort(key=lambda x: -x[1])
                head = ', '.join([f"c{i}:{v}" for i, v in nz_pairs[:10]])
                if head:
                    print(f"[COVERAGE] Top classes: {head}")
    except Exception as _e:
        print(f"[COVERAGE] coverage check skipped: {_e}")
    # ===== 覆盖率自检结束 =====

    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=det_collate_fn)
    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=det_collate_fn) if val_ds else None

    # 模型
    model = build_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 优化与调度
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and device.type=='cuda'))
    criterion = CenterLoss(num_classes=args.num_classes, hm_weight=args.loss_hm, wh_weight=args.loss_wh, off_weight=args.loss_off)

    # mAP 指标（可选）
    try:
        import importlib
        MeanAveragePrecision = importlib.import_module('torchmetrics.detection.mean_ap').MeanAveragePrecision
        has_map = True
        # ✅ 强制在CPU设备初始化,避免GPU/CPU混用
        metric_map = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3, 0.5, 0.75])
        metric_map = metric_map.to('cpu')  # 显式移到CPU
        # 放宽每图最大检测数并关闭冗余告警
        try:
            metric_map.max_detection_threshold = 1000
            metric_map.warn_on_many_detections = False
        except Exception:
            pass
        print("[INFO] torchmetrics.MeanAveragePrecision loaded successfully (CPU mode).")
    except Exception as e:
        has_map = False
        metric_map = None
        print(f"[WARN] torchmetrics not available ({e}), will use fallback mAP calculation.")

    def _decode_predictions(
        preds,
        conf_thresh=0.05,
        topk=60,
        nms_thresh=0.5,
        min_wh_px=3.0,
        curriculum_epoch=None,
        max_total_boxes=100,
        max_wh_p2=96.0,
        max_wh_p3=256.0,
        max_wh_p4=512.0,
        use_scales='all'
    ):
        """多尺度解码：从 P2/P3/P4 三个尺度的 hm 取 top-k，结合 wh/off 构框
        
        ⚠️ 关键改进:
        1. 三尺度联合解码 (P2 负责小目标, P3 中目标, P4 大目标)
        2. wh 已归一化到 [0,1],需反归一化到像素坐标
        3. offset 已用 tanh 限制到 [-1,1]
        4. 各尺度独立 min_wh 阈值 (P2宽松, P4严格)
        5. 📋 模块2: 各尺度独立最大wh上限 (防止极端大框)
        ✅ 课程化收紧: epoch 0-1 极宽松 → epoch 2-3 中等 → epoch 4-7 紧 → epoch 8+ 严格
        """
        from torchvision.ops import batched_nms
        # 🎓 课程学习: 严格收紧策略(VTS专用,更激进)
        if curriculum_epoch is not None:
            if curriculum_epoch < 2:
                # Epoch 0-1: 极宽松,捕捉早期信号
                conf_thresh = 0.01
                topk = 200
                min_wh_px = 0.5
                max_total_boxes = 150
            elif curriculum_epoch < 4:
                # Epoch 2-3: 中等过滤
                conf_thresh = 0.03
                topk = 120
                min_wh_px = 2.0
                max_total_boxes = 120
            elif curriculum_epoch < 8:
                # Epoch 4-7: 紧凑过滤
                conf_thresh = 0.04
                topk = 80
                min_wh_px = 2.5
                max_total_boxes = 100
            else:
                # Epoch 8+: 严格过滤
                conf_thresh = 0.05
                topk = 60
                min_wh_px = 3.0
                max_total_boxes = 80
        
        # ✅ 各尺度独立的最小框阈值
        scale_min_wh = {
            'p2': min_wh_px * 0.8,  # P2 更宽松 (FCS 小目标友好)
            'p3': min_wh_px,
            'p4': min_wh_px * 1.5   # P4 更严格 (过滤小框噪声)
        }
        
        # 📋 模块2: 各尺度最大框上限 (像素)
        scale_max_wh = {
            'p2': max_wh_p2,
            'p3': max_wh_p3,
            'p4': max_wh_p4
        }
        
        # ✅ 解析三尺度预测
        if isinstance(use_scales, str):
            if use_scales.lower() == 'all':
                scales = ['p2', 'p3', 'p4']
            else:
                scales = [s.strip().lower() for s in use_scales.split(',') if s.strip()]
        else:
            scales = list(use_scales)
        out = []
        decode_call_count = getattr(_decode_predictions, '_call_count', 0)
        _decode_predictions._call_count = decode_call_count + 1
        
        # 获取 batch size (从任意尺度)
        B = preds['p2']['hm'].shape[0]
        
        for b in range(B):
            all_boxes, all_scores, all_labels = [], [], []
            
            # 遍历所选尺度
            for scale in scales:
                hm = preds[scale]['hm'][b:b+1]  # [1,C,H,W]
                wh = preds[scale]['wh'][b:b+1]  # [1,2,H,W]
                off = preds[scale]['off'][b:b+1]  # [1,2,H,W]
                stride = preds[scale]['stride']
                
                C, H, W = hm.shape[1:]
                
                # Max pooling 找局部最大值
                pool = torch.nn.functional.max_pool2d(hm, 3, 1, 1)
                keep = (hm == pool)
                hm = hm * keep
                
                # Top-k 候选
                scores, inds = torch.topk(hm.view(C, -1), topk)
                xs = (inds % W).float()
                ys = (inds // W).float()
                
                wh_flat = wh.view(2, -1)
                off_flat = off.view(2, -1)
                
                for c in range(C):
                    for k in range(topk):
                        sc = scores[c, k].item()
                        if sc < conf_thresh:
                            continue
                        
                        ind = inds[c, k].item()
                        # ✅ offset 已经 tanh 限制到 [-1,1],直接使用
                        xc = (xs[c, k] + off_flat[0, ind]).item()
                        yc = (ys[c, k] + off_flat[1, ind]).item()
                        
                        # ✅ wh 已归一化到 [0,1],需转为特征图坐标再映射到512
                        w_norm = max(0.0, wh_flat[0, ind].item())
                        h_norm = max(0.0, wh_flat[1, ind].item())
                        w_feat = w_norm * W
                        h_feat = h_norm * H
                        
                        # 转为 512×512 像素坐标
                        scale_x = 512.0 / W
                        scale_y = 512.0 / H
                        x1 = (xc - w_feat / 2) * scale_x
                        y1 = (yc - h_feat / 2) * scale_y
                        x2 = (xc + w_feat / 2) * scale_x
                        y2 = (yc + h_feat / 2) * scale_y
                        
                        # Clamp到[0,512]
                        x1 = max(0.0, min(x1, 512.0))
                        y1 = max(0.0, min(y1, 512.0))
                        x2 = max(0.0, min(x2, 512.0))
                        y2 = max(0.0, min(y2, 512.0))
                        
                        # ✅ 各尺度独立的最小框过滤
                        w_px = x2 - x1
                        h_px = y2 - y1
                        if w_px < scale_min_wh[scale] or h_px < scale_min_wh[scale]:
                            continue
                        
                        # 📋 模块2: 各尺度最大框上限过滤 (防止极端大框)
                        if w_px > scale_max_wh[scale] or h_px > scale_max_wh[scale]:
                            continue
                        
                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(sc)
                        all_labels.append(c)
            
            if len(all_boxes) == 0:
                out.append({'boxes': torch.zeros((0,4), dtype=torch.float32), 'scores': torch.zeros((0,), dtype=torch.float32), 'labels': torch.zeros((0,), dtype=torch.long)})
            else:
                # 构造CPU张量
                boxes_t = torch.tensor(all_boxes, dtype=torch.float32, device='cpu')
                scores_t = torch.tensor(all_scores, dtype=torch.float32, device='cpu')
                labels_t = torch.tensor(all_labels, dtype=torch.long, device='cpu')
                
                # NMS去重
                if nms_thresh > 0 and len(boxes_t) > 0:
                    keep_idx = batched_nms(boxes_t, scores_t, labels_t, nms_thresh)
                    boxes_t = boxes_t[keep_idx]
                    scores_t = scores_t[keep_idx]
                    labels_t = labels_t[keep_idx]
                
                # 最终限量
                if len(scores_t) > max_total_boxes:
                    top_indices = torch.argsort(scores_t, descending=True)[:max_total_boxes]
                    boxes_t = boxes_t[top_indices]
                    scores_t = scores_t[top_indices]
                    labels_t = labels_t[top_indices]
                
                out.append({'boxes': boxes_t, 'scores': scores_t, 'labels': labels_t})
        
        # 🔍 诊断日志: 首次解码详情
        if decode_call_count == 0:
            total_boxes = sum(len(o['boxes']) for o in out)
            avg_boxes = total_boxes / B if B > 0 else 0
            print(f"\n[DEBUG _decode_predictions] Multi-scale decoding: B={B}")
            print(f"  Curriculum: conf={conf_thresh:.4f}, topk={topk}, min_wh={min_wh_px:.1f}px, max_total={max_total_boxes}")
            print(f"  Scale-specific min_wh: P2={scale_min_wh['p2']:.1f}, P3={scale_min_wh['p3']:.1f}, P4={scale_min_wh['p4']:.1f}")
            print(f"  Decoded: {total_boxes} boxes ({avg_boxes:.1f} per image)")
            if total_boxes > 0:
                sample_boxes = out[0]['boxes'][:3] if len(out[0]['boxes']) > 0 else None
                if sample_boxes is not None:
                    print(f"  Sample boxes (pixel coords [0,512]): {sample_boxes.tolist()}")
        
        return out

    def _targets_to_metric_fmt(targets):
        # 将 [cls,x,y,w,h,conf] 归一坐标 转为 torchmetrics 需要的像素 xyxy
        # ✅ 强制在CPU上构造,避免设备混用
        t_fmt = []
        for t in targets:
            # 确保输入在CPU
            t_cpu = t.cpu() if t.is_cuda else t
            if t_cpu.numel() == 0:
                t_fmt.append({'boxes': torch.zeros((0,4), dtype=torch.float32), 'labels': torch.zeros((0,), dtype=torch.long)})
                continue
            boxes = t_cpu[:, 1:5]
            xyxy = torch.zeros_like(boxes)
            xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * 512
            xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * 512
            xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * 512
            xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * 512
            t_fmt.append({'boxes': xyxy, 'labels': t_cpu[:, 0].long()})
        return t_fmt

    # IoU 与额外指标工具
    def _box_iou(a: torch.Tensor, b: torch.Tensor):
        # a: [Na,4], b: [Nb,4] in xyxy
        if a.numel() == 0 or b.numel() == 0:
            return a.new_zeros((a.shape[0], b.shape[0]))
        tl = torch.maximum(a[:, None, :2], b[None, :, :2])
        br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
        wh = (br - tl).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
        area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
        union = area_a[:, None] + area_b[None, :] - inter + 1e-6
        return inter / union

    def _compute_detection_metrics(all_dets, all_tgts, num_classes: int, iou_thr: float = 0.5):
        # 聚合所有预测（像素 xyxy，带 scores/labels）与GT（像素 xyxy，带 labels）
        # 构造每类的列表
        cls_pred_scores = [[] for _ in range(num_classes)]
        cls_pred_tpflag = [[] for _ in range(num_classes)]
        cls_num_gt = [0 for _ in range(num_classes)]  # 每类 GT 数量
        # 混淆矩阵
        conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
        total_TP = 0; total_FP = 0; total_FN = 0
        # 正确性用于校准
        all_scores = []
        all_correct = []
        
        # 第一遍：统计 GT 数量（避免空集问题）
        for tgt in all_tgts:
            gl = tgt['labels']
            cls_vals, counts = torch.unique(gl, return_counts=True)
            for c, n in zip(cls_vals.tolist(), counts.tolist()):
                if 0 <= c < num_classes:
                    cls_num_gt[c] += int(n)
        
        # 第二遍：匹配预测与 GT
        for det, tgt in zip(all_dets, all_tgts):
            pb = det['boxes']; ps = det['scores']; pl = det['labels']
            gb = tgt['boxes']; gl = tgt['labels']
            if pb.numel() == 0 and gb.numel() == 0:
                continue
            if pb.numel() == 0 and gb.numel() > 0:
                total_FN += int(gb.shape[0])
                continue
            if gb.numel() == 0 and pb.numel() > 0:
                total_FP += int(pb.shape[0])
                for s in ps.tolist():
                    all_scores.append(float(s)); all_correct.append(0.0)
                continue
            iou = _box_iou(pb, gb)
            # 贪心匹配：按分数降序遍历预测
            order = torch.argsort(ps, descending=True)
            matched_g = set()
            for idx in order.tolist():
                pbox = pb[idx]; plab = int(pl[idx]); score = float(ps[idx])
                # 找到与此预测 IoU 最高的GT
                ious = iou[idx]
                best_iou, best_g = float(0.0), int(-1)
                if ious.numel() > 0:
                    j = torch.argmax(ious).item()
                    best_iou = float(ious[j].item()); best_g = int(j)
                is_tp = False
                if best_iou >= iou_thr and best_g not in matched_g:
                    matched_g.add(best_g)
                    gcls = int(gl[best_g])
                    conf_mat[gcls, plab] += 1
                    if plab == gcls:
                        is_tp = True
                # 记录分数和 TP/FP
                if 0 <= plab < num_classes:
                    cls_pred_scores[plab].append(score)
                    cls_pred_tpflag[plab].append(1 if is_tp else 0)
                all_scores.append(score); all_correct.append(1.0 if is_tp else 0.0)
                if is_tp:
                    total_TP += 1
                else:
                    total_FP += 1
            # 未匹配的GT计为FN
            fn = int(gb.shape[0] - len(matched_g))
            total_FN += fn
        # 逐类 PR 与 AP（空集保护）
        per_class_ap = np.zeros((num_classes,), dtype=np.float32)
        pr_curves = {}
        eps = 1e-8
        for c in range(num_classes):
            scores = np.array(cls_pred_scores[c], dtype=np.float32)
            flags = np.array(cls_pred_tpflag[c], dtype=np.int32)
            npos = int(cls_num_gt[c])
            # 该类无 GT：AP=0，PR 为空
            if npos == 0:
                pr_curves[c] = (np.array([0.0]), np.array([0.0]))
                per_class_ap[c] = 0.0
                continue
            # 该类有 GT 但无预测：AP=0
            if scores.size == 0:
                pr_curves[c] = (np.array([0.0]), np.array([0.0]))
                per_class_ap[c] = 0.0
                continue
            order = np.argsort(-scores)
            flags = flags[order]
            tp = np.cumsum(flags)
            fp = np.cumsum(1 - flags)
            recall = tp / max(1, npos)
            precision = tp / np.maximum(tp + fp, eps)
            # AP（插值包络法）
            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([0.0], precision, [0.0]))
            for i in range(mpre.size - 1, 0, -1):
                mpre[i-1] = max(mpre[i-1], mpre[i])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
            per_class_ap[c] = ap
            pr_curves[c] = (recall, precision)
        # micro/macro 指标
        micro_p = total_TP / max(1, total_TP + total_FP)
        micro_r = total_TP / max(1, total_TP + total_FN)
        micro_f1 = 2 * micro_p * micro_r / max(eps, (micro_p + micro_r))
        class_ps, class_rs = [], []
        for c in range(num_classes):
            scores = np.array(cls_pred_scores[c], dtype=np.float32)
            flags = np.array(cls_pred_tpflag[c], dtype=np.int32)
            if scores.size == 0:
                class_ps.append(0.0); class_rs.append(0.0); continue
            tp = int(flags.sum()); fp = int((1 - flags).sum()); fn = int(cls_num_gt[c] - tp)
            p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
            class_ps.append(p); class_rs.append(r)
        mask = np.array([n > 0 for n in cls_num_gt])
        macro_p = float(np.mean(np.array(class_ps)[mask])) if np.any(mask) else 0.0
        macro_r = float(np.mean(np.array(class_rs)[mask])) if np.any(mask) else 0.0
        macro_f1 = 2 * macro_p * macro_r / max(eps, (macro_p + macro_r))
        # 校准：可靠性图 & ECE & NLL（对所有预测，空集保护）
        scores_all = np.array(all_scores, dtype=np.float32)
        correct_all = np.array(all_correct, dtype=np.float32)
        if scores_all.size == 0:
            # 完全无预测：ECE=0，NLL=0，bin 全空
            bin_confs, bin_accs, bin_cnts = [0.0]*10, [0.0]*10, [0]*10
            ece, nll = 0.0, 0.0
        else:
            bins = np.linspace(0.0, 1.0, 11)
            bin_ids = np.digitize(scores_all, bins) - 1
            bin_confs, bin_accs, bin_cnts = [], [], []
            for b in range(10):
                m = (bin_ids == b)
                n = int(np.sum(m))
                if n == 0:
                    bin_confs.append(0.0); bin_accs.append(0.0); bin_cnts.append(0)
                else:
                    bin_confs.append(float(np.mean(scores_all[m])))
                    bin_accs.append(float(np.mean(correct_all[m])))
                    bin_cnts.append(n)
            N = max(1, int(scores_all.size))
            ece = float(np.sum([ (bin_cnts[i]/N) * abs(bin_accs[i] - bin_confs[i]) for i in range(10) ]))
            # NLL（对二分类正确性）
            s = np.clip(scores_all, 1e-6, 1.0-1e-6)
            nll = float(-np.mean(correct_all * np.log(s) + (1.0 - correct_all) * np.log(1.0 - s)))
        return {
            'per_class_ap': per_class_ap,
            'pr_curves': pr_curves,
            'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1,
            'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1,
            'conf_mat': conf_mat,
            'gt_per_class': cls_num_gt,  # 新增：每类 GT数量
            'reliability': {
                'bin_confs': bin_confs, 'bin_accs': bin_accs, 'bin_cnts': bin_cnts,
                'ece': ece, 'nll': nll
            }
        }

    # 运行目录
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # 自动：使用与分类相同的 CLS_0_200_ex1314 作为根目录
        train_list_path = Path(args.train_list).resolve()
        ds_dir = train_list_path.parent  # e.g., FCS_0_200_ex1314 or VTS_0_200_ex1314
        run_root = ds_dir.parent / 'CLS_0_200_ex1314'
        ts = time.strftime('%Y%m%d-%H%M')
        tag = 'Det-B1' if args.backbone_type == 'convnext' else 'Det-B0'
        run_dir = Path(run_root) / f'{tag}-{ts}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    cfg = {
        'data': {
            'train_list': str(Path(args.train_list).resolve()),
            'val_list': (str(Path(args.val_list).resolve()) if args.val_list else ''),
            'num_classes': args.num_classes,
            'smnet_preproc': bool(args.smnet_preproc),
            'edge_high': (None if args.edge_high is None else float(args.edge_high)),
            'edge_low': (None if args.edge_low is None else float(args.edge_low)),
            'edge_high_pct': (None if args.edge_high_pct is None else float(args.edge_high_pct)),
            'edge_low_pct': (None if args.edge_low_pct is None else float(args.edge_low_pct)),
            'corner_high': (None if args.corner_high is None else float(args.corner_high)),
            'corner_low': (None if args.corner_low is None else float(args.corner_low)),
            'corner_high_pct': (None if args.corner_high_pct is None else float(args.corner_high_pct)),
            'corner_low_pct': (None if args.corner_low_pct is None else float(args.corner_low_pct)),
            'harris_k': float(args.harris_k if args.harris_k is not None else 0.04),
            'spec_norm_mode': args.spec_norm_mode,
            'posenc_scale': float(args.posenc_scale),
            'strict_val': bool(args.strict_val),
        },
        'model': {
            'backbone_type': args.backbone_type,
            'pretrained': bool(args.pretrained),
            'checkpoint': args.checkpoint,
            'num_anchors': args.num_anchors,
            'head_out_up': int(max(1, args.head_out_up)),
            'pfaware_gate_target': float(args.pfaware_gate_target),
        },
        'train': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'amp': bool(args.amp),
            'num_workers': args.num_workers,
            'lr': args.lr,
            'exclude_classes': list(exc),
            'conf_thresh': args.conf_thresh,
            'loss': {'hm': args.loss_hm, 'wh': args.loss_wh, 'off': args.loss_off},
        },
        'decode': {
            'scales': args.decode_scales,
            'disable_curriculum': bool(args.disable_curriculum),
            'conf': (None if args.decode_conf is None else float(args.decode_conf)),
            'topk': (None if args.decode_topk is None else int(args.decode_topk)),
            'min_wh': (None if args.decode_min_wh is None else float(args.decode_min_wh)),
            'nms': float(args.decode_nms),
            'max_wh_p2': float(args.max_wh_p2),
            'max_wh_p3': float(args.max_wh_p3),
            'max_wh_p4': float(args.max_wh_p4),
        },
        'run_dir': str(run_dir.resolve()),
    }
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')

    # 监控指标：优先 mAP@0.5（max），否则使用 val_loss（min）
    monitor_mode = 'max' if (dl_val is not None) else 'min'
    best_val = float('-inf') if monitor_mode == 'max' else float('inf')
    best_epoch = -1
    no_improve = 0
    
    # 📋 模块3: 梯度监控初始化
    grad_stats = [] if args.enable_grad_monitor else None
    if args.enable_grad_monitor:
        print("✅ Gradient monitoring enabled → run_dir/grad_stats.json")

    def improved(curr):
        nonlocal best_val
        if curr != curr:
            return False
        if monitor_mode == 'max':
            return (curr > best_val + args.min_delta)
        else:
            return (curr < best_val - args.min_delta)

    def update_best(curr, epoch):
        nonlocal best_val, best_epoch, no_improve
        if improved(curr):
            best_val = curr; best_epoch = epoch; no_improve = 0; return True
        else:
            no_improve += 1; return False

    # 构造 CSV 头（包含每类 AP）
    ap_cols = [f"ap_c{i}" for i in range(int(args.num_classes))]
    log_csv = run_dir / 'log.csv'
    with log_csv.open('w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        header = [
            'epoch',
            'train_loss','train_loss_hm','train_loss_wh','train_loss_off',
            'val_loss','val_loss_hm','val_loss_wh','val_loss_off',
            'map','map50','map75',
            'micro_p','micro_r','micro_f1','macro_p','macro_r','macro_f1',
            'ar1','ar10','ar100',
            *ap_cols,
            'lr','epoch_time_sec'
        ]
        writer.writerow(header)
        for epoch in range(args.epochs):
            t0 = time.time()
            
            # 🔥 改进2: PF-Aware渐进式预热 (前5个epoch从0→target_scale)
            if hasattr(model, 'module'):
                head = model.module.head if hasattr(model.module, 'head') else model.module
            else:
                head = model.head if hasattr(model, 'head') else model
            
            if hasattr(head, 'pf_warmup_gate'):
                warmup_epochs = 5
                if epoch < warmup_epochs:
                    gate_value = float(args.pfaware_gate_target) * (epoch / warmup_epochs)
                else:
                    gate_value = float(args.pfaware_gate_target)
                head.pf_warmup_gate.data.fill_(gate_value)
                print(f"[PF-Aware Gate] Epoch {epoch+1}: {gate_value:.3f} / {float(args.pfaware_gate_target):.3f}")
            
            # 🔥 改进5: 训练早期保守策略 (前3个epoch降低学习率和梯度裁剪阈值)
            if epoch < 3:
                warmup_lr_scale = 0.1
                warmup_grad_clip = 1.0
                # 临时调整学习率
                base_lr = args.lr if epoch == 0 else optimizer.param_groups[0]['lr'] / 0.1  # 保存原始lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr * warmup_lr_scale
                print(f"[Early Warmup] Epoch {epoch+1}: LR={base_lr * warmup_lr_scale:.6f}, GradClip={warmup_grad_clip}")
            else:
                warmup_grad_clip = 2.0
            
            # 训练
            model.train(); total=0; sum_loss=0.0; sum_hm=0.0; sum_wh=0.0; sum_off=0.0
            train_iter = tqdm(dl_train, total=len(dl_train), desc=f"Train {epoch+1}/{args.epochs}", leave=False) if tqdm else dl_train
            # 训练期的临时 mAP@0.5（仅用于进度展示）
            metric_map_train = None
            if has_map:
                try:
                    metric_map_train = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])
                    try:
                        metric_map_train.max_detection_threshold = 1000
                        metric_map_train.warn_on_many_detections = False
                    except Exception:
                        pass
                    metric_map_train.reset()
                except Exception:
                    metric_map_train = None
            map_interval = 20  # 每N个batch估算一次 mAP50
            for step, (imgs, targets) in enumerate(train_iter):
                # ---------- NaN / Inf 检查 - 早期捕获并跳过有问题的 batch ----------
                # NOTE: 不要改变数据内容, 只检测并跳过含 NaN 的样本以保护模型权重
                def _has_nan_in_targets(ts):
                    for _t in ts:
                        if _t is None:
                            continue
                        t_cpu = _t if not _t.is_cuda else _t.cpu()
                        if t_cpu.numel() > 0 and torch.isnan(t_cpu).any():
                            return True
                    return False

                imgs = imgs.to(device, non_blocking=True)
                targets = [t.to(device) for t in targets]

                if torch.isnan(imgs).any():
                    print(f"[WARN] NaN detected in imgs at epoch {epoch+1} step {step}; skipping batch")
                    continue
                if _has_nan_in_targets(targets):
                    print(f"[WARN] NaN detected in targets at epoch {epoch+1} step {step}; skipping batch")
                    continue

                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    try:
                        with torch.amp.autocast(device_type='cuda', enabled=True):
                            pred = model(imgs)
                            # 检查模型输出中是否存在 NaN/Inf
                            def _has_nan_in_preds(p):
                                if isinstance(p, dict):
                                    for v in p.values():
                                        if _has_nan_in_preds(v):
                                            return True
                                    return False
                                if isinstance(p, torch.Tensor):
                                    return torch.isnan(p).any()
                                return False
                            if _has_nan_in_preds(pred):
                                print(f"[WARN] NaN in model outputs at epoch {epoch+1} step {step}; skipping batch")
                                continue

                            loss, parts = criterion(pred, targets)
                    except RuntimeError as e:
                        print(f"[ERROR] RuntimeError during forward at epoch {epoch+1} step {step}: {e}")
                        continue

                    # NaN/Inf 保护 + 诊断日志
                    if not torch.isfinite(loss):
                        if tqdm: train_iter.set_postfix(warn='skip NaN loss')
                        # 诊断输出：batch ID、loss 分量、第一个样本的框
                        print(f"\n[NaN Loss] Epoch {epoch+1} Step {step}: loss_hm={parts.get('loss_hm','?')} loss_wh={parts.get('loss_wh','?')} loss_off={parts.get('loss_off','?')}")
                        if len(targets) > 0 and targets[0].numel() > 0:
                            print(f"  Sample0 boxes: {targets[0][:3].tolist() if targets[0].size(0)>=3 else targets[0].tolist()}")
                        continue
                    
                    # ✅ 检查loss是否过大(可能导致梯度爆炸)
                    if float(loss.item()) > 100.0:
                        print(f"\n[WARN] Extremely large loss={float(loss.item()):.2f} at epoch {epoch+1} step {step}; skipping")
                        continue
                    
                    scaler.scale(loss).backward()
                    # 🔥 改进5: 动态梯度裁剪阈值 (早期更保守)
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=warmup_grad_clip)
                    
                    # 📋 模块3: 记录梯度异常
                    if not torch.isfinite(grad_norm):
                        print(f"\n[WARN] Infinite gradient norm at epoch {epoch+1} step {step}; skipping optimizer step")
                        if grad_stats is not None and len(grad_stats) < 500:  # 限制日志大小
                            grad_stats.append({
                                'epoch': epoch+1,
                                'step': step,
                                'grad_norm': 'inf',
                                'loss': float(loss.item()),
                                'loss_parts': {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in parts.items()},
                            })
                        scaler.update()
                        continue
                    
                    # 📋 模块3: 记录正常梯度(采样)
                    if grad_stats is not None and step % 50 == 0:
                        grad_stats.append({
                            'epoch': epoch+1,
                            'step': step,
                            'grad_norm': float(grad_norm.item()),
                            'loss': float(loss.item()),
                        })
                    
                    scaler.step(optimizer); scaler.update()
                else:
                    try:
                        pred = model(imgs)
                    except RuntimeError as e:
                        print(f"[ERROR] RuntimeError during forward at epoch {epoch+1} step {step}: {e}")
                        continue
                    # 模型输出 NaN 检查
                    def _has_nan_in_preds(p):
                        if isinstance(p, dict):
                            for v in p.values():
                                if _has_nan_in_preds(v):
                                    return True
                            return False
                        if isinstance(p, torch.Tensor):
                            return torch.isnan(p).any()
                        return False
                    if _has_nan_in_preds(pred):
                        print(f"[WARN] NaN in model outputs at epoch {epoch+1} step {step}; skipping batch")
                        continue

                    loss, parts = criterion(pred, targets)
                    if not torch.isfinite(loss):
                        if tqdm: train_iter.set_postfix(warn='skip NaN loss')
                        print(f"\n[NaN Loss] Epoch {epoch+1} Step {step}: loss not finite, skipping")
                        continue
                    
                    # ✅ 检查loss是否过大
                    if float(loss.item()) > 100.0:
                        print(f"\n[WARN] Extremely large loss={float(loss.item()):.2f} at epoch {epoch+1} step {step}; skipping")
                        continue
                    
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=warmup_grad_clip)
                    
                    # 📋 模块3: 记录梯度异常(非AMP分支)
                    if not torch.isfinite(grad_norm):
                        print(f"\n[WARN] Infinite gradient norm at epoch {epoch+1} step {step}; skipping optimizer step")
                        if grad_stats is not None and len(grad_stats) < 500:
                            grad_stats.append({
                                'epoch': epoch+1,
                                'step': step,
                                'grad_norm': 'inf',
                                'loss': float(loss.item()),
                                'loss_parts': {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in parts.items()},
                            })
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    # 📋 模块3: 记录正常梯度(采样,非AMP)
                    if grad_stats is not None and step % 50 == 0:
                        grad_stats.append({
                            'epoch': epoch+1,
                            'step': step,
                            'grad_norm': float(grad_norm.item()),
                            'loss': float(loss.item()),
                        })
                    
                    optimizer.step()
                bs = int(imgs.size(0))
                sum_loss += float(loss) * bs; total += bs
                # 累加分量（若提供）
                if isinstance(parts, dict):
                    sum_hm += float(parts.get('loss_hm', 0.0)) * bs
                    sum_wh += float(parts.get('loss_wh', 0.0)) * bs
                    sum_off += float(parts.get('loss_off', 0.0)) * bs
                # 训练期 mAP50（轻量、间隔评估，仅展示）
                curr_map50 = None
                # ✅ 完全禁用训练期metrics避免CUDA崩溃
                # if metric_map_train is not None:
                #     try:
                #         # 训练期使用更低阈值+topk捕捉早期信号
                #         dets = _decode_predictions(pred, conf_thresh=max(0.001, args.conf_thresh*0.5), topk=100, curriculum_epoch=epoch)
                #         dets_fmt = [{'boxes': d['boxes'].detach().cpu(), 'scores': d['scores'].detach().cpu(), 'labels': d['labels'].detach().cpu()} for d in dets]
                #         tgts_cpu = [t.detach().cpu() for t in targets]
                #         metric_map_train.update(dets_fmt, _targets_to_metric_fmt(tgts_cpu))
                #         if (step + 1) % map_interval == 0:
                #             res_tr = metric_map_train.compute()
                #             curr_map50 = float(res_tr.get('map_50', float('nan')))
                #     except Exception:
                #         curr_map50 = None
                if tqdm:
                    running = sum_loss / max(1, total)
                    postfix = { 'loss': f"{running:.4f}" }
                    if curr_map50 is not None and curr_map50==curr_map50:
                        postfix['mAP50'] = f"{curr_map50:.3f}"
                    train_iter.set_postfix(**postfix)
            train_loss = sum_loss / max(1,total)
            train_hm = (sum_hm / max(1,total)) if total>0 else 0.0
            train_wh = (sum_wh / max(1,total)) if total>0 else 0.0
            train_off= (sum_off/ max(1,total)) if total>0 else 0.0

            # 验证
            val_loss = float('nan'); val_hm=float('nan'); val_wh=float('nan'); val_off=float('nan')
            val_map=None; val_map50=None; val_map75=None
            micro_p=macro_p=micro_r=macro_r=micro_f1=macro_f1=float('nan')
            ar1=ar10=ar100=float('nan')
            per_class_ap = [float('nan')]*int(args.num_classes)
            dets_for_extra = []
            tgts_for_extra = []
            if dl_val is not None:
                model.eval(); total=0; sum_loss=0.0; sum_hm=0.0; sum_wh=0.0; sum_off=0.0
                if has_map:
                    metric_map.reset()
                val_iter = tqdm(dl_val, total=len(dl_val), desc="Val", leave=False) if tqdm else dl_val
                with torch.no_grad():
                    for vstep, (imgs, targets) in enumerate(val_iter):
                        imgs = imgs.to(device, non_blocking=True)
                        targets = [t.to(device) for t in targets]
                        preds = model(imgs)
                        loss, parts = criterion(preds, targets)
                        bs = int(imgs.size(0))
                        sum_loss += float(loss) * bs; total += bs
                        if isinstance(parts, dict):
                            sum_hm += float(parts.get('loss_hm', 0.0)) * bs
                            sum_wh += float(parts.get('loss_wh', 0.0)) * bs
                            sum_off += float(parts.get('loss_off', 0.0)) * bs
                        # ✅ 验证期使用课程化解码策略(不传显式参数,让curriculum_epoch控制)
                        # 选择静态或课程化解码
                        if args.disable_curriculum:
                            dec_conf = args.decode_conf if args.decode_conf is not None else args.conf_thresh
                            dec_topk = args.decode_topk if args.decode_topk is not None else 80
                            dec_minwh = args.decode_min_wh if args.decode_min_wh is not None else 2.0
                            dets = _decode_predictions(
                                preds,
                                conf_thresh=dec_conf,
                                topk=dec_topk,
                                nms_thresh=args.decode_nms,
                                min_wh_px=dec_minwh,
                                curriculum_epoch=None,
                                max_wh_p2=args.max_wh_p2,
                                max_wh_p3=args.max_wh_p3,
                                max_wh_p4=args.max_wh_p4,
                                use_scales=args.decode_scales
                            )
                        else:
                            dets = _decode_predictions(
                                preds,
                                curriculum_epoch=epoch,
                                nms_thresh=args.decode_nms,
                                max_wh_p2=args.max_wh_p2,
                                max_wh_p3=args.max_wh_p3,
                                max_wh_p4=args.max_wh_p4,
                                use_scales=args.decode_scales
                            )
                        # ✅ _decode_predictions 现在直接输出CPU张量,无需再转换
                        dets_fmt = dets
                        tgts_cpu = [t.detach().cpu() for t in targets]
                        tgts_fmt = _targets_to_metric_fmt(tgts_cpu)
                        # 🔍 诊断日志: 验证首批预测和GT的坐标范围
                        if vstep == 0 and epoch == 0:
                            pred_boxes = [d['boxes'] for d in dets_fmt if d['boxes'].numel() > 0]
                            gt_boxes = [t['boxes'] for t in tgts_fmt if t['boxes'].numel() > 0]
                            # 根据epoch显示当前课程化参数
                            if epoch < 2:
                                curr_params = "conf=0.01, topk=200, min_wh=0.5"
                            elif epoch < 4:
                                curr_params = "conf=0.03, topk=120, min_wh=2.0"
                            elif epoch < 8:
                                curr_params = "conf=0.04, topk=80, min_wh=2.5"
                            else:
                                curr_params = "conf=0.05, topk=60, min_wh=3.0"
                            print(f"\n[DEBUG Val Batch 0] Predictions: {len(pred_boxes)} samples with boxes (curriculum: {curr_params})")
                            if pred_boxes:
                                all_pred = torch.cat(pred_boxes, dim=0)
                                # 计算框的宽高
                                pred_wh = all_pred[:, 2:] - all_pred[:, :2]
                                print(f"  Pred boxes: count={all_pred.size(0)}, coord_range=[{all_pred.min():.1f}, {all_pred.max():.1f}]")
                                print(f"  Pred wh_range: w=[{pred_wh[:, 0].min():.1f}, {pred_wh[:, 0].max():.1f}], h=[{pred_wh[:, 1].min():.1f}, {pred_wh[:, 1].max():.1f}]")
                                print(f"  Sample pred boxes: {all_pred[:3].tolist()}")
                            else:
                                print(f"  ⚠️ No predictions generated (curriculum parameters may be too strict)")
                            if gt_boxes:
                                all_gt = torch.cat(gt_boxes, dim=0)
                                gt_wh = all_gt[:, 2:] - all_gt[:, :2]
                                print(f"  GT boxes: count={all_gt.size(0)}, coord_range=[{all_gt.min():.1f}, {all_gt.max():.1f}]")
                                print(f"  GT wh_range: w=[{gt_wh[:, 0].min():.1f}, {gt_wh[:, 0].max():.1f}], h=[{gt_wh[:, 1].min():.1f}, {gt_wh[:, 1].max():.1f}]")
                                print(f"  Sample GT boxes: {all_gt[:3].tolist()}")
                        # ✅ 确保所有张量都在CPU(dets_fmt/tgts_fmt已经是CPU张量,但double-check)
                        dets_for_extra.extend([{ 
                            'boxes': x['boxes'].cpu() if x['boxes'].is_cuda else x['boxes'].clone(), 
                            'scores': x['scores'].cpu() if x['scores'].is_cuda else x['scores'].clone(), 
                            'labels': x['labels'].cpu() if x['labels'].is_cuda else x['labels'].clone() 
                        } for x in dets_fmt])
                        tgts_for_extra.extend([{ 
                            'boxes': t['boxes'].cpu() if t['boxes'].is_cuda else t['boxes'].clone(), 
                            'labels': t['labels'].cpu() if t['labels'].is_cuda else t['labels'].clone() 
                        } for t in tgts_fmt])
                        if has_map:
                            # ✅ 现在可以启用torchmetrics update,因为metric_map和数据都在CPU
                            try:
                                metric_map.update(dets_fmt, tgts_fmt)
                            except Exception as e:
                                if vstep == 0 or vstep < 3:  # 打印前3次错误
                                    print(f"[WARN Vstep {vstep}] metric_map.update failed: {type(e).__name__}: {e}")
                                if vstep == 0:
                                    import traceback
                                    traceback.print_exc()
                        # 验证进度条：running val_loss + mAP50（间隔）
                        curr_vmap50 = None
                        # ✅ 禁用间隔compute避免CUDA错误
                        # if has_map and (vstep + 1) % map_interval == 0:
                        #     res_tmp = metric_map.compute()
                        #     curr_vmap50 = float(res_tmp.get('map_50', float('nan')))
                        if tqdm:
                            running = sum_loss / max(1, total)
                            postfix = { 'val_loss': f"{running:.4f}" }
                            if curr_vmap50 is not None and curr_vmap50==curr_vmap50:
                                postfix['mAP50'] = f"{curr_vmap50:.3f}"
                            val_iter.set_postfix(**postfix)
                val_loss = sum_loss / max(1,total)
                val_hm = (sum_hm / max(1,total)) if total>0 else float('nan')
                val_wh = (sum_wh / max(1,total)) if total>0 else float('nan')
                val_off= (sum_off/ max(1,total)) if total>0 else float('nan')
                
                # ✅ 计算 mAP: 现在metric_map在CPU且update已启用,可以正常使用torchmetrics
                if has_map:
                    try:
                        res = metric_map.compute()
                        # 同时取 0.3/0.5/0.75
                        val_map   = float(res.get('map', float('nan')))
                        val_map30 = float(res.get('map_30', float('nan')))
                        val_map50 = float(res.get('map_50', float('nan')))
                        val_map75 = float(res.get('map_75', float('nan')))
                        ar1  = float(res.get('mar_1', float('nan')))
                        ar10 = float(res.get('mar_10', float('nan')))
                        ar100= float(res.get('mar_100', float('nan')))
                        # Reset for next epoch
                        metric_map.reset()
                    except Exception as e:
                        print(f"[WARN] torchmetrics.compute() failed: {e}, using fallback")
                        has_map = False
                
                if not has_map:
                    # Fallback：先算额外指标，用其中的 per_class_ap
                    val_map, val_map50, val_map75 = float('nan'), float('nan'), float('nan')
                    ar1, ar10, ar100 = float('nan'), float('nan'), float('nan')
                    
                # 额外指标（micro/macro P/R/F1、每类AP、PR、混淆、可靠性）
                extra = _compute_detection_metrics(dets_for_extra, tgts_for_extra, int(args.num_classes), iou_thr=0.5)
                micro_p = extra['micro_p']; micro_r = extra['micro_r']; micro_f1 = extra['micro_f1']
                macro_p = extra['macro_p']; macro_r = extra['macro_r']; macro_f1 = extra['macro_f1']
                per_class_ap = extra['per_class_ap'].tolist()
                gt_per_class = extra['gt_per_class']  # 每类 GT 数量
                
                # Fallback mAP50：只统计有 GT 的类（避免空集稀释）
                if not has_map or not np.isfinite(val_map50):
                    valid_classes = [i for i in range(len(per_class_ap)) if gt_per_class[i] > 0]
                    valid_aps = [per_class_ap[i] for i in valid_classes if np.isfinite(per_class_ap[i])]
                    val_map50 = float(np.mean(valid_aps)) if valid_aps else 0.0
                    print(f"[INFO] Using fallback mAP@0.5 = {val_map50:.4f} (from {len(valid_aps)}/{len(valid_classes)} classes with GT)")
                
                # ✅ 只在最后epoch导出详细图表,减少磁盘IO
                is_final_epoch = (epoch + 1 == args.epochs)
                try:
                    if is_final_epoch:
                        # 每类AP
                        with (run_dir / f"per_class_ap_final.csv").open('w', newline='', encoding='utf-8') as fpc:
                            wpc = csv.writer(fpc); wpc.writerow(['class','ap'])
                            for i, ap in enumerate(per_class_ap): wpc.writerow([i, f"{ap:.6f}"])
                        # 混淆矩阵
                        cm = extra['conf_mat']
                        np.savetxt(run_dir / f"confusion_final.csv", cm, fmt='%d', delimiter=',')
                        if _has_plt:
                            plt.figure(figsize=(6,5))
                            cm_norm = cm / np.maximum(1, cm.sum(axis=1, keepdims=True))
                            plt.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
                            plt.title('Confusion (row=GT, col=Pred)')
                            plt.colorbar(fraction=0.046, pad=0.04)
                            plt.xlabel('Pred'); plt.ylabel('GT')
                            plt.tight_layout(); plt.savefig(run_dir / f"confusion_final.png", dpi=200)
                            plt.close()
                        # 可靠性
                        rel = extra['reliability']
                        with (run_dir / f"reliability_final.csv").open('w', newline='', encoding='utf-8') as fr:
                            wr = csv.writer(fr); wr.writerow(['bin','conf','acc','count'])
                            for i,(c,a,n) in enumerate(zip(rel['bin_confs'], rel['bin_accs'], rel['bin_cnts'])):
                                wr.writerow([i, f"{c:.6f}", f"{a:.6f}", n])
                            wr.writerow(['ECE', f"{rel['ece']:.6f}"])
                            wr.writerow(['NLL', f"{rel['nll']:.6f}"])
                        if _has_plt:
                            plt.figure(figsize=(5,5))
                            xs = np.linspace(0,1,100)
                            plt.plot(xs, xs, 'k--', label='Ideal')
                            plt.plot(rel['bin_confs'], rel['bin_accs'], 'o-', label=f"ECE={rel['ece']:.3f}")
                            plt.xlim(0,1); plt.ylim(0,1)
                            plt.xlabel('Confidence'); plt.ylabel('Accuracy')
                            plt.legend(); plt.tight_layout(); plt.savefig(run_dir / f"reliability_final.png", dpi=200)
                            plt.close()
                        # PR 曲线（导出CSV；图绘制Top-6 AP）
                        pr_dir = run_dir / f"pr_curves_final"
                        pr_dir.mkdir(parents=True, exist_ok=True)
                        ap_pairs = []
                        for c,(rec,pre) in extra['pr_curves'].items():
                            np.savetxt(pr_dir / f"cls_{c:02d}.csv", np.stack([rec, pre], axis=1), delimiter=',', fmt='%.6f')
                            ap_pairs.append((per_class_ap[c], c))
                        ap_pairs.sort(reverse=True)
                        if _has_plt:
                            plt.figure(figsize=(7,5))
                            for ap,c in ap_pairs[:6]:
                                rec, pre = extra['pr_curves'][c]
                                plt.plot(rec, pre, label=f"c{c} AP={ap:.2f}")
                            plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Top-6 PR Curves')
                            plt.xlim(0,1); plt.ylim(0,1); plt.legend()
                            plt.tight_layout(); plt.savefig(run_dir / f"pr_top6_final.png", dpi=200)
                            plt.close()
                except Exception:
                    pass

            # 记录与打印
            lr_curr = float(optimizer.param_groups[0]['lr'])
            epoch_time = time.time() - t0
            row = [
                epoch+1,
                f"{train_loss:.6f}", f"{train_hm:.6f}", f"{train_wh:.6f}", f"{train_off:.6f}",
                (f"{val_loss:.6f}" if val_loss==val_loss else ''),
                (f"{val_hm:.6f}" if val_hm==val_hm else ''),
                (f"{val_wh:.6f}" if val_wh==val_wh else ''),
                (f"{val_off:.6f}" if val_off==val_off else ''),
                (f"{val_map:.6f}" if val_map is not None else ''),
                (f"{val_map50:.6f}" if val_map50 is not None else ''),
                (f"{val_map75:.6f}" if val_map75 is not None else ''),
                (f"{micro_p:.6f}" if micro_p==micro_p else ''),
                (f"{micro_r:.6f}" if micro_r==micro_r else ''),
                (f"{micro_f1:.6f}" if micro_f1==micro_f1 else ''),
                (f"{macro_p:.6f}" if macro_p==macro_p else ''),
                (f"{macro_r:.6f}" if macro_r==macro_r else ''),
                (f"{macro_f1:.6f}" if macro_f1==macro_f1 else ''),
                (f"{ar1:.6f}" if ar1==ar1 else ''),
                (f"{ar10:.6f}" if ar10==ar10 else ''),
                (f"{ar100:.6f}" if ar100==ar100 else ''),
            ]
            # 每类AP
            for ap in per_class_ap:
                row.append(f"{ap:.6f}")
            row += [f"{lr_curr:.6e}", f"{epoch_time:.3f}"]
            writer.writerow(row)
            fcsv.flush()

            # 控制台输出：突出检测关键指标
            msg = f"Epoch {epoch+1}/{args.epochs} - loss={train_loss:.4f}"
            if val_loss==val_loss: 
                msg += f" | val_loss={val_loss:.4f}"
            # 核心检测指标（带 mAP@0.3 诊断）
            if 'val_map30' in locals() and np.isfinite(val_map30):
                msg += f" | mAP@0.3={val_map30:.4f}"
            if val_map50 is not None and np.isfinite(val_map50): 
                msg += f" | mAP@0.5={val_map50:.4f}"
            if val_map75 is not None and np.isfinite(val_map75): 
                msg += f" | mAP@0.75={val_map75:.4f}"
            if np.isfinite(micro_f1): 
                msg += f" | micro_F1={micro_f1:.4f}"
            if np.isfinite(macro_f1): 
                msg += f" | macro_F1={macro_f1:.4f}"
            print(msg)

            # 保存 last
            if (epoch + 1) % max(1, args.save_interval) == 0:
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch + 1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'scaler': scaler.state_dict() if scaler is not None else None,
                }, run_dir / 'last.pt')

            # 选择监控值：优先 mAP@0.5
            monitor_val = None
            if dl_val is not None and has_map and (val_map50 is not None) and (val_map50==val_map50):
                monitor_val = val_map50
                monitor_mode = 'max'
            elif dl_val is not None and val_loss==val_loss:
                monitor_val = val_loss
                monitor_mode = 'min'

            # early stop + 保存 best
            if monitor_val is not None:
                if update_best(monitor_val, epoch+1):
                    torch.save({
                        'model': model.state_dict(),
                        'epoch': epoch + 1,
                        'best_monitor': monitor_val,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler is not None else None,
                        'scaler': scaler.state_dict() if scaler is not None else None,
                    }, run_dir / 'best.pt')
                if args.early_stop and no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch+1} (best at epoch {best_epoch}: {best_val:.6f})")
                    break

            scheduler.step()

    # 保存最终指标摘要
    summary = {
        'best_epoch': best_epoch,
        'best_value': best_val,
        'monitor': ('mAP@0.5' if monitor_mode=='max' else 'val_loss'),
    }
    (run_dir / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    
    # 保存梯度监控数据 (若启用)
    if 'grad_stats' in locals() and grad_stats:
        (run_dir / 'grad_stats.json').write_text(json.dumps(grad_stats, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Saved {len(grad_stats)} gradient monitoring records to grad_stats.json")


if __name__ == '__main__':
    main()
