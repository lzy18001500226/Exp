import argparse
import json
import os
from pathlib import Path
import sys
import time
import csv
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils as nn_utils

# 将 DSFNet/src 加入 sys.path
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
# 新增：工作区根（DSFNet 的上一级目录），用于默认输出到 Exp/Model 下
WORKSPACE_ROOT = ROOT_DIR.parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.base_fusion_model import BaseFusionModel
from data.dataset_pair import PairFCSVTSNpzDataset
try:
    from utils.seed import set_seed
except Exception:
    def set_seed(seed: int = 3407):
        try:
            import random, numpy as _np, torch as _torch
            random.seed(seed)
            _np.random.seed(seed)
            _torch.manual_seed(seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

def parse_args():
    p = argparse.ArgumentParser()
    # 基本/模式
    p.add_argument('--mode', type=str, default='single', choices=['single', 'concat'])
    p.add_argument('--backbone_type', type=str, default='swin', choices=['swin', 'convnext'])

    # 数据与运行目录
    p.add_argument('--train_list', type=str, required=True)
    p.add_argument('--val_list', type=str, default=None)
    p.add_argument('--vts_list', type=str, default=None, help='当 mode=concat 时需要提供 VTS 列表')
    p.add_argument('--run_dir', type=str, default='',
                   help='Run directory. If empty, auto create under DSFNet-Output/CLS_0_200_ex1314 as Run-B0-* or Run-B1-*')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--stats', type=str, default=None)
    p.add_argument('--stem_min', type=int, default=0)
    p.add_argument('--stem_max', type=int, default=200)

    # 模型与训练基本参数
    p.add_argument('--embed_dim', type=int, default=512)
    p.add_argument('--num_classes', type=int, default=22)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--save_interval', type=int, default=1)

    # 预训练/骨干（用于记录到配置，实际加载由各 encoder 控制）
    p.add_argument('--swin_name', type=str, default='swin_base_patch4_window7_224')
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--swin_checkpoint', type=str, default='')

    # 优化相关
    p.add_argument('--backbone_lr', type=float, default=1e-4)
    p.add_argument('--head_lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--freeze_backbone_epochs', type=int, default=5)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--clip_grad', type=float, default=1.0)

    # 数据增强参数（保持向后兼容）
    p.add_argument('--is_train_aug', action='store_true')
    p.add_argument('--use_bg_mix', action='store_true')
    p.add_argument('--bg_mix_p', type=float, default=0.5)
    p.add_argument('--alpha_min', type=float, default=0.1)
    p.add_argument('--alpha_max', type=float, default=0.3)

    p.add_argument('--use_freq_shift', action='store_true')
    p.add_argument('--freq_shift_p', type=float, default=0.8)
    p.add_argument('--shift_mhz', type=float, default=10.0)
    p.add_argument('--use_frac_freq_shift', action='store_true')

    p.add_argument('--use_time_stretch', action='store_true')
    p.add_argument('--time_stretch_p', type=float, default=0.8)
    p.add_argument('--time_scale_min', type=float, default=0.9)
    p.add_argument('--time_scale_max', type=float, default=1.1)

    p.add_argument('--use_noise', action='store_true')
    p.add_argument('--noise_p', type=float, default=0.5)
    p.add_argument('--noise_k_min', type=float, default=0.01)
    p.add_argument('--noise_k_max', type=float, default=0.05)

    p.add_argument('--use_specaug', action='store_true')
    p.add_argument('--specaug_p', type=float, default=0.5)
    p.add_argument('--specaug_time_ratio', type=float, default=0.2)
    p.add_argument('--specaug_freq_ratio', type=float, default=0.2)
    p.add_argument('--specaug_num_masks', type=int, default=2)

    p.add_argument('--bandwidth_mhz', type=float, default=None)

    args = p.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return args


def _load_stats(path: str | None):
    if not path:
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def build_dataloaders(args):
    # 数据与加载器（训练）
    if args.mode == 'concat':
        if not args.vts_list:
            raise ValueError('mode=concat 需要提供 --vts_list')
        ds_train = PairFCSVTSNpzDataset(
            fcs_list_file=args.train_list,
            vts_list_file=args.vts_list,
            stats_fcs=None,
            stats_vts=None,
            is_train=True,
        )
        ds_val = None
        if args.val_list:
            ds_val = PairFCSVTSNpzDataset(
                fcs_list_file=args.val_list,
                vts_list_file=args.vts_list,
                stats_fcs=None,
                stats_vts=None,
                is_train=False,
            )
    else:
        # 单模态数据集
        from data.dataset_fcs_vts import FCSVTSNpzDataset
        ds_train = FCSVTSNpzDataset(
            list_file=args.train_list,
            normalize_stats=None,
            stem_min=args.stem_min,
            stem_max=args.stem_max,
            is_train=True,
            use_bg_mix=args.use_bg_mix,
            alpha_range=(args.alpha_min, args.alpha_max),
            use_freq_shift=args.use_freq_shift,
            shift_mhz=args.shift_mhz,
            use_time_stretch=args.use_time_stretch,
            time_scale_range=(args.time_scale_min, args.time_scale_max),
            use_noise=args.use_noise,
            snr_range=(args.noise_k_min, args.noise_k_max),
            use_specaug=args.use_specaug,
            specaug_time_ratio=args.specaug_time_ratio,
            specaug_freq_ratio=args.specaug_freq_ratio,
            specaug_num_masks=args.specaug_num_masks,
            bandwidth_mhz=args.bandwidth_mhz,
        )
        ds_val = None
        if args.val_list:
            ds_val = FCSVTSNpzDataset(
                list_file=args.val_list,
                normalize_stats=None,
                stem_min=None,
                stem_max=None,
                is_train=False,
            )

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_val = None
    if ds_val is not None:
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return dl_train, dl_val


def build_model(args, num_classes: int):
    # 与当前 BaseFusionModel 签名对齐
    model = BaseFusionModel(num_classes=num_classes, backbone_type=args.backbone_type, mode=args.mode, embed_dim=args.embed_dim)
    return model


def run_one_batch(model, batch, criterion):
    if isinstance(batch, (list, tuple)) and len(batch) == 4:
        fcs, vts, vts_valid, labels = batch
        logits = model(fcs.cuda(non_blocking=True), vts.cuda(non_blocking=True), vts_valid.cuda(non_blocking=True))
        loss = criterion(logits, labels.cuda(non_blocking=True))
        return loss, logits, labels
    else:
        x, labels = batch
        logits = model(x.cuda(non_blocking=True))
        loss = criterion(logits, labels.cuda(non_blocking=True))
        return loss, logits, labels

# 内联最小 Trainer（B0）：支持 AMP 与 tqdm，可返回 (avg_loss, avg_acc)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

class Trainer:
    def __init__(self, model, optimizer, device: str = 'cuda', amp: bool = True, criterion: torch.nn.Module | None = None, clip_grad: float | None = 1.0):
        use_cuda = (device == 'cuda' and torch.cuda.is_available())
        self.device = torch.device('cuda') if use_cuda else torch.device('cpu')
        self.amp = amp and use_cuda
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else torch.nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp) if self.amp else None
        self.global_step = 0
        self.clip_grad = float(clip_grad) if clip_grad and clip_grad > 0 else None

    def fit_one_epoch(self, loader):
        self.model.train()
        total = 0
        sum_loss = 0.0
        sum_correct = 0
        iterator = tqdm(enumerate(loader), total=len(loader), desc="Training", leave=False) if tqdm else enumerate(loader)
        for it, (fcs, target) in iterator:
            fcs = fcs.to(self.device, non_blocking=True)
            # 强制标签为 long（类索引）
            target = target.to(self.device, non_blocking=True).long()
            self.optimizer.zero_grad(set_to_none=True)

            # 获取模型的类别数，便于后续形状自适应
            num_classes = None
            try:
                num_classes = int(getattr(self.model.cls_head[-1], 'out_features', None))
            except Exception:
                num_classes = None

            def _to_logits_2d(out):
                # 期望返回 [B,C]
                if out.dim() == 2:
                    return out
                if num_classes is not None:
                    # [B,N,C] 情况
                    if out.dim() >= 3 and out.shape[-1] == num_classes:
                        return out.mean(dim=tuple(range(1, out.dim()-1)))
                    # [B,C,N] 或 [B,C,H,W] 情况
                    if out.dim() >= 3 and out.shape[1] == num_classes:
                        return out.mean(dim=tuple(range(2, out.dim())))
                # 回退：将除 batch 外所有维度展平为特征，再线性投影（不理想，但可避免报错）
                return out.view(out.shape[0], -1)

            if self.amp:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    logits = self.model(fcs)
                    logits = _to_logits_2d(logits)
                    loss = self.criterion(logits, target)
                self.scaler.scale(loss).backward()
                # AMP 下先反缩放再裁剪
                if self.clip_grad is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(fcs)
                logits = _to_logits_2d(logits)
                loss = self.criterion(logits, target)
                loss.backward()
                if self.clip_grad is not None:
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct = (preds == target).sum().item()
            bs = fcs.size(0)
            sum_loss += float(loss) * bs
            sum_correct += int(correct)
            total += bs
            if tqdm and iterator is not None:
                iterator.set_postfix(loss=float(loss), acc=correct / max(1, bs))
            self.global_step += 1
        return (sum_loss / max(1, total), sum_correct / max(1, total))


def main():
    args = parse_args()

    set_seed(args.seed)
    # 确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 解析归一化统计（B0/B1 默认不使用）
    stats = None
    if getattr(args, 'stats', None) and os.path.isfile(args.stats):
        with open(args.stats, 'r', encoding='utf-8') as f:
            try:
                stats = json.load(f)
            except Exception:
                stats = None

    # 数据与加载器
    dl_train, dl_val = build_dataloaders(args)

    # 模型
    model = build_model(args, num_classes=args.num_classes)

    # 参数组：主干与头部分别学习率
    backbone_names_prefix = 'encoder_fcs.backbone.'
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(backbone_names_prefix):
            backbone_params.append(p)
        else:
            head_params.append(p)
    param_groups = [
        {'params': backbone_params, 'lr': args.backbone_lr, 'weight_decay': args.weight_decay},
        {'params': head_params, 'lr': args.head_lr, 'weight_decay': args.weight_decay},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    # 运行目录
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_root = WORKSPACE_ROOT / 'Model' / 'DSFNet-Output' / 'CLS_0_200_ex1314'
        ts = time.strftime('%Y%m%d-%H%M')
        tag = 'B1' if args.backbone_type == 'convnext' else 'B0'
        run_dir = Path(run_root) / f'Cls-{tag}-{ts}'
        run_dir.mkdir(parents=True, exist_ok=True)

    # 写 config.yaml
    cfg = {
        'data': {
            'train_list': str(Path(args.train_list).resolve()),
            'val_list': (str(Path(args.val_list).resolve()) if args.val_list else ''),
            'stats': str(Path(args.stats).resolve()) if getattr(args, 'stats', None) else '',
            'num_classes': args.num_classes,
            'stem_min': args.stem_min,
            'stem_max': args.stem_max,
        },
        'model': {
            'backbone_type': args.backbone_type,
            'embed_dim': args.embed_dim,
            'swin_name': args.swin_name,
            'swin_checkpoint': str(Path(args.swin_checkpoint).resolve()) if args.swin_checkpoint else '',
            'pretrained': bool(args.pretrained),
        },
        'train': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'seed': args.seed,
            'amp': bool(args.amp),
            'num_workers': args.num_workers,
            'save_interval': args.save_interval,
        },
        'aug': {
            'is_train_aug': bool(args.is_train_aug),
            'use_bg_mix': bool(args.use_bg_mix),
            'bg_mix_p': float(args.bg_mix_p),
            'alpha_range': [float(args.alpha_min), float(args.alpha_max)],
            'use_freq_shift': bool(args.use_freq_shift),
            'freq_shift_p': float(args.freq_shift_p),
            'shift_mhz': float(args.shift_mhz),
            'use_frac_freq_shift': bool(args.use_frac_freq_shift),
            'use_time_stretch': bool(args.use_time_stretch),
            'time_stretch_p': float(args.time_stretch_p),
            'time_scale_range': [float(args.time_scale_min), float(args.time_scale_max)],
            'use_noise': bool(args.use_noise),
            'noise_p': float(args.noise_p),
            'noise_k_range': [float(args.noise_k_min), float(args.noise_k_max)],
            'use_specaug': bool(args.use_specaug),
            'specaug_p': float(args.specaug_p),
            'specaug_time_ratio': float(args.specaug_time_ratio),
            'specaug_freq_ratio': float(args.specaug_freq_ratio),
            'specaug_num_masks': int(args.specaug_num_masks),
            'bandwidth_mhz': (float(args.bandwidth_mhz) if args.bandwidth_mhz is not None else None),
        },
        'run_dir': str(run_dir.resolve()),
    }
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')

    # --- 数据一致性自检（B0/B1） ---
    def _read_list(p):
        try:
            return [s.strip() for s in open(p, 'r', encoding='utf-8') if s.strip()]
        except Exception:
            return []
    def _stem_of(path: str):
        try:
            import re as _re
            m = _re.match(r'^(\d+)', Path(path).stem)
            return int(m.group(1)) if m else None
        except Exception:
            return None
    def _count_by_class(paths):
        d = {}
        for p in paths:
            try:
                c = int(Path(p).parent.name)
                d[c] = d.get(c, 0) + 1
            except Exception:
                continue
        return dict(sorted(d.items()))

    if args.mode != 'concat':
        train_paths = _read_list(args.train_list)
        val_paths = _read_list(args.val_list) if args.val_list else []
        stems_train = {s for s in (_stem_of(p) for p in train_paths) if s is not None}
        stems_val = {s for s in (_stem_of(p) for p in val_paths) if s is not None}
        inter = stems_train & stems_val
        sanity = {
            'train': {
                'count': len(train_paths),
                'stems': len(stems_train),
                'class_counts': _count_by_class(train_paths),
            },
            'val': {
                'count': len(val_paths),
                'stems': len(stems_val),
                'class_counts': _count_by_class(val_paths),
            },
            'overlap': {
                'stems_intersection': len(inter)
            }
        }
        (run_dir / 'data_sanity.json').write_text(json.dumps(sanity, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"[Sanity] stems train={len(stems_train)} val={len(stems_val)} inter={len(inter)}")
    # --- 自检结束 ---

    # 训练器与损失
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    trainer = Trainer(model, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu', amp=args.amp, criterion=criterion, clip_grad=args.clip_grad)

    # 验证评估
    def _infer_to_logits_2d(out: torch.Tensor) -> torch.Tensor:
        num_classes = None
        try:
            num_classes = int(getattr(model.cls_head[-1], 'out_features', None))
        except Exception:
            num_classes = None
        if out.dim() == 2:
            return out
        if num_classes is not None:
            if out.dim() >= 3 and out.shape[-1] == num_classes:
                return out.mean(dim=tuple(range(1, out.dim()-1)))
            if out.dim() >= 3 and out.shape[1] == num_classes:
                return out.mean(dim=tuple(range(2, out.dim())))
        return out.view(out.shape[0], -1)

    def evaluate(model, val_loader, criterion, device):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        if val_loader is None:
            return float('nan'), float('nan')
        with torch.no_grad():
            for fcs, target in val_loader:
                fcs = fcs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True).long()
                logits = model(fcs)
                logits = _infer_to_logits_2d(logits)
                loss = criterion(logits, target)
                preds = logits.argmax(dim=1)
                total_loss += float(loss) * fcs.size(0)
                total_correct += int((preds == target).sum().item())
                total_samples += int(fcs.size(0))
        if total_samples == 0:
            return float('nan'), float('nan')
        return total_loss / total_samples, total_correct / total_samples

    log_csv = run_dir / 'log.csv'
    with log_csv.open('w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr_backbone', 'lr_head'])
        best_loss = float('inf')
        best_epoch = -1
        best_acc = 0.0
        for epoch in range(args.epochs):
            # 按需冻结/解冻主干
            freeze = (epoch < max(0, args.freeze_backbone_epochs))
            if hasattr(model, 'encoder_fcs') and hasattr(model.encoder_fcs, 'backbone'):
                for p in model.encoder_fcs.backbone.parameters():
                    p.requires_grad = not freeze

            # 记录学习率（两个参数组）
            lr_bb = optimizer.param_groups[0]['lr'] if optimizer.param_groups else args.backbone_lr
            lr_hd = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else args.head_lr

            train_loss, train_acc = trainer.fit_one_epoch(dl_train)
            val_loss, val_acc = evaluate(model, dl_val, criterion, trainer.device) if dl_val is not None else (float('nan'), float('nan'))

            writer.writerow([
                epoch + 1,
                f"{train_loss:.6f}",
                (f"{train_acc:.6f}" if train_acc == train_acc else ""),
                (f"{val_loss:.6f}" if val_loss == val_loss else ""),
                (f"{val_acc:.6f}" if val_acc == val_acc else ""),
                f"{lr_bb:.6e}", f"{lr_hd:.6e}"
            ])
            fcsv.flush()
            msg = f"Epoch {epoch+1}/{args.epochs} - loss={train_loss:.4f}"
            if train_acc == train_acc:
                msg += f" acc={train_acc:.4f}"
            if dl_val is not None and val_loss == val_loss:
                msg += f" | val_loss={val_loss:.4f}"
                if val_acc == val_acc:
                    msg += f" val_acc={val_acc:.4f}"
            msg += f" | lr_bb={lr_bb:.2e} lr_hd={lr_hd:.2e}"
            print(msg)

            # 保存 last.pt（按间隔）
            if (epoch + 1) % max(1, args.save_interval) == 0:
                torch.save({'model': model.state_dict(), 'epoch': epoch + 1}, run_dir / 'last.pt')

            # 保存 best.pt（优先依据验证集最小 loss；若无验证集则用训练集）
            ref_loss = val_loss if (dl_val is not None and val_loss == val_loss) else train_loss
            ref_acc = val_acc if (dl_val is not None and val_acc == val_acc) else train_acc
            if ref_loss < best_loss:
                best_loss = ref_loss
                best_epoch = epoch + 1
                best_acc = ref_acc if ref_acc == ref_acc else best_acc
                torch.save({'model': model.state_dict(), 'epoch': best_epoch, 'best_loss': best_loss, 'best_acc': best_acc}, run_dir / 'best.pt')

            # 调度器步进
            scheduler.step()

    # 写 metrics.json（记录 best 基于谁）
    metrics = {
        'best_epoch': best_epoch,
        'best_loss': float(best_loss) if best_epoch > 0 else None,
        'best_acc': float(best_acc) if (best_epoch > 0 and best_acc == best_acc) else None,
        'has_val': bool(dl_val is not None),
        'run_dir': str(run_dir.resolve()),
    }
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
