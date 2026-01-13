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
WORKSPACE_ROOT = ROOT_DIR.parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.b3_fusion_model import B3FusionModel
from data.dataset_pair import PairFCSVTSNpzDataset


def set_seed(seed: int = 3407):
    try:
        import torch
        import numpy as np
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser()
    # 基本
    p.add_argument('--backbone_type', type=str, default='swin', choices=['swin', 'convnext'])
    p.add_argument('--embed_dim', type=int, default=512)
    p.add_argument('--num_classes', type=int, default=22)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--save_interval', type=int, default=1)

    # 数据与运行目录
    p.add_argument('--train_list', type=str, required=True)
    p.add_argument('--val_list', type=str, default=None)
    p.add_argument('--vts_list', type=str, required=True, help='VTS 列表（与 train_list 配对）')
    p.add_argument('--vts_val_list', type=str, default=None, help='验证集专用 VTS 列表（可选），未提供则复用 --vts_list')
    p.add_argument('--run_dir', type=str, default='', help='若为空则自动写入 CLS_0_200_ex1314/Run-B3-<timestamp>')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--stem_min', type=int, default=0)
    p.add_argument('--stem_max', type=int, default=200)

    # 预训练/骨干
    p.add_argument('--swin_name', type=str, default='swin_base_patch4_window7_224')
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--swin_checkpoint', type=str, default='')

    # 优化器/正则
    p.add_argument('--backbone_lr', type=float, default=3e-5)
    p.add_argument('--head_lr', type=float, default=3e-3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--freeze_backbone_epochs', type=int, default=5)
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--clip_grad', type=float, default=0)

    # B3 融合超参
    p.add_argument('--fusion_depth', type=int, default=1)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--attn_dropout', type=float, default=0.1)
    p.add_argument('--mlp_ratio', type=float, default=4.0)

    # 增强开关（与 B2 保持一致命名，数据集内部使用）
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

    # stats（默认 None，即不开启归一化）
    p.add_argument('--stats_fcs', type=str, default=None)
    p.add_argument('--stats_vts', type=str, default=None)

    # ========== 融合调度相关 ==========
    p.add_argument('--fusion_warmup_epochs', type=int, default=5, help='预热期，融合强度保持最小')
    p.add_argument('--fusion_ramp_epochs', type=int, default=15, help='融合强度线性从最小涨到1.0的轮数')
    p.add_argument('--vts_freeze_epochs', type=int, default=2, help='前多少个 epoch 冻结 VTS 编码器')
    p.add_argument('--init_fusion_scale', type=float, default=1e-3, help='初始融合强度，用于第0轮设置')

    # ========== 早停（科研常用） ==========
    p.add_argument('--early_stop', action='store_true', help='启用早停')
    p.add_argument('--early_stop_metric', type=str, default='val_loss', choices=['val_loss', 'val_acc'], help='早停监控指标')
    p.add_argument('--early_stop_patience', type=int, default=8, help='早停耐心轮数')
    p.add_argument('--early_stop_min_delta', type=float, default=0.0, help='早停最小改进幅度')

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
    ds_train = PairFCSVTSNpzDataset(
        fcs_list_file=args.train_list,
        vts_list_file=args.vts_list,
        stats_fcs=_load_stats(args.stats_fcs),
        stats_vts=_load_stats(args.stats_vts),
        stem_min=args.stem_min,
        stem_max=args.stem_max,
        is_train=True,
    )
    ds_val = None
    if args.val_list:
        ds_val = PairFCSVTSNpzDataset(
            fcs_list_file=args.val_list,
            vts_list_file=(args.vts_val_list if args.vts_val_list else args.vts_list),
            stats_fcs=_load_stats(args.stats_fcs),
            stats_vts=_load_stats(args.stats_vts),
            stem_min=args.stem_min,
            stem_max=args.stem_max,
            is_train=False,
        )

    def _worker_init_fn(worker_id: int):
        base_seed = args.seed
        seed = (base_seed + worker_id) % (2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=_worker_init_fn, generator=g)
    dl_val = None
    if ds_val is not None:
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, worker_init_fn=_worker_init_fn, generator=g)
    return dl_train, dl_val


def build_model(args, num_classes: int):
    model = B3FusionModel(
        num_classes=num_classes,
        backbone_type=args.backbone_type,
        embed_dim=args.embed_dim,
        dropout=0.1,
        fusion_depth=args.fusion_depth,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout,
        mlp_ratio=args.mlp_ratio,
        swin_name=args.swin_name,
        pretrained=args.pretrained,
        swin_checkpoint=args.swin_checkpoint,
    )
    return model


# 通用 batch 运行
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _forward_and_loss(model, batch, criterion, device):
    fcs, vts, vts_valid, labels = batch
    fcs = fcs.to(device, non_blocking=True)
    vts = vts.to(device, non_blocking=True)
    vts_valid = vts_valid.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True).long()
    logits = model(fcs, vts, vts_valid)
    loss = criterion(logits, labels)
    return loss, logits, labels


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
        for it, batch in iterator:
            self.optimizer.zero_grad(set_to_none=True)
            if self.amp:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    loss, logits, labels = _forward_and_loss(self.model, batch, self.criterion, self.device)
                self.scaler.scale(loss).backward()
                if self.clip_grad is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, logits, labels = _forward_and_loss(self.model, batch, self.criterion, self.device)
                loss.backward()
                if self.clip_grad is not None:
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
            bs = labels.size(0)
            sum_loss += float(loss) * bs
            sum_correct += int(correct)
            total += bs
            if tqdm and iterator is not None:
                iterator.set_postfix(loss=float(loss), acc=correct / max(1, bs))
            self.global_step += 1
        return (sum_loss / max(1, total), sum_correct / max(1, total))


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    if val_loader is None:
        return float('nan'), float('nan')
    with torch.no_grad():
        for batch in val_loader:
            loss, logits, labels = _forward_and_loss(model, batch, criterion, device)
            preds = logits.argmax(dim=1)
            total_loss += float(loss) * labels.size(0)
            total_correct += int((preds == labels).sum().item())
            total_samples += int(labels.size(0))
    if total_samples == 0:
        return float('nan'), float('nan')
    return total_loss / total_samples, total_correct / total_samples


# ========== 早停实现 ==========

class EarlyStopping:
    def __init__(self, patience: int = 8, mode: str = 'min', min_delta: float = 0.0):
        self.patience = int(max(1, patience))
        self.mode = mode  # 'min' or 'max'
        self.min_delta = float(min_delta)
        self.best = None
        self.num_bad = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = False
        if self.best is None:
            improved = True
        else:
            if self.mode == 'min':
                improved = value < (self.best - self.min_delta)
            else:
                improved = value > (self.best + self.min_delta)
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
        return improved


# ========== 融合强度调度函数 ==========

def get_fusion_scale_schedule(epoch: int, warmup_epochs: int = 5, ramp_epochs: int = 15, min_scale: float = 1e-3) -> float:
    if epoch < warmup_epochs:
        return float(min_scale)
    elif epoch < warmup_epochs + ramp_epochs:
        progress = (epoch - warmup_epochs) / max(1, ramp_epochs)
        return float(min_scale + progress * (1.0 - min_scale))
    else:
        return 1.0


def main():
    args = parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 数据与加载器
    dl_train, dl_val = build_dataloaders(args)

    # 模型与优化器
    model = build_model(args, num_classes=args.num_classes)

    # 初始融合强度
    if hasattr(model, 'set_fusion_scale'):
        model.set_fusion_scale(args.init_fusion_scale)

    # ===== 参数分组：分别为 FCS/VTS 主干与其余(head) =====
    prefix_fcs = 'encoder_fcs.backbone.'
    prefix_vts = 'encoder_vts.backbone.'
    fcs_backbone_params, vts_backbone_params, head_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(prefix_fcs):
            fcs_backbone_params.append(p)
        elif name.startswith(prefix_vts):
            vts_backbone_params.append(p)
        else:
            head_params.append(p)
    param_groups = []
    if fcs_backbone_params:
        param_groups.append({'params': fcs_backbone_params, 'lr': args.backbone_lr, 'weight_decay': args.weight_decay})
    if vts_backbone_params:
        # 与 FCS 主干相同的低学习率，避免解冻时爆炸
        param_groups.append({'params': vts_backbone_params, 'lr': args.backbone_lr, 'weight_decay': args.weight_decay})
    if head_params:
        param_groups.append({'params': head_params, 'lr': args.head_lr, 'weight_decay': args.weight_decay})
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    # 运行目录
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_root = WORKSPACE_ROOT / 'Model' / 'DSFNet-Output' / 'CLS_0_200_ex1314'
        ts = time.strftime('%Y%m%d-%H%M')
        run_dir = Path(run_root) / f'Run-B3-{ts}'
        run_dir.mkdir(parents=True, exist_ok=True)

    # 写 config.yaml（精简）
    cfg = {
        'data': {
            'train_list': str(Path(args.train_list).resolve()),
            'val_list': (str(Path(args.val_list).resolve()) if args.val_list else ''),
            'vts_list': (str(Path(args.vts_list).resolve()) if args.vts_list else ''),
            'vts_val_list': (str(Path(args.vts_val_list).resolve()) if args.vts_val_list else ''),
            'stats_fcs': args.stats_fcs or '',
            'stats_vts': args.stats_vts or '',
            'num_classes': args.num_classes,
        },
        'model': {
            'name': 'B3FusionModel',
            'backbone_type': args.backbone_type,
            'embed_dim': args.embed_dim,
            'fusion_depth': args.fusion_depth,
            'num_heads': args.num_heads,
            'attn_dropout': args.attn_dropout,
            'mlp_ratio': args.mlp_ratio,
            'init_fusion_scale': args.init_fusion_scale,
        },
        'train': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'amp': bool(args.amp),
            'num_workers': args.num_workers,
            'save_interval': args.save_interval,
            'backbone_lr': args.backbone_lr,
            'head_lr': args.head_lr,
            'fusion_warmup_epochs': args.fusion_warmup_epochs,
            'fusion_ramp_epochs': args.fusion_ramp_epochs,
            'vts_freeze_epochs': args.vts_freeze_epochs,
            'early_stop': bool(args.early_stop),
            'early_stop_metric': args.early_stop_metric,
            'early_stop_patience': args.early_stop_patience,
            'early_stop_min_delta': args.early_stop_min_delta,
        },
        'run_dir': str(run_dir.resolve()),
    }
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')

    # 数据一致性自检
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
        for pth in paths:
            try:
                c = int(Path(pth).parent.name)
                d[c] = d.get(c, 0) + 1
            except Exception:
                continue
        return dict(sorted(d.items()))

    train_fcs_paths = _read_list(args.train_list)
    train_vts_paths = _read_list(args.vts_list) if args.vts_list else []
    val_fcs_paths = _read_list(args.val_list) if args.val_list else []
    val_vts_paths = _read_list(args.vts_val_list) if args.vts_val_list else (_read_list(args.vts_list) if args.vts_list else [])

    stems_train_fcs = {s for s in (_stem_of(p) for p in train_fcs_paths) if s is not None}
    stems_val_fcs = {s for s in (_stem_of(p) for p in val_fcs_paths) if s is not None}
    stems_train_vts = {s for s in (_stem_of(p) for p in train_vts_paths) if s is not None}
    stems_val_vts = {s for s in (_stem_of(p) for p in val_vts_paths) if s is not None}

    inter_fcs = stems_train_fcs & stems_val_fcs
    inter_vts = stems_train_vts & stems_val_vts

    sanity = {
        'train': {
            'fcs_count': len(train_fcs_paths),
            'vts_count': len(train_vts_paths),
            'stems_fcs': len(stems_train_fcs),
            'stems_vts': len(stems_train_vts),
            'class_counts_fcs': _count_by_class(train_fcs_paths),
            'class_counts_vts': _count_by_class(train_vts_paths),
        },
        'val': {
            'fcs_count': len(val_fcs_paths),
            'vts_count': len(val_vts_paths),
            'stems_fcs': len(stems_val_fcs),
            'stems_vts': len(stems_val_vts),
            'class_counts_fcs': _count_by_class(val_fcs_paths),
            'class_counts_vts': _count_by_class(val_vts_paths),
        },
        'overlap': {
            'stems_intersection_fcs': len(inter_fcs),
            'stems_intersection_vts': len(inter_vts),
        }
    }
    (run_dir / 'data_sanity.json').write_text(json.dumps(sanity, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[Sanity] stems(FCS) train={len(stems_train_fcs)} val={len(stems_val_fcs)} inter={len(inter_fcs)}")
    print(f"[Sanity] stems(VTS) train={len(stems_train_vts)} val={len(stems_val_vts)} inter={len(inter_vts)}")

    # 训练器与损失
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    trainer = Trainer(model, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu', amp=args.amp, criterion=criterion, clip_grad=args.clip_grad)

    # 早停对象
    early_stopper = None
    early_mode = 'min' if args.early_stop_metric.endswith('loss') else 'max'
    if args.early_stop:
        early_stopper = EarlyStopping(patience=args.early_stop_patience, mode=early_mode, min_delta=args.early_stop_min_delta)

    log_csv = run_dir / 'log.csv'
    with log_csv.open('w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr_backbone', 'lr_head'])
        best_loss = float('inf')
        best_epoch = -1
        best_acc = 0.0
        early_stop_epoch = None
        for epoch in range(args.epochs):
            # ========== 融合强度调度与 VTS 冻结 ==========
            fusion_scale = get_fusion_scale_schedule(epoch, args.fusion_warmup_epochs, args.fusion_ramp_epochs, min_scale=args.init_fusion_scale)
            if hasattr(model, 'set_fusion_scale'):
                model.set_fusion_scale(fusion_scale)
            if hasattr(model, 'freeze_vts_encoder') and hasattr(model, 'unfreeze_vts_encoder'):
                if epoch < max(0, args.vts_freeze_epochs):
                    model.freeze_vts_encoder()
                    vts_frozen = True
                else:
                    model.unfreeze_vts_encoder()
                    vts_frozen = False
            else:
                vts_frozen = False

            # 按需冻结/解冻主干（FCS）
            freeze = (epoch < max(0, args.freeze_backbone_epochs))
            if hasattr(model, 'encoder_fcs') and hasattr(model.encoder_fcs, 'backbone'):
                for p in model.encoder_fcs.backbone.parameters():
                    p.requires_grad = not freeze

            # 记录学习率（考虑三组：FCS/VTS backbone 与 head）
            def _get_lr(i, default):
                return optimizer.param_groups[i]['lr'] if len(optimizer.param_groups) > i else default
            lr_bb_fcs = _get_lr(0, args.backbone_lr)
            lr_bb_vts = _get_lr(1, args.backbone_lr) if len(optimizer.param_groups) >= 2 and vts_backbone_params else lr_bb_fcs
            lr_hd = _get_lr(2 if len(optimizer.param_groups) > 2 else (1 if len(optimizer.param_groups) > 1 and not vts_backbone_params else 0), args.head_lr)

            train_loss, train_acc = trainer.fit_one_epoch(dl_train)
            val_loss, val_acc = evaluate(model, dl_val, criterion, trainer.device) if dl_val is not None else (float('nan'), float('nan'))

            writer.writerow([
                epoch + 1,
                f"{train_loss:.6f}",
                (f"{train_acc:.6f}" if train_acc == train_acc else ""),
                (f"{val_loss:.6f}" if val_loss == val_loss else ""),
                (f"{val_acc:.6f}" if val_acc == val_acc else ""),
                f"{lr_bb_fcs:.6e}", f"{lr_hd:.6e}"
            ])
            fcsv.flush()
            msg = f"Epoch {epoch+1}/{args.epochs} - loss={train_loss:.4f}"
            if train_acc == train_acc:
                msg += f" acc={train_acc:.4f}"
            if dl_val is not None and val_loss == val_loss:
                msg += f" | val_loss={val_loss:.4f}"
                if val_acc == val_acc:
                    msg += f" val_acc={val_acc:.4f}"
            msg += f" | lr_fcs={lr_bb_fcs:.2e} lr_vts={lr_bb_vts:.2e} lr_hd={lr_hd:.2e}"
            msg += f" | fusion_scale={fusion_scale:.4f} vts_frozen={vts_frozen}"
            print(msg)

            # 保存 last.pt（按间隔）
            if (epoch + 1) % max(1, args.save_interval) == 0:
                torch.save({'model': model.state_dict(), 'epoch': epoch + 1}, run_dir / 'last.pt')

            # 保存 best.pt
            ref_loss = val_loss if (dl_val is not None and val_loss == val_loss) else train_loss
            ref_acc = val_acc if (dl_val is not None and val_acc == val_acc) else train_acc
            if ref_loss < best_loss:
                best_loss = ref_loss
                best_epoch = epoch + 1
                best_acc = ref_acc if ref_acc == ref_acc else best_acc
                torch.save({'model': model.state_dict(), 'epoch': best_epoch, 'best_loss': best_loss, 'best_acc': best_acc}, run_dir / 'best.pt')

            # 早停逻辑
            if early_stopper is not None:
                monitor_value = ref_loss if args.early_stop_metric == 'val_loss' else (ref_acc if ref_acc == ref_acc else float('nan'))
                if monitor_value == monitor_value:  # 不是 NaN
                    early_stopper.step(float(monitor_value))
                    if early_stopper.should_stop:
                        early_stop_epoch = epoch + 1
                        print(f"[EarlyStop] triggered at epoch {early_stop_epoch}")
                        break

            scheduler.step()

    metrics = {
        'best_epoch': best_epoch,
        'best_loss': float(best_loss) if best_epoch > 0 else None,
        'best_acc': float(best_acc) if (best_epoch > 0 and best_acc == best_acc) else None,
        'has_val': bool(dl_val is not None),
        'run_dir': str(run_dir.resolve()),
        'early_stop': bool(args.early_stop),
        'early_stop_epoch': int(early_stop_epoch) if early_stop_epoch is not None else None,
        'early_stop_metric': (args.early_stop_metric if args.early_stop else ''),
    }
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
