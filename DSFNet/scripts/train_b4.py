import argparse
import json
from pathlib import Path
import sys
import time
import csv
import yaml
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils as nn_utils

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
WORKSPACE_ROOT = ROOT_DIR.parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.b4_fusion_model import B4FusionModel
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
    p.add_argument('--vts_list', type=str, required=True)
    p.add_argument('--vts_val_list', type=str, default=None)
    p.add_argument('--run_dir', type=str, default='', help='若为空则自动写入 CLS_0_200_ex1314/Run-B4-<timestamp>')
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

    # B4 融合超参
    p.add_argument('--fusion_depth', type=int, default=1)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--attn_dropout', type=float, default=0.1)
    p.add_argument('--mlp_ratio', type=float, default=4.0)

    # 辅助损失
    p.add_argument('--align_loss_weight', type=float, default=0.05)
    p.add_argument('--consistency_loss_weight', type=float, default=0.02)
    p.add_argument('--contrastive_temperature', type=float, default=0.07)

    # 增强（与 B2/B3 名称一致）
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

    # stats
    p.add_argument('--stats_fcs', type=str, default=None)
    p.add_argument('--stats_vts', type=str, default=None)

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
    model = B4FusionModel(
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


# Losses
def cosine_similarity_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE 风格的对齐损失（批内正对为配对）。"""
    z1 = torch.nn.functional.normalize(z1, dim=-1)
    z2 = torch.nn.functional.normalize(z2, dim=-1)
    logits = (z1 @ z2.t()) / max(1e-6, float(temperature))  # [B,B]
    targets = torch.arange(z1.size(0), device=z1.device)
    return torch.nn.functional.cross_entropy(logits, targets)


def kl_divergence_loss(p_logits: torch.Tensor, q_logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """一致性蒸馏损失：softmax(p/T) 与 softmax(q/T) 的 KL。"""
    p = torch.nn.functional.log_softmax(p_logits / T, dim=-1)
    q = torch.nn.functional.softmax(q_logits / T, dim=-1)
    return torch.nn.functional.kl_div(p, q, reduction='batchmean') * (T * T)


# 训练器
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _forward_and_losses(model, batch, criterion, device, align_w: float, cons_w: float, temp: float):
    fcs, vts, vts_valid, labels = batch
    fcs = fcs.to(device, non_blocking=True)
    vts = vts.to(device, non_blocking=True)
    vts_valid = vts_valid.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True).long()

    # 主分类
    logits = model(fcs, vts, vts_valid)
    loss_cls = criterion(logits, labels)

    # 取中间特征做辅助损失
    feats = model.extract_feats(fcs, vts, vts_valid)
    f_feat, v_feat = feats['f_feat'], feats['v_feat']
    fused_xattn, concat_feat = feats['fused_xattn'], feats['concat_feat']

    loss_align = cosine_similarity_loss(f_feat, v_feat, temperature=temp) if align_w > 0 else torch.tensor(0.0, device=device)

    # 一致性：让 fused_xattn 与 concat_feat 的预测接近
    if cons_w > 0:
        head = model.head
        p_logits = head(torch.cat([fused_xattn, model.presence_embed(feats['presence'])], dim=-1))
        q_logits = head(torch.cat([concat_feat, model.presence_embed(feats['presence'])], dim=-1))
        loss_cons = kl_divergence_loss(p_logits, q_logits, T=1.0)
    else:
        loss_cons = torch.tensor(0.0, device=device)

    loss = loss_cls + align_w * loss_align + cons_w * loss_cons
    return loss, logits, labels, {
        'loss_cls': float(loss_cls.detach()),
        'loss_align': float(loss_align.detach()),
        'loss_cons': float(loss_cons.detach()),
    }


class Trainer:
    def __init__(self, model, optimizer, device: str = 'cuda', amp: bool = True, criterion: torch.nn.Module | None = None, clip_grad: float | None = 1.0,
                 align_w: float = 0.05, cons_w: float = 0.02, temp: float = 0.07):
        use_cuda = (device == 'cuda' and torch.cuda.is_available())
        self.device = torch.device('cuda') if use_cuda else torch.device('cpu')
        self.amp = amp and use_cuda
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else torch.nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp) if self.amp else None
        self.global_step = 0
        self.clip_grad = float(clip_grad) if clip_grad and clip_grad > 0 else None
        self.align_w = align_w
        self.cons_w = cons_w
        self.temp = temp

    def fit_one_epoch(self, loader):
        self.model.train()
        total = 0
        sum_loss = 0.0
        sum_correct = 0
        sum_cls = 0.0
        sum_align = 0.0
        sum_cons = 0.0
        iterator = tqdm(enumerate(loader), total=len(loader), desc="Training", leave=False) if tqdm else enumerate(loader)
        for it, batch in iterator:
            self.optimizer.zero_grad(set_to_none=True)
            if self.amp:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    loss, logits, labels, parts = _forward_and_losses(self.model, batch, self.criterion, self.device, self.align_w, self.cons_w, self.temp)
                self.scaler.scale(loss).backward()
                if self.clip_grad is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, logits, labels, parts = _forward_and_losses(self.model, batch, self.criterion, self.device, self.align_w, self.cons_w, self.temp)
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
            sum_cls += parts['loss_cls'] * bs
            sum_align += parts['loss_align'] * bs
            sum_cons += parts['loss_cons'] * bs
            total += bs
            if tqdm and iterator is not None:
                iterator.set_postfix(loss=float(loss), acc=correct / max(1, bs))
            self.global_step += 1
        return (sum_loss / max(1, total), sum_correct / max(1, total), sum_cls / max(1, total), sum_align / max(1, total), sum_cons / max(1, total))


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    if val_loader is None:
        return float('nan'), float('nan')
    with torch.no_grad():
        for batch in val_loader:
            fcs, vts, vts_valid, labels = batch
            fcs = fcs.to(device, non_blocking=True)
            vts = vts.to(device, non_blocking=True)
            vts_valid = vts_valid.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            logits = model(fcs, vts, vts_valid)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += float(loss) * labels.size(0)
            total_correct += int((preds == labels).sum().item())
            total_samples += int(labels.size(0))
    if total_samples == 0:
        return float('nan'), float('nan')
    return total_loss / total_samples, total_correct / total_samples


def main():
    args = parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 数据与加载器
    dl_train, dl_val = build_dataloaders(args)

    # 模型与优化器
    model = build_model(args, num_classes=args.num_classes)

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
        run_dir = Path(run_root) / f'Run-B4-{ts}'
        run_dir.mkdir(parents=True, exist_ok=True)

    # 写 config.yaml
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
            'name': 'B4FusionModel',
            'backbone_type': args.backbone_type,
            'embed_dim': args.embed_dim,
            'fusion_depth': args.fusion_depth,
            'num_heads': args.num_heads,
            'attn_dropout': args.attn_dropout,
            'mlp_ratio': args.mlp_ratio,
            'align_w': args.align_loss_weight,
            'cons_w': args.consistency_loss_weight,
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
        },
        'run_dir': str(run_dir.resolve()),
    }
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')

    # 数据一致性自检（同 B2/B3）
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
    trainer = Trainer(
        model, optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        amp=args.amp,
        criterion=criterion,
        clip_grad=args.clip_grad,
        align_w=args.align_loss_weight,
        cons_w=args.consistency_loss_weight,
        temp=args.contrastive_temperature,
    )

    log_csv = run_dir / 'log.csv'
    with log_csv.open('w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr_backbone', 'lr_head', 'loss_cls', 'loss_align', 'loss_cons'])
        best_loss = float('inf')
        best_epoch = -1
        best_acc = 0.0
        for epoch in range(args.epochs):
            # 冻结/解冻主干
            freeze = (epoch < max(0, args.freeze_backbone_epochs))
            if hasattr(model, 'encoder_fcs') and hasattr(model.encoder_fcs, 'backbone'):
                for p in model.encoder_fcs.backbone.parameters():
                    p.requires_grad = not freeze

            # 记录学习率
            lr_bb = optimizer.param_groups[0]['lr'] if optimizer.param_groups else args.backbone_lr
            lr_hd = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else args.head_lr

            train_loss, train_acc, loss_cls, loss_align, loss_cons = trainer.fit_one_epoch(dl_train)
            val_loss, val_acc = evaluate(model, dl_val, criterion, trainer.device) if dl_val is not None else (float('nan'), float('nan'))

            writer.writerow([
                epoch + 1,
                f"{train_loss:.6f}",
                (f"{train_acc:.6f}" if train_acc == train_acc else ""),
                (f"{val_loss:.6f}" if val_loss == val_loss else ""),
                (f"{val_acc:.6f}" if val_acc == val_acc else ""),
                f"{lr_bb:.6e}", f"{lr_hd:.6e}",
                f"{loss_cls:.6f}", f"{loss_align:.6f}", f"{loss_cons:.6f}"
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
            msg += f" | parts: cls={loss_cls:.4f} align={loss_align:.4f} cons={loss_cons:.4f}"
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

            scheduler.step()

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
