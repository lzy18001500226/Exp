import argparse
import json
import os
import random
import time
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.cuda.amp import autocast, GradScaler
# 新增：进度条
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

import sys
FILE_DIR = os.path.dirname(__file__)
if FILE_DIR not in sys.path:
    sys.path.append(FILE_DIR)

from dataset import create_dataloaders, DroneRFDataset
from model import build_model


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN/tf32 设置
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = False if deterministic else True
        torch.backends.cudnn.allow_tf32 = False if deterministic else True
    except Exception:
        pass
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


# 新增：通用部分加载函数
def load_state_dict_partial(module: nn.Module, state_dict: Dict[str, torch.Tensor], prefix: str = '') -> Dict[str, List[str]]:
    """Load with relaxed matching: key must exist in module.state_dict and tensor shape must match.
    Returns {'loaded': [...], 'skipped': [...]} lists of parameter names.
    """
    msd = module.state_dict()
    loaded, skipped = [], []
    with torch.no_grad():
        for k, v in state_dict.items():
            mk = k if prefix == '' else (prefix + k)
            if mk in msd and msd[mk].shape == v.shape:
                msd[mk].copy_(v)
                loaded.append(mk)
            else:
                skipped.append(mk)
    module.load_state_dict(msd)
    return {'loaded': loaded, 'skipped': skipped}


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def calibrate_gamma(energies: np.ndarray, alpha: float = 0.05) -> float:
    """Gamma = (1-alpha)-quantile of known validation energies."""
    alpha = float(alpha)
    alpha = min(max(alpha, 0.0), 0.5)
    q = np.quantile(energies, 1.0 - alpha)
    return float(q)


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    top1_sum, top5_sum, n = 0.0, 0.0, 0
    all_energy: List[float] = []
    with torch.no_grad():
        for x_tex, x_pos, y in loader:
            x_tex = x_tex.to(device, non_blocking=True)
            x_pos = x_pos.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, energy = model(x_tex, x_pos)
            acc1, acc5 = accuracy(logits, y, topk=(1, 5))
            bs = y.size(0)
            top1_sum += acc1.item() * bs / 100.0
            top5_sum += acc5.item() * bs / 100.0
            n += bs
            all_energy.extend(energy.detach().cpu().tolist())
    return {
        "top1": 100.0 * top1_sum / max(n, 1),
        "top5": 100.0 * top5_sum / max(n, 1),
        "n": float(n),
        "energy_mean": float(np.mean(all_energy) if all_energy else 0.0),
        "energy_list_count": float(len(all_energy)),
    }


def eval_open_set(model: nn.Module, known_loader, unknown_loader, gamma: float, device: torch.device) -> Dict[str, float]:
    model.eval()
    # Known
    A_known, N_known, TP_known = 0, 0, 0
    with torch.no_grad():
        for x_tex, x_pos, y in known_loader:
            x_tex = x_tex.to(device, non_blocking=True)
            x_pos = x_pos.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, energy = model(x_tex, x_pos)
            accept_known = (energy < gamma)
            A_known += int(accept_known.sum().item())
            N_known += y.numel()
            # correct among accepted
            if accept_known.any():
                y_pred = logits.argmax(dim=1)
                TP_known += int(((y_pred == y) & accept_known).sum().item())
    # Unknown
    TN_unknown, N_unknown, FP_unknown = 0, 0, 0
    with torch.no_grad():
        for x_tex, x_pos, y in unknown_loader:
            x_tex = x_tex.to(device, non_blocking=True)
            x_pos = x_pos.to(device, non_blocking=True)
            # y is dummy for unknown; ignore labels
            logits, energy = model(x_tex, x_pos)
            reject = (energy >= gamma)
            TN_unknown += int(reject.sum().item())
            N_unknown += reject.numel()
            FP_unknown += int((~reject).sum().item())
    TKR = A_known / max(N_known, 1)
    TUR = TN_unknown / max(N_unknown, 1)
    KP = TP_known / max((TP_known + FP_unknown), 1)
    return {"TKR": 100.0 * TKR, "TUR": 100.0 * TUR, "KP": 100.0 * KP,
            "N_known": float(N_known), "N_unknown": float(N_unknown)}


# 新增：EMA 实现（简化版）
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        self.ema = self._clone_model(model).eval()
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _clone_model(self, model: nn.Module) -> nn.Module:
        import copy
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + msd[k].to(v.device, dtype=v.dtype) * (1.0 - d))


def train_one_epoch(model, loader, criterion, optimizer, device, scaler: Optional[GradScaler] = None, clip_grad: float = 0.0, ema: Optional[ModelEMA] = None):
    model.train()
    loss_sum, top1_sum, top5_sum, n = 0.0, 0.0, 0.0, 0
    total_batches = len(loader)
    it = loader if tqdm is None else tqdm(loader, total=total_batches, desc="Train", ncols=100)
    for i, (x_tex, x_pos, y) in enumerate(it):
        x_tex = x_tex.to(device, non_blocking=True)
        x_pos = x_pos.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            with autocast():
                logits, _ = model(x_tex, x_pos)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if clip_grad and clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(x_tex, x_pos)
            loss = criterion(logits, y)
            loss.backward()
            if clip_grad and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        if ema is not None:
            ema.update(model)
        acc1, acc5 = accuracy(logits, y, topk=(1, 5))
        bs = y.size(0)
        loss_sum += loss.item() * bs
        top1_sum += acc1.item() * bs / 100.0
        top5_sum += acc5.item() * bs / 100.0
        n += bs
        # 更新进度显示
        avg_loss = loss_sum / max(n, 1)
        avg_top1 = 100.0 * top1_sum / max(n, 1)
        if tqdm is not None:
            it.set_postfix({"loss": f"{avg_loss:.4f}", "top1": f"{avg_top1:.2f}"})
        else:
            # 无 tqdm 时，每 10% 进度打印一次
            if (i + 1) % max(1, total_batches // 10) == 0 or (i + 1) == total_batches:
                print(f"  [Train] {i+1}/{total_batches} | loss {avg_loss:.4f} top1 {avg_top1:.2f}")
    if tqdm is not None and hasattr(it, "close"):
        it.close()
    return {
        "loss": loss_sum / max(n, 1),
        "top1": 100.0 * top1_sum / max(n, 1),
        "top5": 100.0 * top5_sum / max(n, 1),
    }


# 新增：参数统计
def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def summarize_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(
        num_classes=args.num_classes,
        use_swin=args.use_swin,
        out_chs=tuple(args.out_chs),
        swin_heads=args.swin_heads,
        swin_layers=args.swin_layers,
        pos_backbone=args.pos_backbone,
        align_resize=args.align_resize,
    ).to(device)

    shapes: Dict[str, Tuple[int, ...]] = {}

    def reg(name: str, module: nn.Module, pre: bool = False):
        if pre:
            module.register_forward_pre_hook(lambda m, inp: shapes.__setitem__(name, tuple(inp[0].shape)))
        else:
            module.register_forward_hook(lambda m, inp, out: shapes.__setitem__(name, tuple(out.shape)))

    # 注册关键节点
    if hasattr(model, 'tex'):
        reg('tex.p3', model.tex.stage3)
        reg('tex.p4', model.tex.stage4)
        reg('tex.p5', model.tex.stage5)
    if hasattr(model, 'pos'):
        # 位置流：若是 MobileNetV3，则 stage3/4/5 不存在，跳过；只看 neck 前融合层
        try:
            reg('pos.p3', model.pos.stage3)
            reg('pos.p4', model.pos.stage4)
            reg('pos.p5', model.pos.stage5)
        except Exception:
            pass
    if hasattr(model, 'neck'):
        reg('neck.align3', model.neck.align3)
        reg('neck.align4', model.neck.align4)
        reg('neck.align5', model.neck.align5)
    if hasattr(model, 'swin'):
        reg('f5.pre_swin', model.swin, pre=True)
        reg('f5.post_swin', model.swin)
    if hasattr(model, 'pool'):
        reg('head.pool_in', model.pool, pre=True)
        reg('head.pool_out', model.pool)

    # dummy 输入（谱图已归一化为 3 通道）
    x_tex = torch.randn(1, 3, 256, 256, device=device)
    x_pos = torch.randn(1, 3, 256, 256, device=device)
    model.eval()
    with torch.no_grad():
        logits, energy = model(x_tex, x_pos)

    # 打印形状与参数量
    print('==== Model Summary ====')
    print(f"use_swin={args.use_swin} | out_chs={tuple(args.out_chs)} | swin_heads={args.swin_heads} | swin_layers={args.swin_layers} | pos_backbone={args.pos_backbone} | align_resize={args.align_resize}")
    for k in sorted(shapes.keys()):
        print(f"{k:>16}: {shapes[k]}")
    # 参数量
    total = count_params(model)
    parts = {
        'tex': count_params(model.tex) if hasattr(model, 'tex') else 0,
        'pos': count_params(model.pos) if hasattr(model, 'pos') else 0,
        'neck': count_params(model.neck) if hasattr(model, 'neck') else 0,
        'swin': count_params(model.swin) if hasattr(model, 'swin') else 0,
        'head': count_params(model.fc) + count_params(model.pool) if hasattr(model, 'fc') else 0,
    }
    print('---- Params (trainable elements) ----')
    for n, v in parts.items():
        print(f"{n:>6}: {v/1e6:.2f}M")
    print(f"TOTAL: {total/1e6:.2f}M")
    print('====================================')


def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--train_list', type=str, default='./experiment_groups/1-known_for_train')
    parser.add_argument('--val_list', type=str, default='./experiment_groups/2-known_for_val')
    parser.add_argument('--test_list', type=str, default='./experiment_groups/3-known_for_test')
    parser.add_argument('--interference_pool', type=str, default='', help='path to interference pool dir (optional)')
    parser.add_argument('--unknown_val_list', type=str, default='')
    parser.add_argument('--unknown_test_list', type=str, default='')
    parser.add_argument('--data_root', type=str, default='C:/Users/HP/Desktop/Exp/Data', help='root dir of Data for resolving relative paths in lists')
    parser.add_argument('--limit_train', type=int, default=0, help='limit number of training samples (0=unlimited)')
    parser.add_argument('--limit_eval', type=int, default=0, help='limit number of val/test samples each (0=unlimited)')
    # model
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--use_swin', action='store_true', help='Enable Method2 (Swin on F5)')
    # 新：三尺度输出通道，默认 [192,384,768]
    parser.add_argument('--out_chs', type=int, nargs=3, default=[192,384,768])
    parser.add_argument('--swin_heads', type=int, default=24)
    parser.add_argument('--swin_layers', type=int, default=2)
    # 新增：Swin 与 Position 权重/变体
    parser.add_argument('--swin_weights', type=str, default='', help='path to Swin *.pth (e.g., swin_small_patch4_window7_224.pth)')
    parser.add_argument('--swin_stage4_weights', type=str, default='', help='path to Swin-Small/Base *.pth for Stage4 adapter')
    parser.add_argument('--pos_backbone', type=str, default='dwconv', choices=['dwconv','mobilenetv3'])
    parser.add_argument('--pos_weights', type=str, default='', help='path to MobileNetV3 *.pth if pos_backbone=mobilenetv3')
    parser.add_argument('--align_resize', action='store_true', help='if set, allow emergency nearest resize to align P3/P4/P5 between streams')
    # train
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing')
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--freeze_epochs', type=int, default=5, help='freeze pos backbone and swin for first N epochs')
    parser.add_argument('--deterministic', action='store_true', help='enable deterministic training for reproducibility')
    parser.add_argument('--clip_grad', type=float, default=0.0, help='max grad norm (0 to disable)')
    parser.add_argument('--ema', action='store_true', help='enable EMA for model weights')
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--warmup_epochs', type=int, default=3, help='linear warmup epochs before main scheduler')
    # open-set
    parser.add_argument('--alpha', type=float, default=0.05, help='known false-reject rate for gamma calibration')
    # misc
    parser.add_argument('--out_dir', type=str, default='', help='output dir for classification checkpoints (default: <project_root>/checkpoints')
    parser.add_argument('--summary', action='store_true', help='print model summary and exit')
    # 新：断言两流同尺度空间尺寸一致
    parser.add_argument('--assert_shapes', action='store_true', help='assert P3/P4/P5 HxW match between texture and position streams')
    # 新：仅探测权重映射并退出（不读数据集）
    parser.add_argument('--probe_weights', action='store_true', help='build model and try loading pos/swin weights, print stats then exit')
    # 新：自动推断类别数
    parser.add_argument('--auto_num_classes', action='store_true', help='infer num_classes from train_list before building model')
    # 新：恢复训练与结果CSV
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume (e.g., last.pt)')
    parser.add_argument('--results_csv', type=str, default='', help='results csv path (default: <out_dir>/results_*.csv)')

    args = parser.parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    # 统一输出目录：Exp/Model/Output/50标签训练X（X 自增）
    if not args.out_dir or args.out_dir.strip() == '':
        project_root = Path(FILE_DIR).parent.parent  # .../TSFNet/src -> .../Exp
        base_out = project_root / 'Model' / 'Output'
        base_out.mkdir(parents=True, exist_ok=True)
        import re
        pat = re.compile(r'^50标签训练(\d+)$')
        max_idx = 0
        for d in base_out.iterdir():
            if d.is_dir():
                m = pat.match(d.name)
                if m:
                    try:
                        max_idx = max(max_idx, int(m.group(1)))
                    except Exception:
                        pass
        next_idx = max_idx + 1
        args.out_dir = str((base_out / f"50标签训练{next_idx}").resolve())
    os.makedirs(args.out_dir, exist_ok=True)

    # 若未显式提供，自动指向本地预训练目录 Model/Pretrain/*
    try:
        project_root = Path(FILE_DIR).parent.parent
        pretrain_root = project_root / 'Model' / 'Pretrain'
        bb_dir = pretrain_root / 'Backbone'
        swin_dir = pretrain_root / 'Swin'
        def pick_first(*cands: Path) -> str:
            for c in cands:
                if c and c.exists():
                    return str(c.resolve())
            return ''
        if (not args.pos_weights) and bb_dir.exists():
            cand_pos = pick_first(bb_dir / 'mobilenet_v3_large-8738ca79.pth')
            if cand_pos:
                args.pos_weights = cand_pos
        if (not args.swin_stage4_weights) and swin_dir.exists():
            cand_swin = pick_first(swin_dir / 'swin_small_patch4_window7_224.pth')
            if cand_swin:
                args.swin_stage4_weights = cand_swin
        if (not args.swin_weights) and swin_dir.exists():
            cand_swin2 = pick_first(swin_dir / 'swin_small_patch4_window7_224.pth')
            if cand_swin2:
                args.swin_weights = cand_swin2
    except Exception:
        pass

    if args.summary:
        summarize_model(args)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 若启用自动推断类别数，先扫描训练列表
    if args.auto_num_classes:
        def infer_num_classes(list_file: str, data_root: str) -> int:
            mx = -1
            dr = data_root
            try:
                with open(list_file, 'r') as f:
                    for line in f:
                        s = line.strip().strip('"').strip("'")
                        if not s:
                            continue
                        parts = s.split()
                        label = None
                        if len(parts) >= 2:
                            try:
                                label = int(parts[1])
                            except Exception:
                                label = None
                        if label is None:
                            # 从父目录名推断
                            p = parts[0].strip().strip('"').strip("'")
                            # 处理 ./Data 或 Data 相对路径
                            if p.startswith('./Data') or p.startswith('.\\Data') or p.startswith('Data'):
                                if p.startswith('./Data'):
                                    rest = p[len('./Data'):]
                                elif p.startswith('.\\Data'):
                                    rest = p[len('.\\Data'):]
                                elif p.startswith('Data'):
                                    rest = p[len('Data'):]
                                else:
                                    rest = p
                                rest = rest.lstrip('\\/').replace('/', os.sep).replace('\\', os.sep)
                                p = os.path.join(dr, rest)
                            parent = os.path.basename(os.path.dirname(p))
                            if parent.isdigit():
                                label = int(parent)
                            else:
                                # 再尝试文件名前缀数字
                                base = os.path.basename(p)
                                num = ''
                                for ch in base:
                                    if ch.isdigit():
                                        num += ch
                                    else:
                                        break
                                if num != '':
                                    label = int(num)
                        if label is None:
                            label = 0
                        mx = max(mx, int(label))
            except Exception:
                pass
            return (mx + 1) if mx >= 0 else max(args.num_classes, 1)
        inferred = infer_num_classes(getattr(args, 'train_list', ''), getattr(args, 'data_root', ''))
        if inferred != args.num_classes:
            print(f"[auto] num_classes inferred: {inferred} (was {args.num_classes})")
            args.num_classes = inferred

    # 先构建模型（便于仅断言/探测时不依赖数据）
    model = build_model(
        num_classes=args.num_classes,
        use_swin=args.use_swin,
        out_chs=tuple(args.out_chs),
        swin_heads=args.swin_heads,
        swin_layers=args.swin_layers,
        pos_backbone=args.pos_backbone,
        align_resize=args.align_resize,
    ).to(device)

    # 前置输出：关键路径提示
    try:
        last_alias = os.path.join(args.out_dir, 'last.pt')
        best_alias = os.path.join(args.out_dir, 'best.pt')
        print(f"[paths] out_dir={args.out_dir}")
        print(f"[paths] results_csv={args.results_csv}")
        print(f"[paths] last={last_alias}")
        print(f"[paths] best={best_alias}")
    except Exception:
        pass

    # 形状断言（首次小批次前向）
    if args.assert_shapes:
        model.eval()
        with torch.no_grad():
            x_tex = torch.randn(2,3,512,512, device=device)
            x_pos = torch.randn(2,3,512,512, device=device)
            p3_t, p4_t, p5_t = model.tex(x_tex)
            p3_p, p4_p, p5_p = model.pos(x_pos)
            for (ta, pa, name) in [(p3_t, p3_p, 'P3'), (p4_t, p4_p, 'P4'), (p5_t, p5_p, 'P5')]:
                assert ta.shape[-2:] == pa.shape[-2:], f"{name} spatial mismatch: tex {ta.shape[-2:]} vs pos {pa.shape[-2:]}"

        print('Assert shapes: OK')
        return

    # 加载 Position 权重（仅当选择 mobilenetv3）
    if args.pos_backbone == 'mobilenetv3' and args.pos_weights and os.path.exists(args.pos_weights):
        try:
            sd = torch.load(args.pos_weights, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            if isinstance(sd, dict):
                new_sd = {}
                for k, v in sd.items():
                    # 只保留 features.* 下的权重，去掉前缀以适配 self.m 的键
                    if k.startswith('features.'):
                        nk = k[len('features.'):]
                        new_sd[nk] = v
                # 将权重加载到 pos.m（而不是 pos 本体），键名直接匹配 m 的子模块
                target_module = getattr(model.pos, 'm', model.pos)
                report = load_state_dict_partial(target_module, new_sd, prefix='')
                print(f"[pos] loaded={len(report['loaded'])}, skipped={len(report['skipped'])}")
        except Exception as e:
            print(f"[pos] failed to load weights: {e}")

    # 加载 Swin 权重（Stage4 适配器优先）
    if args.use_swin and args.swin_stage4_weights and os.path.exists(args.swin_stage4_weights):
        try:
            sd = torch.load(args.swin_stage4_weights, map_location='cpu')
            if isinstance(sd, dict) and 'model' in sd:
                sd = sd['model']
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            if isinstance(sd, dict):
                mapped = {}
                for k, v in sd.items():
                    # 兼容 timm/官方多种命名：layers.3.* 或 stages.3.*
                    if k.startswith('layers.3.'):
                        base = k[len('layers.3.'):]
                    elif k.startswith('stages.3.'):
                        base = k[len('stages.3.'):]
                    else:
                        continue
                    # 相对位置偏置表和索引通常形状不一致，跳过
                    if 'relative_position' in base:
                        continue
                    # 直接映射到 stage.*
                    mapped['stage.' + base] = v
                cand = len(mapped)
                report = load_state_dict_partial(model.swin, mapped, prefix='')
                loaded_n = len(report['loaded'])
                print(f"[swin_stage4] candidates={cand}, loaded={loaded_n}, skipped={len(report['skipped'])}")
                if loaded_n:
                    print('[swin_stage4] sample loaded keys:')
                    for n in report['loaded'][:10]:
                        print('  ', n)
        except Exception as e:
            print(f"[swin_stage4] failed to load weights: {e}")
    elif args.use_swin and args.swin_weights and os.path.exists(args.swin_weights):
        try:
            sd = torch.load(args.swin_weights, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            if isinstance(sd, dict):
                report = load_state_dict_partial(model.swin, sd, prefix='')
                print(f"[swin] loaded={len(report['loaded'])}, skipped={len(report['skipped'])}")
        except Exception as e:
            print(f"[swin] failed to load weights: {e}")

    # 仅探测权重则退出
    if args.probe_weights:
        # 报告是否使用 timm BasicLayer
        use_timm = hasattr(model.swin, 'stage') and model.swin.__class__.__name__ == 'SwinStage4Adapter' and getattr(model.swin, 'stage', None) is not None
        print(f"[probe] swin_backend={'timm.BasicLayer' if use_timm else 'fallback-Transformer'} | params={sum(p.numel() for p in model.swin.parameters())/1e6:.2f}M")
        return

    # 参数分组（区分不同模块学习率）
    def param_groups(m: nn.Module):
        groups = []
        small_lr, mid_lr, big_lr = args.lr * 0.1, args.lr * 0.3, args.lr
        wd = args.weight_decay
        def add_group(params, lr):
            if not params:
                return
            norms, others = [], []
            for p in params:
                name = getattr(p, 'name', '')
                # 简化：通过维度判断 norm/bias（也可通过模块遍历更精确）
                if p.ndim == 1:
                    norms.append(p)
                else:
                    others.append(p)
            if others:
                groups.append({'params': others, 'lr': lr, 'weight_decay': wd})
            if norms:
                groups.append({'params': norms, 'lr': lr, 'weight_decay': 0.0})
        # 注：为保证单一调度器贯穿全程，这里始终将模块的所有参数加入优化器；
        # 冻结阶段通过 requires_grad=False 令其梯度为 None，从而不会被更新。
        if args.pos_backbone == 'mobilenetv3':
            add_group([p for p in m.pos.parameters()], small_lr)
        add_group([p for p in m.tex.parameters()], mid_lr)
        add_group([p for p in m.neck.parameters()], big_lr)
        if args.use_swin:
            add_group([p for p in m.swin.parameters()], mid_lr)
        head_params = []
        if hasattr(m, 'pool'):
            head_params += list(m.pool.parameters())
        if hasattr(m, 'fc'):
            head_params += list(m.fc.parameters())
        add_group(head_params, big_lr)
        return groups

    criterion = nn.CrossEntropyLoss()

    # 冻结策略：前 N 个 epoch 冻结位置流和 Swin
    def apply_freeze(m: nn.Module, freeze: bool):
        targets = []
        if args.pos_backbone == 'mobilenetv3':
            targets.append(m.pos)
        if args.use_swin:
            targets.append(m.swin)
        for t in targets:
            for p in t.parameters():
                p.requires_grad = not freeze

    # 初始按冻结状态设置 requires_grad，并构建优化器
    apply_freeze(model, freeze=(args.freeze_epochs > 0))
    optimizer = optim.AdamW(param_groups(model))

    # Warmup + 主调度器（贯穿整个训练过程）
    if args.cosine:
        main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        main_scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    warmup_epochs = max(0, int(args.warmup_epochs))
    if warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            return 1.0
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        warmup_scheduler = None

    scaler = GradScaler(enabled=args.amp)
    ema = ModelEMA(model, decay=args.ema_decay, device=device) if args.ema else None

    # 统一保存文件名标签（与 results_csv 一致）
    tag = ('m2' if args.use_swin else 'm1')

    # 设置 results_csv 的默认路径并在首次创建时写入表头
    if (not args.results_csv) or args.results_csv.strip() == '':
        args.results_csv = os.path.join(args.out_dir, f"results_{tag}.csv")
    try:
        if not os.path.exists(args.results_csv):
            with open(args.results_csv, 'w', encoding='utf-8', newline='') as f:
                # epoch 级：基础训练/验证；最终一行：补全 test/open-set/gamma/alpha
                f.write("epoch,lr,train_loss,train_top1,val_top1,val_top5,test_top1,test_top5,TKR_val,TUR_val,KP_val,TKR_test,TUR_test,KP_test,gamma,alpha\n")
    except Exception:
        pass

    # 可选：从 checkpoint 恢复（若未显式提供，则尝试自动从 last 恢复）
    start_epoch = 0
    best_val_top1 = -1.0
    best_path = None
    if (not args.resume) or args.resume.strip() == '':
        # 优先 last_{tag}.pt，其次 last.pt
        cand1 = os.path.join(args.out_dir, f"last_{tag}.pt")
        cand2 = os.path.join(args.out_dir, 'last.pt')
        if os.path.exists(cand1):
            args.resume = cand1
        elif os.path.exists(cand2):
            args.resume = cand2
    if args.resume and os.path.exists(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=device)
            if isinstance(ckpt, dict):
                if 'model' in ckpt:
                    model.load_state_dict(ckpt['model'])
                if 'optimizer' in ckpt and ckpt['optimizer']:
                    optimizer.load_state_dict(ckpt['optimizer'])
                if 'scheduler' in ckpt and ckpt['scheduler'] and hasattr(main_scheduler, 'load_state_dict'):
                    try:
                        main_scheduler.load_state_dict(ckpt['scheduler'])
                    except Exception:
                        pass
                if args.ema and ema is not None and 'ema' in ckpt and ckpt['ema']:
                    try:
                        ema.ema.load_state_dict(ckpt['ema'])
                    except Exception:
                        pass
                start_epoch = int(ckpt.get('epoch', -1)) + 1
                best_val_top1 = float(ckpt.get('best_val_top1', ckpt.get('val_top1', -1.0)))
            print(f"[resume] loaded '{args.resume}' | start_epoch={start_epoch} | best_val_top1={best_val_top1:.2f}")
        except Exception as e:
            print(f"[resume] failed to load '{args.resume}': {e}")

    # data（放到最后，避免仅断言时报错）
    train_loader, val_loader, test_loader = create_dataloaders(
        args.train_list, args.val_list, args.test_list,
        interference_pool_dir=(args.interference_pool if args.interference_pool and os.path.isdir(args.interference_pool) else None),
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_epoch=args.epochs,
        data_root=args.data_root,
        limit_train=(args.limit_train if args.limit_train and args.limit_train > 0 else None),
        limit_val=(args.limit_eval if args.limit_eval and args.limit_eval > 0 else None),
        limit_test=(args.limit_eval if args.limit_eval and args.limit_eval > 0 else None),
        seed=args.seed,
        deterministic=args.deterministic,
    )
    # 新增：打印数据规模与加载配置
    try:
        print(f"Loaded datasets | train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)} | batch_size={args.batch_size} workers={args.workers}")
    except Exception:
        pass

    # 训练循环（支持从中断处继续）
    start = time.time()
    last_alias = os.path.join(args.out_dir, 'last.pt')
    best_alias = os.path.join(args.out_dir, 'best.pt')

    for epoch in range(start_epoch, args.epochs):
        # 解冻时机（仅切换 requires_grad，不重建优化器/调度器，保持单一 LR 曲线）
        if epoch == args.freeze_epochs:
            apply_freeze(model, freeze=False)

        # curriculum update（兼容潜在包装）
        base_ds = getattr(train_loader, 'dataset', None)
        while hasattr(base_ds, 'dataset'):
            base_ds = getattr(base_ds, 'dataset')
        if isinstance(base_ds, DroneRFDataset):
            base_ds.set_epoch(epoch)
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, clip_grad=args.clip_grad, ema=ema)
        val_stats = evaluate(ema.ema if ema is not None else model, val_loader, device)
        # 调度器步进
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        # 当前学习率（取各 param group 平均）
        try:
            lrs = [pg.get('lr', 0.0) for pg in optimizer.param_groups]
            lr_now = float(sum(lrs) / max(1, len(lrs)))
        except Exception:
            lr_now = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0

        print(f"Epoch {epoch+1}/{args.epochs} | Train loss {train_stats['loss']:.4f} top1 {train_stats['top1']:.2f} | Val top1 {val_stats['top1']:.2f} top5 {val_stats['top5']:.2f}")
        # 立刻 flush，确保前台可见
        try:
            import sys as _sys
            _sys.stdout.flush()
        except Exception:
            pass
        # 写 results_*.csv（实时刷盘）
        try:
            with open(args.results_csv, 'a', encoding='utf-8', newline='') as f:
                row = [
                    str(epoch+1),
                    f"{lr_now:.6g}",
                    f"{train_stats['loss']:.6f}",
                    f"{train_stats['top1']:.2f}",
                    f"{val_stats['top1']:.2f}",
                    f"{val_stats['top5']:.2f}",
                ]
                # 其余列（test 与 open-set、gamma/alpha）在最终汇总行再写，这里留空占位
                while len(row) < 16:
                    row.append('')
                f.write(','.join(row) + "\n")
                f.flush()
        except Exception:
            pass

        # 保存 last_*.pt（每epoch覆盖）
        ckpt_current = {
            'epoch': epoch,
            'model': (ema.ema.state_dict() if ema is not None else model.state_dict()),
            'optimizer': optimizer.state_dict(),
            'scheduler': (main_scheduler.state_dict() if hasattr(main_scheduler, 'state_dict') else {}),
            'args': vars(args),
            'val_top1': float(val_stats['top1']),
            'best_val_top1': float(best_val_top1),
            'ema': (ema.ema.state_dict() if ema is not None else None),
        }
        last_path = os.path.join(args.out_dir, f"last_{tag}.pt")
        torch.save(ckpt_current, last_path)
        # 维护通用别名 last.pt，方便监控更新时间
        try:
            shutil.copyfile(last_path, os.path.join(args.out_dir, 'last.pt'))
        except Exception:
            pass

        # 刷新 best_*.pt
        if val_stats['top1'] > best_val_top1:
            best_val_top1 = val_stats['top1']
            best_path = os.path.join(args.out_dir, f"best_{tag}.pt")
            torch.save(ckpt_current, best_path)
            # 维护通用别名 best.pt
            try:
                shutil.copyfile(best_path, os.path.join(args.out_dir, 'best.pt'))
            except Exception:
                pass

    # calibration of gamma on validation set
    model_eval = ema.ema if ema is not None else model
    model_eval.eval()
    all_energy = []
    with torch.no_grad():
        for x_tex, x_pos, y in val_loader:
            x_tex = x_tex.to(device, non_blocking=True)
            x_pos = x_pos.to(device, non_blocking=True)
            logits, energy = model_eval(x_tex, x_pos)
            all_energy.extend(energy.detach().cpu().tolist())
    gamma_cal = calibrate_gamma(np.array(all_energy, dtype=np.float32), alpha=args.alpha)
    print(f"Calibrated gamma (alpha={args.alpha}): {gamma_cal:.6f}")

    test_stats = evaluate(model_eval, test_loader, device)
    print(f"Test top1 {test_stats['top1']:.2f} top5 {test_stats['top5']:.2f}")

    # 记录 open-set 指标，供统一写入 CSV
    os_val_stats = None
    os_test_stats = None

    if args.unknown_val_list and Path(args.unknown_val_list).exists():
        unknown_val_ds = DroneRFDataset(args.unknown_val_list, augment=False, enable_coord=True)
        unknown_val_loader = torch.utils.data.DataLoader(unknown_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        os_stats = eval_open_set(model_eval, val_loader, unknown_val_loader, gamma_cal, device)
        os_val_stats = os_stats
        print(f"Open-set (val): TKR {os_stats['TKR']:.2f} TUR {os_stats['TUR']:.2f} KP {os_stats['KP']:.2f}")
    if args.unknown_test_list and Path(args.unknown_test_list).exists():
        unknown_test_ds = DroneRFDataset(args.unknown_test_list, augment=False, enable_coord=True)
        unknown_test_loader = torch.utils.data.DataLoader(unknown_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        os_stats = eval_open_set(model_eval, test_loader, unknown_test_loader, gamma_cal, device)
        os_test_stats = os_stats
        print(f"Open-set (test): TKR {os_stats['TKR']:.2f} TUR {os_stats['TUR']:.2f} KP {os_stats['KP']:.2f}")

    # open-set 指标仅写入 results_csv 的最终汇总行，不再生成单独 CSV

    if best_path and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location='cpu')
        ckpt['gamma'] = gamma_cal
        torch.save(ckpt, best_path)
        # 同步到通用别名 best.pt
        try:
            shutil.copyfile(best_path, best_alias)
        except Exception:
            pass

    elapsed = (time.time() - start) / 60.0
    print(f"Done. Best val top1: {best_val_top1:.2f}. Time: {elapsed:.1f} min. Checkpoint: {best_path}")
    try:
        import sys as _sys
        _sys.stdout.flush()
    except Exception:
        pass


if __name__ == '__main__':
    main()
