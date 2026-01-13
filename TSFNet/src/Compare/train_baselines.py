import argparse
import os
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.cuda.amp import autocast, GradScaler

# 可选进度条
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

import sys
FILE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(FILE_DIR)  # .../TSFNet/src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
# 统一获取 Exp 根目录（用于输出与权重自动定位）
EXP_ROOT = Path(__file__).resolve().parents[3]

from dataset import create_dataloaders, DroneRFDataset  # noqa: E402

# ============ 帮助函数 ============

def set_seed(seed: int = 42, deterministic: bool = False):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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


def calibrate_gamma(energies: np.ndarray, alpha: float = 0.05) -> float:
    alpha = float(alpha)
    alpha = min(max(alpha, 0.0), 0.5)
    q = np.quantile(energies, 1.0 - alpha)
    return float(q)


def evaluate(model: nn.Module, loader, device: torch.device, *, input_stream: str = 'tex') -> Dict[str, float]:
    model.eval()
    top1_sum, top5_sum, n = 0.0, 0.0, 0
    all_energy: List[float] = []
    with torch.no_grad():
        for batch in loader:
            # 兼容 (I_tex, I_pos, y) 与 (x, y)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x_tex, x_pos, y = batch
                x = x_tex if input_stream == 'tex' else x_pos
            else:
                x, y = batch  # type: ignore
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            energy = -torch.logsumexp(logits, dim=1)
            acc1, acc5 = accuracy(logits, y, topk=(1, 5))
            bs = y.size(0)
            top1_sum += acc1.item() * bs / 100.0
            top5_sum += acc5.item() * bs / 100.0
            n += bs
            all_energy.extend(energy.detach().cpu().tolist())
    return {
        'top1': 100.0 * top1_sum / max(n, 1),
        'top5': 100.0 * top5_sum / max(n, 1),
        'n': float(n),
        'energy_mean': float(np.mean(all_energy) if all_energy else 0.0),
        'energy_list_count': float(len(all_energy)),
    }


def eval_open_set(model: nn.Module, known_loader, unknown_loader, gamma: float, device: torch.device, *, input_stream: str = 'tex') -> Dict[str, float]:
    model.eval()
    # Known
    A_known, N_known, TP_known = 0, 0, 0
    with torch.no_grad():
        for batch in known_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x_tex, x_pos, y = batch
                x = x_tex if input_stream == 'tex' else x_pos
            else:
                x, y = batch  # type: ignore
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            energy = -torch.logsumexp(logits, dim=1)
            accept_known = (energy < gamma)
            A_known += int(accept_known.sum().item())
            N_known += y.numel()
            if accept_known.any():
                y_pred = logits.argmax(dim=1)
                TP_known += int(((y_pred == y) & accept_known).sum().item())
    # Unknown
    TN_unknown, N_unknown, FP_unknown = 0, 0, 0
    with torch.no_grad():
        for batch in unknown_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x_tex, x_pos, _ = batch
                x = x_tex if input_stream == 'tex' else x_pos
            else:
                x, _ = batch  # type: ignore
            x = x.to(device, non_blocking=True)
            logits = model(x)
            energy = -torch.logsumexp(logits, dim=1)
            reject = (energy >= gamma)
            TN_unknown += int(reject.sum().item())
            N_unknown += reject.numel()
            FP_unknown += int((~reject).sum().item())
    TKR = A_known / max(N_known, 1)
    TUR = TN_unknown / max(N_unknown, 1)
    KP = TP_known / max((TP_known + FP_unknown), 1)
    return {'TKR': 100.0 * TKR, 'TUR': 100.0 * TUR, 'KP': 100.0 * KP,
            'N_known': float(N_known), 'N_unknown': float(N_unknown)}


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


def load_state_dict_relaxed(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    msd = module.state_dict()
    loaded, skipped = 0, 0
    with torch.no_grad():
        for k, v in state_dict.items():
            kk = k
            if kk.startswith('module.'):
                kk = kk[len('module.'):]
            if kk in msd and msd[kk].shape == v.shape:
                msd[kk].copy_(v)
                loaded += 1
            else:
                skipped += 1
    module.load_state_dict(msd)
    return loaded, skipped


# ============ 模型构建 ============

def build_baseline(arch: str, num_classes: int) -> nn.Module:
    # 优先 timm，覆盖更多家族
    model = None
    timm_name = map_arch_to_timm(arch)
    try:
        import timm  # type: ignore
        if timm_name is not None:
            model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes, in_chans=3)
        else:
            # 如果直接传了 timm 架构名
            model = timm.create_model(arch, pretrained=False, num_classes=num_classes, in_chans=3)
        return model
    except Exception:
        pass
    # 退回 torchvision 常见模型
    import torchvision.models as tvm  # type: ignore
    fam = arch.lower()
    if fam == 'alexnet':
        model = tvm.alexnet(num_classes=num_classes)
    elif fam in ('convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    elif fam in ('densenet121', 'densenet169', 'densenet201'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    elif fam in ('efficientnet_b0','efficientnet_b1','efficientnet_b2','efficientnet_b3','efficientnet_b4','efficientnet_b5','efficientnet_b6','efficientnet_b7'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    elif fam in ('efficientnet_v2_s','efficientnet_v2_m','efficientnet_v2_l'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    elif fam == 'googlenet':
        model = tvm.googlenet(num_classes=num_classes, aux_logits=False)
    elif fam == 'inception_v3':
        model = tvm.inception_v3(num_classes=num_classes, aux_logits=False)
    elif fam.startswith('mnasnet'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    elif fam in ('mobilenet_v2','mobilenet_v3_small','mobilenet_v3_large'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    elif fam.startswith('regnet'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    elif fam in ('resnet18','resnet34','resnet50','resnet101','resnet152'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    elif fam in ('resnext50_32x4d','resnext101_32x8d'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    elif fam.startswith('shufflenet_v2'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    elif fam.startswith('squeezenet'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    elif fam in ('swin_t','swin_s','swin_b'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    elif fam in ('vgg11','vgg13','vgg16','vgg19','vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    elif fam in ('vit_b_16','vit_b_32','vit_l_16','vit_l_32'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    elif fam in ('wide_resnet50_2','wide_resnet101_2'):
        ctor = getattr(tvm, fam)
        model = ctor(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported arch '{arch}'. Try installing 'timm' or choose a torchvision arch.")
    return model


def map_arch_to_timm(arch: str) -> Optional[str]:
    a = arch.lower()
    mapping = {
        # 基础家族映射（可扩展）
        'alexnet': 'alexnet',
        'convnext': 'convnext_base',
        'convnext_tiny': 'convnext_tiny',
        'convnext_small': 'convnext_small',
        'convnext_base': 'convnext_base',
        'convnext_large': 'convnext_large',
        'densenet': 'densenet121',
        'densenet121': 'densenet121',
        'densenet169': 'densenet169',
        'densenet201': 'densenet201',
        'efficientnet': 'efficientnet_b0',
        'efficientnet_b0': 'efficientnet_b0',
        'efficientnetv2': 'efficientnetv2_s',
        'efficientnetv2_s': 'efficientnetv2_s',
        'efficientnetv2_m': 'efficientnetv2_m',
        'efficientnetv2_l': 'efficientnetv2_l',
        'googlenet': 'googlenet',
        'inceptionv3': 'inception_v3',
        'inception_v3': 'inception_v3',
        'maxvit': 'maxvit_tiny_tf_224',
        'maxvit_tiny': 'maxvit_tiny_tf_224',
        'mnasnet': 'mnasnet_100',
        'mnasnet1_0': 'mnasnet_100',
        'mobilenetv2': 'mobilenetv2_100',
        'mobilenet_v2': 'mobilenetv2_100',
        'mobilenetv3': 'mobilenetv3_large_100',
        'mobilenet_v3_large': 'mobilenetv3_large_100',
        'regnet': 'regnety_008',
        'regnet_y_8gf': 'regnety_008',
        'resnet': 'resnet50',
        'resnet50': 'resnet50',
        'resnext': 'resnext50_32x4d',
        'resnext50_32x4d': 'resnext50_32x4d',
        'shufflenetv2': 'shufflenet_v2_x1_0',
        'shufflenet_v2_x1_0': 'shufflenet_v2_x1_0',
        'squeezenet': 'squeezenet1_1',
        'squeezenet1_1': 'squeezenet1_1',
        'swintransformer': 'swin_base_patch4_window7_224',
        'swin_t': 'swin_tiny_patch4_window7_224',
        'vgg': 'vgg16_bn',
        'vgg16': 'vgg16_bn',
        'visiontransformer': 'vit_base_patch16_224',
        'vit_base_patch16_224': 'vit_base_patch16_224',
        'wideresnet': 'wide_resnet50_2',
        'wide_resnet50_2': 'wide_resnet50_2',
    }
    return mapping.get(a, None)


# ============ 训练 ============

def train_one_epoch(model, loader, criterion, optimizer, device, scaler: Optional[GradScaler] = None, clip_grad: float = 0.0, *, input_stream: str = 'tex'):
    model.train()
    loss_sum, top1_sum, top5_sum, n = 0.0, 0.0, 0.0, 0
    it = loader if tqdm is None else tqdm(loader, total=len(loader), desc='Train', ncols=100)
    for i, batch in enumerate(it):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x_tex, x_pos, y = batch
            x = x_tex if input_stream == 'tex' else x_pos
        else:
            x, y = batch  # type: ignore
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if clip_grad and clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if clip_grad and clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        acc1, acc5 = accuracy(logits, y, topk=(1, 5))
        bs = y.size(0)
        loss_sum += loss.item() * bs
        top1_sum += acc1.item() * bs / 100.0
        top5_sum += acc5.item() * bs / 100.0
        n += bs
        if tqdm is not None:
            it.set_postfix({'loss': f"{loss_sum/max(n,1):.4f}", 'top1': f"{100.0*top1_sum/max(n,1):.2f}"})
    if tqdm is not None and hasattr(it, 'close'):
        it.close()
    return {
        'loss': loss_sum / max(n, 1),
        'top1': 100.0 * top1_sum / max(n, 1),
        'top5': 100.0 * top5_sum / max(n, 1),
    }


def auto_pick_weights(pretrain_dir: str) -> str:
    p = Path(pretrain_dir)
    if not p.exists():
        return ''
    # 优先 *.pth / *.pt
    for ext in ('*.pth', '*.pt', '*.ckpt'):
        files = sorted(p.glob(ext))
        if files:
            return str(files[0].resolve())
    # 递归找一下
    for ext in ('*.pth', '*.pt', '*.ckpt'):
        files = sorted(p.rglob(ext))
        if files:
            return str(files[0].resolve())
    return ''


def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--train_list', type=str, default='./experiment_groups/1-known_for_train')
    parser.add_argument('--val_list', type=str, default='./experiment_groups/2-known_for_val')
    parser.add_argument('--test_list', type=str, default='./experiment_groups/3-known_for_test')
    parser.add_argument('--unknown_val_list', type=str, default='')
    parser.add_argument('--unknown_test_list', type=str, default='')
    parser.add_argument('--data_root', type=str, default='C:/Users/HP/Desktop/Exp/Data')
    parser.add_argument('--limit_train', type=int, default=0)
    parser.add_argument('--limit_eval', type=int, default=0)
    # model
    parser.add_argument('--arch', type=str, default='resnet50',
                        help='baseline architecture, e.g., resnet50/mobilenet_v3_large/convnext_base/vit_base_patch16_224/swin_t, or a timm model name')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--weights', type=str, default='', help='path to weights (.pth/.pt). If a directory is given, auto pick the first file')
    # train
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.05, help='open-set gamma calibration alpha')
    # misc
    parser.add_argument('--out_dir', type=str, default='')
    # 新增：选择使用纹理流或位置流作为单流基准输入
    parser.add_argument('--input_stream', type=str, default='tex', choices=['tex','pos'], help='use texture (tex) or position (pos) stream as baseline input')

    args = parser.parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 统一输出目录：Exp/Model/Output/Compare/<arch>/runX
    if not args.out_dir or args.out_dir.strip() == '':
        project_root = EXP_ROOT  # .../Exp
        base_out = project_root / 'Model' / 'Output' / 'Compare' / args.arch
        base_out.mkdir(parents=True, exist_ok=True)
        # 自动递增 run 号
        runs = [d for d in base_out.iterdir() if d.is_dir() and d.name.startswith('run')]
        def parse_idx(name: str) -> int:
            try:
                return int(name[3:])
            except Exception:
                return 0
        next_idx = (max([parse_idx(d.name) for d in runs]) + 1) if runs else 1
        args.out_dir = str((base_out / f"run{next_idx}").resolve())
    os.makedirs(args.out_dir, exist_ok=True)

    # 自动权重目录推断（来自 Model/Pretrain/Basic/*）
    if args.weights and os.path.isdir(args.weights):
        args.weights = auto_pick_weights(args.weights)
    elif not args.weights:
        try:
            project_root = EXP_ROOT
            basic_dir = project_root / 'Model' / 'Pretrain' / 'Basic'
            # 通过家族目录名推断
            family_map = {
                'alexnet': 'AlexNet',
                'convnext': 'ConvNeXt',
                'densenet': 'DenseNet',
                'efficientnet': 'EfficientNet',
                'efficientnetv2': 'EfficientNetV2',
                'googlenet': 'GoogLeNet',
                'inception_v3': 'InceptionV3',
                'inceptionv3': 'InceptionV3',
                'maxvit': 'MaxVit',
                'mnasnet': 'MNASNet',
                'mobilenet_v2': 'MobileNetV2',
                'mobilenetv2': 'MobileNetV2',
                'mobilenet_v3_large': 'MobileNetV3',
                'mobilenetv3': 'MobileNetV3',
                'regnet': 'RegNet',
                'resnet': 'ResNet',
                'resnet50': 'ResNet',
                'resnext': 'ResNeXt',
                'resnext50_32x4d': 'ResNeXt',
                'shufflenet_v2_x1_0': 'ShuffleNetV2',
                'shufflenetv2': 'ShuffleNetV2',
                'squeezenet': 'SqueezeNet',
                'swin_t': 'SwinTransformer',
                'swin': 'SwinTransformer',
                'vgg': 'VGG',
                'vgg16': 'VGG',
                'vit_base_patch16_224': 'VisionTransformer',
                'visiontransformer': 'VisionTransformer',
                'wide_resnet50_2': 'WideResNet',
                'wideresnet': 'WideResNet',
            }
            fam_key = args.arch.lower()
            fam_dir = family_map.get(fam_key, None)
            if fam_dir:
                cand = basic_dir / fam_dir
                if cand.exists():
                    args.weights = auto_pick_weights(str(cand))
        except Exception:
            pass

    # 构建模型
    model = build_baseline(args.arch, args.num_classes).to(device)

    # 尝试加载权重（非严格）
    if args.weights and os.path.exists(args.weights):
        try:
            sd = torch.load(args.weights, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            if isinstance(sd, dict):
                loaded, skipped = load_state_dict_relaxed(model, sd)
                print(f"[weights] loaded={loaded} skipped={skipped} from {args.weights}")
        except Exception as e:
            print(f"[weights] failed to load '{args.weights}': {e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 调度器
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

    # 数据
    train_loader, val_loader, test_loader = create_dataloaders(
        args.train_list, args.val_list, args.test_list,
        interference_pool_dir=None,
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
    try:
        print(f"Loaded datasets | train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)} | batch_size={args.batch_size} workers={args.workers}")
    except Exception:
        pass

    # 路径提示
    last_alias = os.path.join(args.out_dir, 'last.pt')
    best_alias = os.path.join(args.out_dir, 'best.pt')
    print(f"[paths] out_dir={args.out_dir}")
    print(f"[paths] last={last_alias}")
    print(f"[paths] best={best_alias}")

    # CSV 初始化
    results_csv = os.path.join(args.out_dir, f"results_{args.arch}.csv")
    if not os.path.exists(results_csv):
        with open(results_csv, 'w', encoding='utf-8', newline='') as f:
            f.write("epoch,lr,train_loss,train_top1,val_top1,val_top5,test_top1,test_top5,TKR_val,TUR_val,KP_val,TKR_test,TUR_test,KP_test,gamma,alpha\n")

    # 训练
    best_val_top1 = -1.0
    best_path = None
    start = time.time()

    for epoch in range(0, args.epochs):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, clip_grad=args.clip_grad, input_stream=args.input_stream)
        val_stats = evaluate(model, val_loader, device, input_stream=args.input_stream)
        # 调度器步进
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        # 当前学习率
        try:
            lrs = [pg.get('lr', 0.0) for pg in optimizer.param_groups]
            lr_now = float(sum(lrs) / max(1, len(lrs)))
        except Exception:
            lr_now = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0

        print(f"Epoch {epoch+1}/{args.epochs} | Train loss {train_stats['loss']:.4f} top1 {train_stats['top1']:.2f} | Val top1 {val_stats['top1']:.2f} top5 {val_stats['top5']:.2f}")
        try:
            sys.stdout.flush()
        except Exception:
            pass

        # 追加 CSV（test/open-set 留空，末尾汇总）
        with open(results_csv, 'a', encoding='utf-8', newline='') as f:
            row = [str(epoch+1), f"{lr_now:.6g}", f"{train_stats['loss']:.6f}", f"{train_stats['top1']:.2f}", f"{val_stats['top1']:.2f}", f"{val_stats['top5']:.2f}"]
            while len(row) < 16:
                row.append('')
            f.write(','.join(row) + "\n")
            f.flush()

        # 保存 last/best
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'val_top1': float(val_stats['top1']),
            'best_val_top1': float(best_val_top1),
        }
        last_path = os.path.join(args.out_dir, f"last_{args.arch}.pt")
        torch.save(ckpt, last_path)
        try:
            shutil.copyfile(last_path, last_alias)
        except Exception:
            pass

        if val_stats['top1'] > best_val_top1:
            best_val_top1 = val_stats['top1']
            best_path = os.path.join(args.out_dir, f"best_{args.arch}.pt")
            torch.save(ckpt, best_path)
            try:
                shutil.copyfile(best_path, best_alias)
            except Exception:
                pass

    # gamma 校准 + 测试
    model.eval()
    all_energy = []
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x_tex, x_pos, _ = batch
                x = x_tex if args.input_stream == 'tex' else x_pos
            else:
                x, _ = batch  # type: ignore
            x = x.to(device, non_blocking=True)
            logits = model(x)
            energy = -torch.logsumexp(logits, dim=1)
            all_energy.extend(energy.detach().cpu().tolist())
    gamma_cal = calibrate_gamma(np.array(all_energy, dtype=np.float32), alpha=args.alpha)
    print(f"Calibrated gamma (alpha={args.alpha}): {gamma_cal:.6f}")

    test_stats = evaluate(model, test_loader, device, input_stream=args.input_stream)
    print(f"Test top1 {test_stats['top1']:.2f} top5 {test_stats['top5']:.2f}")

    # open-set（可选）
    os_val_stats = None
    os_test_stats = None
    if args.unknown_val_list and Path(args.unknown_val_list).exists():
        unknown_val_ds = DroneRFDataset(args.unknown_val_list, augment=False, enable_coord=True)
        unknown_val_loader = torch.utils.data.DataLoader(unknown_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        os_stats = eval_open_set(model, val_loader, unknown_val_loader, gamma_cal, device, input_stream=args.input_stream)
        os_val_stats = os_stats
        print(f"Open-set (val): TKR {os_stats['TKR']:.2f} TUR {os_stats['TUR']:.2f} KP {os_stats['KP']:.2f}")
    if args.unknown_test_list and Path(args.unknown_test_list).exists():
        unknown_test_ds = DroneRFDataset(args.unknown_test_list, augment=False, enable_coord=True)
        unknown_test_loader = torch.utils.data.DataLoader(unknown_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        os_stats = eval_open_set(model, test_loader, unknown_test_loader, gamma_cal, device, input_stream=args.input_stream)
        os_test_stats = os_stats
        print(f"Open-set (test): TKR {os_stats['TKR']:.2f} TUR {os_stats['TUR']:.2f} KP {os_stats['KP']:.2f}")

    # 末尾写入一行汇总
    with open(results_csv, 'a', encoding='utf-8', newline='') as f:
        row = [
            'final', '', '', '', '', '',
            f"{test_stats['top1']:.2f}", f"{test_stats['top5']:.2f}",
            f"{(os_val_stats['TKR'] if os_val_stats else 0):.2f}",
            f"{(os_val_stats['TUR'] if os_val_stats else 0):.2f}",
            f"{(os_val_stats['KP'] if os_val_stats else 0):.2f}",
            f"{(os_test_stats['TKR'] if os_test_stats else 0):.2f}",
            f"{(os_test_stats['TUR'] if os_test_stats else 0):.2f}",
            f"{(os_test_stats['KP'] if os_test_stats else 0):.2f}",
            f"{gamma_cal:.6f}", f"{args.alpha}"
        ]
        f.write(','.join(row) + "\n")
        f.flush()

    # 把 gamma 写入 best.ckpt
    if best_path and os.path.exists(best_path):
        try:
            ckpt = torch.load(best_path, map_location='cpu')
            ckpt['gamma'] = gamma_cal
            torch.save(ckpt, best_path)
            shutil.copyfile(best_path, best_alias)
        except Exception:
            pass

    elapsed = (time.time() - start) / 60.0
    print(f"Done. Best val top1: {best_val_top1:.2f}. Time: {elapsed:.1f} min. Checkpoint: {best_path}")
    try:
        sys.stdout.flush()
    except Exception:
        pass


if __name__ == '__main__':
    main()
