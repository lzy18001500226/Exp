import argparse
import json
import os
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.ops import box_iou
import torchvision
from torchvision.ops import nms

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# ---- COCO 数据集封装（最小实现） ----
class CocoDetDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: str, annotations: str, image_ids: List[int] = None, score_thr: float = 0.0):
        self.images_dir = Path(images_dir)
        with open(annotations, 'r', encoding='utf-8') as f:
            coco = json.load(f)
        self.images = {img['id']: img for img in coco['images']}
        self.anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco['annotations']:
            if ann.get('iscrowd', 0) == 1:
                continue
            self.anns_by_img.setdefault(ann['image_id'], []).append(ann)
        # 类别从 1..K，映射到 1..K（0 保留为背景）
        self.cat_ids: List[int] = sorted([c['id'] for c in coco['categories']])
        self.catid2contig = {cid: i + 1 for i, cid in enumerate(self.cat_ids)}
        self.contig2catid = {v: k for k, v in self.catid2contig.items()}
        self.classes = len(self.cat_ids) + 1  # 含背景 0
        # 采样的 image_ids
        if image_ids is None:
            self.ids = list(self.images.keys())
        else:
            self.ids = image_ids
        self.ids.sort()
        self.score_thr = float(score_thr)

    def __len__(self):
        return len(self.ids)

    def _load_image(self, p: Path):
        from PIL import Image  # lazy import
        img = Image.open(str(p)).convert('RGB')
        return torchvision.transforms.functional.to_tensor(img)  # [0,1]

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.images[img_id]
        file_name = info['file_name']
        img_path = self.images_dir / file_name
        image = self._load_image(img_path)
        anns = self.anns_by_img.get(img_id, [])
        boxes = []
        labels = []
        areas = []
        for a in anns:
            x, y, w, h = a['bbox']
            if w <= 1e-6 or h <= 1e-6:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.catid2contig.get(a['category_id'], 0))
            areas.append(float(a.get('area', w * h)))
        target: Dict[str, Any] = {}
        if boxes:
            target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.tensor(labels, dtype=torch.int64)
            target['area'] = torch.tensor(areas, dtype=torch.float32)
        else:
            target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
            target['labels'] = torch.empty((0,), dtype=torch.int64)
            target['area'] = torch.empty((0,), dtype=torch.float32)
        target['image_id'] = torch.tensor([img_id], dtype=torch.int64)
        return image, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


# ---- 指标计算（IoU 匹配） ----
def greedy_match_iou(pred_boxes: torch.Tensor, pred_scores: torch.Tensor, gt_boxes: torch.Tensor, iou_thr: float) -> Tuple[int, int, int]:
    if gt_boxes.numel() == 0:
        tp = 0
        fp = int(pred_boxes.size(0))
        fn = 0
        return tp, fp, fn
    if pred_boxes.numel() == 0:
        return 0, 0, int(gt_boxes.size(0))
    ious = box_iou(pred_boxes, gt_boxes)  # [Np, Ng]
    matched_gt = set()
    order = torch.argsort(pred_scores, descending=True)
    tp = 0
    for i in order.tolist():
        giou = ious[i]
        max_iou, j = torch.max(giou, dim=0)
        j = int(j.item())
        if float(max_iou.item()) >= iou_thr and j not in matched_gt:
            tp += 1
            matched_gt.add(j)
    fp = int(pred_boxes.size(0)) - tp
    fn = int(gt_boxes.size(0)) - tp
    return tp, fp, fn


def evaluate_simple(model, loader, device, score_thr: float = 0.25, iou_thr: float = 0.5) -> Dict[str, float]:
    model.eval()
    TP = FP = FN = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                boxes = out['boxes']
                scores = out.get('scores', torch.ones((boxes.size(0),), device=boxes.device))
                labels = out.get('labels', torch.zeros((boxes.size(0),), device=boxes.device, dtype=torch.int64))
                keep = scores >= score_thr
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                # 逐类匹配
                gt_boxes = tgt['boxes'].to(device)
                gt_labels = tgt['labels'].to(device)
                for c in torch.unique(torch.cat([labels, gt_labels], dim=0)):
                    if int(c.item()) == 0:
                        continue
                    cb = boxes[labels == c]
                    cs = scores[labels == c]
                    gb = gt_boxes[gt_labels == c]
                    tp, fp, fn = greedy_match_iou(cb, cs, gb, iou_thr)
                    TP += tp; FP += fp; FN += fn
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
    return {
        'precision@0.5': float(precision) * 100.0,
        'recall@0.5': float(recall) * 100.0,
        'f1@0.5': float(f1) * 100.0,
        'TP': float(TP), 'FP': float(FP), 'FN': float(FN)
    }


def sweep_thresholds(model, loader, device, thr_list: List[float], iou_thr: float, out_csv: str = None) -> List[Dict[str, float]]:
    """对一组 score 阈值进行评估，返回按阈值排序的结果列表，并可选写入 CSV。"""
    results: List[Dict[str, float]] = []
    thr_list = sorted(set([float(t) for t in thr_list]))
    # 逐阈值评估（简单起见重复前向；若要加速可缓存输出）
    for thr in thr_list:
        stats = evaluate_simple(model, loader, device, score_thr=thr, iou_thr=iou_thr)
        stats = {'thr': thr, **stats}
        results.append(stats)
    # 控制台打印
    print("[Sweep] thr, precision@0.5, recall@0.5, f1@0.5, TP, FP, FN")
    for r in results:
        print(f"[Sweep] {r['thr']:.3f}, {r['precision@0.5']:.2f}, {r['recall@0.5']:.2f}, {r['f1@0.5']:.2f}, {int(r['TP'])}, {int(r['FP'])}, {int(r['FN'])}")
    # 写 CSV
    if out_csv:
        try:
            header = 'thr,precision@0.5,recall@0.5,f1@0.5,TP,FP,FN\n'
            with open(out_csv, 'w', encoding='utf-8', newline='') as f:
                f.write(header)
                for r in results:
                    f.write(f"{r['thr']:.6g},{r['precision@0.5']:.4f},{r['recall@0.5']:.4f},{r['f1@0.5']:.4f},{int(r['TP'])},{int(r['FP'])},{int(r['FN'])}\n")
        except Exception as e:
            print(f"[Sweep] write csv failed: {e}")
    return results


# ---- 构建检测模型（torchvision） ----
def build_detector(arch: str, num_classes: int):
    a = arch.lower()
    # 明确禁止在线下载：weights=None 且 weights_backbone=None
    if a in ('fasterrcnn_resnet50_fpn', 'fasterrcnn-r50'):
        try:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
        except TypeError:
            # 兼容旧版 torchvision（不支持 weights_backbone 参数）
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    elif a in ('retinanet_resnet50_fpn', 'retinanet-r50'):
        try:
            model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
        except TypeError:
            model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported detector arch: {arch}")
    return model


def set_seed(seed: int = 42, deterministic: bool = False):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 尝试粗分 backbone 与 head
    bb_params = 0
    head_params = 0
    backbone = getattr(model, 'backbone', None)
    if backbone is not None:
        bb_param_ids = set(id(p) for p in backbone.parameters())
        for p in model.parameters():
            if id(p) in bb_param_ids:
                bb_params += p.numel()
            else:
                head_params += p.numel()
    return {'total': total, 'trainable': trainable, 'backbone': bb_params, 'head': head_params}


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--arch', type=str, default='fasterrcnn_resnet50_fpn')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--score_thr', type=float, default=0.3)
    parser.add_argument('--iou_thr', type=float, default=0.5)
    parser.add_argument('--val_split', type=float, default=0.2, help='随机划分 val 比例，0 表示用全部作 train 并用同集评估')
    parser.add_argument('--out_dir', type=str, default='')
    # 新增：本地 ResNet50 预训练权重路径（.pth）。若提供，将加载到 backbone；否则完全不使用预训练也不下载。
    parser.add_argument('--backbone_weights', type=str, default='', help='本地 ResNet50 预训练权重路径（.pth），将以 strict=False 加载到 backbone.body')
    # 新增：阈值扫，逗号分隔，如 "0.05,0.1,0.2,0.3"
    parser.add_argument('--sweep_scores', type=str, default='', help='逗号分隔的 score 阈值列表，示例：0.05,0.1,0.2,0.3')

    args = parser.parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 显式禁用 TORCH 在线缓存下载（可选）
    os.environ.setdefault('TORCH_HOME', str((Path(__file__).resolve().parents[3] / 'Model' / 'Pretrain').resolve()))
    os.environ.setdefault('TORCH_MODEL_ZOO', os.environ['TORCH_HOME'])

    # 读取 COCO 并划分
    with open(args.annotations, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    all_ids = sorted([img['id'] for img in coco['images']])
    # 按 seed 打乱，避免验证集偶然集中为无目标样本
    try:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(all_ids))
        all_ids = [all_ids[i] for i in perm]
    except Exception:
        pass
    n = len(all_ids)
    n_val = int(n * max(0.0, min(1.0, args.val_split)))
    val_ids = set(all_ids[:n_val])
    train_ids = [i for i in all_ids if i not in val_ids]
    val_ids = list(val_ids)

    train_ds = CocoDetDataset(args.images_dir, args.annotations, image_ids=train_ids)
    val_ds = CocoDetDataset(args.images_dir, args.annotations, image_ids=val_ids)

    # 验证集GT统计自检
    try:
        pos = tot = boxes = 0
        for i in range(len(val_ds)):
            _, tgt = val_ds[i]
            tot += 1
            b = tgt.get('boxes', None)
            if b is not None and getattr(b, 'numel', lambda: 0)() > 0:
                pos += 1
                boxes += int(b.size(0)) if hasattr(b, 'size') else 0
        print(f"[Sanity] Val GT images with boxes: {pos}/{tot} | total boxes: {boxes}")
    except Exception:
        pass

    def seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        import random as _r
        _r.seed(worker_seed)
        np.random.seed(worker_seed)

    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g)

    num_classes = train_ds.classes  # 含背景
    model = build_detector(args.arch, num_classes=num_classes).to(device)

    # 打印参数量
    pinfo = count_params(model)
    print(f"[Model] params total={pinfo['total']:,} | trainable={pinfo['trainable']:,} | backbone={pinfo['backbone']:,} | head={pinfo['head']:,}")

    # 若提供本地 backbone 预训练权重，则加载到 backbone.body（strict=False）
    if isinstance(args.backbone_weights, str) and args.backbone_weights.strip():
        wpath = Path(args.backbone_weights).expanduser().resolve()
        if wpath.is_file():
            try:
                ckpt = torch.load(str(wpath), map_location='cpu')
                state = ckpt.get('state_dict', ckpt)
                # 去除多卡前缀
                clean_state = {k.replace('module.', '').replace('backbone.', '').replace('model.', ''): v for k, v in state.items()}
                # 大多数 resnet50 权重使用与 torchvision resnet50 一致的命名
                backbone = getattr(model.backbone, 'body', None) or getattr(model, 'backbone', None)
                missing, unexpected = [], []
                if backbone is not None:
                    info = backbone.load_state_dict(clean_state, strict=False)
                    # torch>=1.12 返回 Missing/Unexpected 列表；旧版可能返回 None
                    if hasattr(info, 'missing_keys') and hasattr(info, 'unexpected_keys'):
                        missing = list(info.missing_keys)
                        unexpected = list(info.unexpected_keys)
                print(f"[Init] Loaded backbone weights: {wpath.name} | strict=False | missing={len(missing)} unexpected={len(unexpected)}")
            except Exception as e:
                print(f"[Init] Failed to load backbone weights from {wpath}: {e}")
        else:
            print(f"[Init] backbone_weights not found: {wpath}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    # 输出目录：Model/Output/CompareDet/<arch>/runX
    if not args.out_dir or args.out_dir.strip() == '':
        project_root = Path(__file__).resolve().parents[3]  # .../Exp
        base_out = project_root / 'Model' / 'Output' / 'CompareDet' / args.arch
        base_out.mkdir(parents=True, exist_ok=True)
        runs = [d for d in base_out.iterdir() if d.is_dir() and d.name.startswith('run')]
        def parse_idx(name: str) -> int:
            try: return int(name[3:])
            except Exception: return 0
        next_idx = (max([parse_idx(d.name) for d in runs]) + 1) if runs else 1
        args.out_dir = str((base_out / f'run{next_idx}').resolve())
    os.makedirs(args.out_dir, exist_ok=True)
    results_csv = os.path.join(args.out_dir, f'results_{args.arch}.csv')
    if not os.path.exists(results_csv):
        with open(results_csv, 'w', encoding='utf-8', newline='') as f:
            f.write('epoch,lr,loss,precision@0.5,recall@0.5,f1@0.5,TP,FP,FN\n')

    last_alias = os.path.join(args.out_dir, 'last.pt')
    best_alias = os.path.join(args.out_dir, 'best.pt')
    best_f1 = -1.0
    best_path = None

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_imgs = 0
        it = train_loader if tqdm is None else tqdm(train_loader, total=len(train_loader), desc=f'Train[{epoch+1}/{args.epochs}]', ncols=100)
        for images, targets in it:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with autocast():
                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                loss.backward()
                optimizer.step()
            bs = len(images)
            epoch_loss += float(loss.item()) * bs
            n_imgs += bs
            if tqdm is not None:
                it.set_postfix({'loss': f'{(epoch_loss/max(n_imgs,1)):.4f}'})
        if tqdm is not None and hasattr(it, 'close'):
            it.close()
        avg_loss = epoch_loss / max(n_imgs, 1)

        # 简单评估
        val_stats = evaluate_simple(model, val_loader, device, score_thr=args.score_thr, iou_thr=args.iou_thr)
        f1 = val_stats['f1@0.5']
        # 写 CSV
        lr_now = optimizer.param_groups[0]['lr']
        with open(results_csv, 'a', encoding='utf-8', newline='') as f:
            row = [str(epoch+1), f'{lr_now:.6g}', f'{avg_loss:.6f}', f"{val_stats['precision@0.5']:.4f}", f"{val_stats['recall@0.5']:.4f}", f"{val_stats['f1@0.5']:.4f}", str(int(val_stats['TP'])), str(int(val_stats['FP'])), str(int(val_stats['FN']))]
            f.write(','.join(row) + '\n'); f.flush()

        # 保存 last/best
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'val_f1': float(f1),
        }
        last_path = os.path.join(args.out_dir, f'last_{args.arch}.pt')
        torch.save(ckpt, last_path)
        try:
            shutil.copyfile(last_path, last_alias)
        except Exception:
            pass
        if f1 > best_f1:
            best_f1 = f1
            best_path = os.path.join(args.out_dir, f'best_{args.arch}.pt')
            torch.save(ckpt, best_path)
            try:
                shutil.copyfile(best_path, best_alias)
            except Exception:
                pass

        # 可选阈值扫（每个 epoch 结束时）
        if isinstance(args.sweep_scores, str) and args.sweep_scores.strip():
            try:
                thr_list = [float(x) for x in args.sweep_scores.split(',') if x.strip()]
                sweep_csv = os.path.join(args.out_dir, f'sweep_{args.arch}_ep{epoch+1}.csv')
                sweep_thresholds(model, val_loader, device, thr_list, args.iou_thr, out_csv=sweep_csv)
            except Exception as e:
                print(f"[Sweep] parse or run failed: {e}")

    print(f'Done. Best F1@0.5: {best_f1:.4f}. Checkpoint: {best_path}')


if __name__ == '__main__':
    main()
