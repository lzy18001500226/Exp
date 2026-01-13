import argparse
from pathlib import Path
import random
import re
from typing import List, Optional, Set, Dict
import json


def parse_args():
    p = argparse.ArgumentParser('Split NPZ dataset into train/val with 80/20 random mode by default (optionally group by stem blocks)')
    p.add_argument('--data_root', type=str, default='C:/Users/HP/Desktop/Exp/Data-512NPZ', help='Root of fused NPZ (class subfolders)')
    p.add_argument('--out_dir', type=str, default='C:/Users/HP/Desktop/Exp/Model/DSFNet-Output', help='Where to save list files under a run dir')
    p.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio (only used in random mode, default 0.2 for 8:2)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--run_name', type=str, default='200标签训练1', help='Run directory name (manually or externally auto-incremented)')
    # 任务类型（仅影响默认类选择与输出后缀，不再基于掩码判断）
    p.add_argument('--task', choices=['classification', 'fcs', 'vts'], default='classification',
                   help='classification/fcs/vts; filtering is based on class id and filename numeric prefix only')
    # 类范围与显式类列表
    p.add_argument('--class_min', type=int, default=0, help='Inclusive min class id to keep')
    p.add_argument('--class_max', type=int, default=23, help='Inclusive max class id to keep')
    p.add_argument('--include_classes', type=str, default='',
                   help='Comma-separated class ids to include (e.g., "0,2,3,5"). If empty, use task-specific defaults within [class_min,class_max].')
    p.add_argument('--exclude_classes', type=str, default='',
                   help='Comma-separated class ids to exclude (e.g., "13,14"). Will be removed from the final class set.')
    # 文件名数值前缀范围（自然序号限制）
    p.add_argument('--stem_min', type=int, default=0, help='Inclusive min numeric prefix of filename stem to keep (e.g., 0 for 0-a)')
    p.add_argument('--stem_max', type=int, default=200, help='Inclusive max numeric prefix of filename stem to keep (e.g., 200 for 200-a)')

    # 新增：分组模式（默认使用随机 80/20 划分）
    p.add_argument('--group_mode', choices=['block', 'random'], default='random',
                   help='random: shuffle by files with val_ratio (default 80/20); block: contiguous stem blocks')
    p.add_argument('--block_size', type=int, default=20, help='Number of consecutive stems per block')
    p.add_argument('--train_blocks', type=int, default=4, help='Number of consecutive train blocks in a cycle')
    p.add_argument('--val_blocks', type=int, default=1, help='Number of consecutive val blocks in a cycle')
    # 新增：输出 JSON 标签清单（检测任务用），并可严格只保留非空 JSON
    p.add_argument('--emit_json_labels', choices=['none', 'fcs', 'vts'], default='none',
                   help='Emit label JSON paths instead of NPZ paths. "fcs" -> FCSLabel, "vts" -> VTSLabel. Default "none" keeps NPZ list.')
    p.add_argument('--strict_json_only', action='store_true',
                   help='When emitting JSON labels, keep only entries whose JSON exists and has shapes>0, and NPZ exists')
    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _iter_npz_files(root: Path):
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name)):
        for f in sorted(cls_dir.glob('*.npz')):
            yield f


def _stem_num(stem: str) -> Optional[int]:
    m = re.match(r'^(\d+)', stem)
    return int(m.group(1)) if m else None


def _parse_id_list(arg: str) -> Set[int]:
    ids: Set[int] = set()
    if arg.strip():
        for tok in arg.split(','):
            tok = tok.strip()
            if not tok:
                continue
            try:
                ids.add(int(tok))
            except ValueError:
                pass
    return ids


def _parse_include_classes(include_arg: str, exclude_arg: str, task: str, cmin: int, cmax: int) -> List[int]:
    inc_set = _parse_id_list(include_arg)
    exc_set = _parse_id_list(exclude_arg)

    if inc_set:
        # 显式传入 include 时，使用 include 列表
        base = [c for c in sorted(inc_set) if cmin <= c <= cmax]
    else:
        # 未显式传入时：
        # - vts 任务默认排除 {1,4,13,14,17–23}
        # - 其他任务默认使用完整区间 [cmin, cmax]
        if task == 'vts':
            default_excluded = {1, 4, 13, 14, 17, 18, 19, 20, 21, 22, 23}
            base = [c for c in range(cmin, cmax + 1) if c not in default_excluded]
        else:
            base = list(range(cmin, cmax + 1))

    # 应用显式排除列表
    base = [c for c in base if c not in exc_set]
    return base


def _keep_by_rules(npz_path: Path, include_classes: List[int], smin: int, smax: int) -> bool:
    # 类ID从父目录名解析
    try:
        cls_id = int(npz_path.parent.name)
    except Exception:
        return False
    if cls_id not in include_classes:
        return False
    # 自然序号（文件名前缀）过滤
    sn = _stem_num(npz_path.stem)
    if sn is None:
        return False
    if sn < smin or sn > smax:
        return False
    return True


def collect_paths(root: Path, include_classes: List[int], smin: int, smax: int):
    items = []
    for f in _iter_npz_files(root):
        if _keep_by_rules(f, include_classes, smin, smax):
            items.append(str(f.resolve()))
    return items


def _suffix_for_task(task: str) -> str:
    return {
        'classification': '_cls',
        'fcs': '_fcs',
        'vts': '_vts',
    }.get(task, '')


def _group_split_by_stem_blocks(items: List[str], block_size: int, train_blocks: int, val_blocks: int):
    """按文件名数值前缀 stem 分组，连续 block_size 个 stem 组成一个 Block，
    循环模式为 train_blocks 个训练块 + val_blocks 个验证块，重复直到覆盖全部 stem。
    同一 stem 的所有文件（不同类）进入同一集合，防止泄露。
    """
    # stem -> paths
    stem_to_paths: Dict[int, List[str]] = {}
    for p in items:
        sn = _stem_num(Path(p).stem)
        if sn is None:
            continue
        stem_to_paths.setdefault(sn, []).append(p)

    stems = sorted(stem_to_paths.keys())
    train_items: List[str] = []
    val_items: List[str] = []

    if block_size <= 0:
        block_size = 20
    cycle = max(1, train_blocks + val_blocks)

    # 遍历按 block 分段的 stem 序列
    for i in range(0, len(stems), block_size):
        block_index = i // block_size
        in_train = (block_index % cycle) < train_blocks
        block_stems = stems[i: i + block_size]
        for sn in block_stems:
            if in_train:
                train_items.extend(stem_to_paths[sn])
            else:
                val_items.extend(stem_to_paths[sn])

    return train_items, val_items, stems


def _npz_to_json(npz_path: Path, label_root: Path) -> Path:
    cls_dir = npz_path.parent.name
    stem = npz_path.stem
    return label_root / cls_dir / f"{stem}.json"


def _json_shapes_count(json_path: Path) -> int:
    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
        shapes = data.get('shapes', [])
        return len(shapes) if isinstance(shapes, list) else 0
    except Exception:
        return 0


def _map_items_to_json_lists(train_npz: List[str], val_npz: List[str], label_root: Path, strict_only: bool):
    cov = {
        'label_root': str(label_root.resolve()),
        'train': {'input_npz': len(train_npz), 'json_missing': 0, 'json_empty': 0, 'kept': 0},
        'val':   {'input_npz': len(val_npz),   'json_missing': 0, 'json_empty': 0, 'kept': 0},
    }
    def _convert(npz_list: List[str], split: str) -> List[str]:
        out: List[str] = []
        for p in npz_list:
            jp = _npz_to_json(Path(p), label_root)
            if not jp.exists():
                cov[split]['json_missing'] += 1
                if strict_only:
                    continue
            shapes_n = _json_shapes_count(jp) if jp.exists() else 0
            if strict_only and shapes_n <= 0:
                cov[split]['json_empty'] += 1
                continue
            out.append(str(jp.resolve()))
        cov[split]['kept'] = len(out)
        return out
    train_json = _convert(train_npz, 'train')
    val_json = _convert(val_npz, 'val')
    return train_json, val_json, cov


def main():
    args = parse_args()
    random.seed(args.seed)

    # 若任务为 fcs/vts 而未显式指定 emit_json_labels，则自动输出对应 JSON 清单
    if args.emit_json_labels == 'none' and args.task in ('fcs', 'vts'):
        args.emit_json_labels = args.task

    data_root = Path(args.data_root)
    # 统一将输出目录命名为 *_json（当输出 JSON 清单时），否则使用原 run_name
    _run_name = args.run_name
    if args.emit_json_labels != 'none' and not _run_name.endswith('_json'):
        _run_name = f"{_run_name}_json"
    run_dir = Path(args.out_dir) / _run_name
    ensure_dir(run_dir)

    include_classes = _parse_include_classes(args.include_classes, args.exclude_classes, args.task, args.class_min, args.class_max)
    if not include_classes:
        print(f"No classes selected. task={args.task} class_range=[{args.class_min},{args.class_max}] include='{args.include_classes}' exclude='{args.exclude_classes}'")
        return

    items = collect_paths(data_root, include_classes, args.stem_min, args.stem_max)
    if not items:
        print(f"No items matched for classes={include_classes} stem_range=[{args.stem_min},{args.stem_max}] under {data_root}")
        return

    suffix = _suffix_for_task(args.task)

    if args.group_mode == 'block':
        train_items, val_items, stems = _group_split_by_stem_blocks(items, args.block_size, args.train_blocks, args.val_blocks)
        print(f"Using block grouping: block_size={args.block_size} train_blocks={args.train_blocks} val_blocks={args.val_blocks}; stems covered={len(stems)}")
    else:
        # 兼容原随机按文件比例划分
        random.shuffle(items)
        n_val = int(len(items) * args.val_ratio)
        val_items = items[:n_val]
        train_items = items[n_val:]
        print(f"Using random file split: val_ratio={args.val_ratio}")

    # 若需要输出 JSON 标签清单，则将 NPZ 路径映射到 JSON，并按 strict_json_only 过滤
    coverage = None
    if args.emit_json_labels != 'none':
        label_root = (data_root.parent / ('FCSLabel' if args.emit_json_labels == 'fcs' else 'VTSLabel'))
        train_items, val_items, coverage = _map_items_to_json_lists(train_items, val_items, label_root, args.strict_json_only)
        # 重新统计每类样本数（基于 JSON 路径）
        def _count_by_class_json(paths: List[str]):
            counts = {c: 0 for c in include_classes}
            for p in paths:
                try:
                    cls_id = int(Path(p).parent.name)
                    if cls_id in counts:
                        counts[cls_id] += 1
                except Exception:
                    pass
            return counts
        train_counts = _count_by_class_json(train_items)
        val_counts = _count_by_class_json(val_items)
    else:
        # 统计每类样本数（基于 NPZ 路径）
        def _count_by_class(paths: List[str]):
            counts = {c: 0 for c in include_classes}
            for p in paths:
                try:
                    cls_id = int(Path(p).parent.name)
                    if cls_id in counts:
                        counts[cls_id] += 1
                except Exception:
                    pass
            return counts
        train_counts = _count_by_class(train_items)
        val_counts = _count_by_class(val_items)

    (run_dir / f'train_list{suffix}.txt').write_text('\n'.join(train_items), encoding='utf-8')
    (run_dir / f'val_list{suffix}.txt').write_text('\n'.join(val_items), encoding='utf-8')

    # 写入类别计数汇总
    class_summary = {
        'task': args.task,
        'include_classes': include_classes,
        'stem_range': [args.stem_min, args.stem_max],
        'group_mode': args.group_mode,
        'block_size': (args.block_size if args.group_mode == 'block' else None),
        'train_blocks': (args.train_blocks if args.group_mode == 'block' else None),
        'val_blocks': (args.val_blocks if args.group_mode == 'block' else None),
        'val_ratio': (args.val_ratio if args.group_mode == 'random' else None),
        'train_counts': train_counts,
        'val_counts': val_counts,
        'train_total': len(train_items),
        'val_total': len(val_items),
        'total': (len(items) if args.emit_json_labels == 'none' else (len(train_items) + len(val_items))),
        'emit_json_labels': args.emit_json_labels,
        'strict_json_only': bool(args.strict_json_only),
    }
    (run_dir / f'class_counts{suffix}.json').write_text(json.dumps(class_summary, ensure_ascii=False, indent=2), encoding='utf-8')

    if coverage is not None:
        (run_dir / f'coverage{suffix}.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"JSON coverage summary saved to {(run_dir / f'coverage{suffix}.json')}")
        print(f"[COVERAGE] train: input_npz={coverage['train']['input_npz']} kept={coverage['train']['kept']} missing_json={coverage['train']['json_missing']} empty_json={coverage['train']['json_empty']}")
        print(f"[COVERAGE] val  : input_npz={coverage['val']['input_npz']} kept={coverage['val']['kept']} missing_json={coverage['val']['json_missing']} empty_json={coverage['val']['json_empty']}")

    print(f"Split done. task={args.task} classes={include_classes} stem=[{args.stem_min},{args.stem_max}] -> {run_dir}")
    print(f"Per-class counts saved to {(run_dir / f'class_counts{suffix}.json')}\nTrain counts: {train_counts}\nVal counts: {val_counts}")


if __name__ == '__main__':
    main()
