import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2


def parse_args():
    p = argparse.ArgumentParser("Convert LabelMe rectangle annotations to COCO JSON")
    p.add_argument('--dirs', dest='dirs', action='append', required=True,
                   help='Image/annotation root directory (can repeat). Will search **/*.png and matching *.json')
    p.add_argument('--out', dest='out_json', required=True, help='Output COCO json path')
    p.add_argument('--skip_unknown', action='store_true',
                   help='Skip paths which include "unknown" in their directory name')
    p.add_argument('--allow_empty', action='store_true',
                   help='Include images even when there is no annotation')
    p.add_argument('--start_category_id', type=int, default=1,
                   help='Category id start value (default 1)')
    p.add_argument('--infer_label_from_folder_when_missing', action='store_true',
                   help='If a shape label is missing/empty, fallback to parent folder name as label')
    p.add_argument('--image_ext', type=str, default='.png', help='Image extension to search for')
    p.add_argument('--add_extra_fields', action='store_true',
                   help='Add extra fields (npy_path, fold, split) into images entries if index.csv is present')
    # 不再使用“最多收录”，默认禁用
    p.add_argument('--max_per_class', type=int, default=0,
                   help='Maximum number of images per category (0 for no limit)')
    # 按类固定索引范围选择（默认类1-23，索引0-50）
    p.add_argument('--gate_classes', type=str, default='1-23',
                   help='Numeric category ids to include, inclusive range (e.g., 1-23)')
    p.add_argument('--index_range', type=str, default='0-50',
                   help='Per-class sample index to include, inclusive (e.g., 0-50)')
    p.add_argument('--include_no_index', action='store_true',
                   help='Include images without a parsable numeric index in filename')
    # 背景类（文件夹名或标签名为“0”），可全部收录（允许无标注）
    p.add_argument('--background_label', type=str, default='0',
                   help='Folder/category name treated as background')
    p.add_argument('--bg_include_all', action='store_true',
                   help='Include background images even without annotations')
    # 新增：自动划分与生成 cls_lists
    p.add_argument('--make_splits', action='store_true', help='Also generate coco_train/val/test.json by a split method')
    p.add_argument('--split_mode', type=str, default='stratified', choices=['stratified', 'deterministic'],
                   help='Split mode: stratified (class-balanced) or deterministic (sorted by filename)')
    p.add_argument('--group_by', type=str, default='parent', choices=['none', 'parent'],
                   help='Grouping unit for split. parent: keep images under the same parent folder in the same split')
    p.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio (0-1)')
    p.add_argument('--test_split', type=float, default=0.0, help='Test split ratio (0-1)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for stratified split')
    p.add_argument('--min_val_per_class', type=int, default=1, help='Min images per class in val if possible')
    p.add_argument('--keep_empty', action='store_true', help='Also split empty (no-annotation) images')
    p.add_argument('--empty_val_ratio', type=float, default=None, help='Val ratio for empty images (default: val_split)')
    p.add_argument('--write_cls_lists', action='store_true', help='Also write train/val/test.list under cls_lists')
    p.add_argument('--cls_lists_dir', type=str, default='', help='Directory to write *.list (default: <out>/../cls_lists)')
    return p.parse_args()


# ---- helpers ----

def _is_rect_shape(shape: Dict) -> bool:
    return shape.get('shape_type', 'rectangle') == 'rectangle' and 'points' in shape and len(shape['points']) == 2


def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float, float, float, float]:
    # ensure tl->br, clip to image
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0.0, min(float(W - 1), float(x1)))
    y1 = max(0.0, min(float(H - 1), float(y1)))
    x2 = max(0.0, min(float(W - 1), float(x2)))
    y2 = max(0.0, min(float(H - 1), float(y2)))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return x1, y1, w, h


def _read_image_size(img_path: Path) -> Tuple[int, int]:
    im = cv2.imread(str(img_path))
    if im is None:
        raise RuntimeError(f'Cannot read image: {img_path}')
    h, w = im.shape[:2]
    return w, h


def _parse_range(s: str, fallback: Tuple[int, int]) -> Tuple[int, int]:
    try:
        a, b = s.split('-')
        return int(a.strip()), int(b.strip())
    except Exception:
        return fallback


def _extract_sample_id(img_path: Path) -> Optional[int]:
    # 优先从文件名末尾数字提取，如 xxx_0007.png -> 7
    stem = img_path.stem
    m = list(re.finditer(r'(\d+)', stem))
    if m:
        try:
            return int(m[-1].group(1))
        except Exception:
            pass
    # 其次尝试从父目录名提取数字
    for part in reversed(img_path.parts):
        if part.isdigit():
            try:
                return int(part)
            except Exception:
                continue
    return None


def _label_to_int(lbl: str) -> Optional[int]:
    try:
        return int(lbl)
    except Exception:
        return None


# ---- main convert ----

def convert_dirs_to_coco(dirs: List[Path], out_json: Path, *, skip_unknown: bool, allow_empty: bool,
                         start_category_id: int, infer_label_from_folder_when_missing: bool,
                         image_ext: str, add_extra_fields: bool, max_per_class: int,
                         gate_classes: Tuple[int, int], index_range: Tuple[int, int], include_no_index: bool,
                         background_label: str, bg_include_all: bool):
    # scan images
    images_list: List[Path] = []
    for root in dirs:
        if skip_unknown and 'unknown' in root.as_posix().lower():
            continue
        for p in root.rglob(f'*{image_ext}'):
            if skip_unknown and 'unknown' in p.as_posix().lower():
                continue
            images_list.append(p)
    images_list.sort()

    # maps
    cat_name_to_id: Dict[str, int] = {}
    next_cat_id = start_category_id
    per_class_img_count: Dict[str, int] = {}

    images_data = []
    annotations = []
    ann_id = 1

    start_cls, end_cls = gate_classes
    start_idx, end_idx = index_range

    for idx, img_path in enumerate(images_list, 1):
        json_path = img_path.with_suffix('.json')
        file_name = str(img_path)
        try:
            W, H = _read_image_size(img_path)
        except Exception as e:
            print(f'[WARN] skip unreadable image: {img_path} ({e})')
            continue

        shapes: List[Dict] = []
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding='utf-8', errors='ignore'))
                shapes = data.get('shapes', []) or []
                W = int(data.get('imageWidth', W) or W)
                H = int(data.get('imageHeight', H) or H)
            except Exception as e:
                print(f'[WARN] bad json: {json_path} ({e})')
                shapes = []
        else:
            # 无标注文件
            pass

        # 解析有效矩形
        raw_valid: List[Tuple[str, List[float]]] = []
        raw_labels: List[str] = []
        for sh in shapes:
            if not _is_rect_shape(sh):
                continue
            label = (sh.get('label') or '').strip()
            if not label and infer_label_from_folder_when_missing:
                label = img_path.parent.name
            if not label:
                continue
            pts = sh['points']
            try:
                (x1, y1), (x2, y2) = pts
                x, y, w, h = _xyxy_to_xywh(float(x1), float(y1), float(x2), float(y2), W, H)
                if w <= 0 or h <= 0:
                    continue
                raw_valid.append((label, [x, y, w, h]))
                if label not in raw_labels:
                    raw_labels.append(label)
            except Exception:
                continue

        # 背景文件夹特判：若无有效标注且父目录名为背景且开启 bg_include_all，则收录（作为负样本），否则按 allow_empty 决定
        parent_label = img_path.parent.name
        if not raw_valid:
            if bg_include_all and parent_label == background_label:
                # 收录为空标注图像
                img_id = idx
                images_data.append({'id': img_id, 'file_name': file_name, 'width': W, 'height': H})
                continue
            if not allow_empty:
                continue

        # 仅保留允许类（1..23）的标注
        kept_anns: List[Tuple[str, List[float]]] = []
        kept_labels: List[str] = []
        for lbl, bb in raw_valid:
            li = _label_to_int(lbl)
            if li is None:
                continue
            if start_cls <= li <= end_cls:
                kept_anns.append((lbl, bb))
                if lbl not in kept_labels:
                    kept_labels.append(lbl)
        if not kept_anns:
            # 没有落入允许类的标注，若父目录是背景且允许全收，则可作为负样本收录；否则丢弃
            if bg_include_all and parent_label == background_label:
                img_id = idx
                images_data.append({'id': img_id, 'file_name': file_name, 'width': W, 'height': H})
            continue

        # 索引过滤：仅当图像属于允许类时才应用（默认 0..50 包含端点）
        sample_id = _extract_sample_id(img_path)
        if sample_id is None and not include_no_index:
            continue
        if sample_id is not None and not (start_idx <= sample_id <= end_idx):
            continue

        # image entry
        img_id = idx
        img_entry = {
            'id': img_id,
            'file_name': file_name,
            'width': W,
            'height': H,
        }
        if add_extra_fields:
            parts = img_path.parts
            split = ''
            for part in parts:
                if '-' in part:
                    split = part
                    break
            img_entry['fold_or_split'] = split
            img_entry['label_folder'] = parent_label
        images_data.append(img_entry)

        # 写入标注（仅允许类）
        for lbl, bbox in kept_anns:
            if lbl not in cat_name_to_id:
                cat_name_to_id[lbl] = next_cat_id
                next_cat_id += 1
            cat_id = cat_name_to_id[lbl]
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': bbox,
                'area': float(bbox[2] * bbox[3]),
                'iscrowd': 0,
                'segmentation': [],
            })
            ann_id += 1

        # 记录每类已收录图片数（当前不做上限控制）
        for lbl in kept_labels:
            per_class_img_count[lbl] = per_class_img_count.get(lbl, 0) + 1

    # categories
    categories = [
        {'id': cid, 'name': name, 'supercategory': 'object'}
        for name, cid in sorted(cat_name_to_id.items(), key=lambda x: x[1])
    ]

    coco = {
        'info': {'description': 'LabelMe to COCO', 'version': '1.0'},
        'licenses': [],
        'images': images_data,
        'annotations': annotations,
        'categories': categories,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco, ensure_ascii=False))
    print(f'Done. images={len(images_data)} anns={len(annotations)} cats={len(categories)} -> {out_json}')

    # 新增：可选生成 train/val/test 划分与 cls_lists
    # 将 images/annotations 建立索引
    id_to_img = {int(im['id']): im for im in images_data}
    imgid_to_anns: Dict[int, List[Dict]] = {}
    for a in annotations:
        iid = int(a['image_id'])
        imgid_to_anns.setdefault(iid, []).append(a)

    # 提取运行参数（通过 main() 注入全局）
    val_split = globals().get('_VAL_SPLIT', 0.2)
    test_split = globals().get('_TEST_SPLIT', 0.0)
    split_mode = globals().get('_SPLIT_MODE', 'stratified')
    group_by = globals().get('_GROUP_BY', 'parent')
    seed = globals().get('_SEED', 42)
    min_val_per_class = globals().get('_MIN_VAL_PER_CLASS', 1)
    keep_empty = globals().get('_KEEP_EMPTY', False)
    empty_val_ratio = globals().get('_EMPTY_VAL_RATIO', None)
    make_splits = globals().get('_MAKE_SPLITS', False)
    write_cls_lists = globals().get('_WRITE_CLS_LISTS', False)
    cls_lists_dir_arg = globals().get('_CLS_LISTS_DIR', '')

    if 'args' in globals():
        a = globals()['args']
        val_split = getattr(a, 'val_split', val_split)
        test_split = getattr(a, 'test_split', test_split)
        split_mode = getattr(a, 'split_mode', split_mode)
        seed = getattr(a, 'seed', seed)
        min_val_per_class = getattr(a, 'min_val_per_class', min_val_per_class)
        keep_empty = getattr(a, 'keep_empty', keep_empty)
        empty_val_ratio = getattr(a, 'empty_val_ratio', empty_val_ratio)
        make_splits = getattr(a, 'make_splits', make_splits)
        write_cls_lists = getattr(a, 'write_cls_lists', write_cls_lists)
        cls_lists_dir_arg = getattr(a, 'cls_lists_dir', cls_lists_dir_arg)

    if make_splits or write_cls_lists:
        from collections import defaultdict
        import random
        base_dir = out_json.parent

        def primary_label(img_id: int) -> int:
            anns = imgid_to_anns.get(int(img_id), [])
            if not anns:
                return -1
            cnt: Dict[int, int] = defaultdict(int)
            for a in anns:
                cnt[int(a['category_id'])] += 1
            m = max(cnt.values())
            cands = [k for k, v in cnt.items() if v == m]
            return int(min(cands))

        val_ratio = max(0.0, min(1.0, float(val_split)))
        test_ratio = max(0.0, min(1.0 - val_ratio, float(test_split)))
        rng = random.Random(int(seed))

        train_ids: set = set()
        val_ids: set = set()
        test_ids: set = set()

        if group_by == 'parent':
            # 1) 构建父目录分组
            groups: Dict[str, List[int]] = defaultdict(list)
            for iid, im in id_to_img.items():
                parent = Path(im['file_name']).parent.name
                groups[parent].append(int(iid))

            # 2) 标注/空组划分
            labeled_groups: Dict[str, List[int]] = {}
            empty_groups: Dict[str, List[int]] = {}
            for g, ids in groups.items():
                has_ann = any(len(imgid_to_anns.get(int(x), [])) > 0 for x in ids)
                if has_ann:
                    labeled_groups[g] = ids
                else:
                    empty_groups[g] = ids

            # 3) 分层（按组的主类）或确定性分配
            if split_mode == 'deterministic':
                # 组名排序后按比例切分，整组进入同一 split
                gnames = sorted(groups.keys())
                n_all = len(gnames)
                n_val_g = int(round(n_all * val_ratio))
                n_test_g = int(round(n_all * test_ratio))
                n_val_g = min(n_val_g, n_all)
                n_test_g = min(n_test_g, max(0, n_all - n_val_g))
                val_g = set(gnames[:n_val_g])
                test_g = set(gnames[n_val_g:n_val_g + n_test_g])
                train_g = set(gnames[n_val_g + n_test_g:])
                for g in val_g:
                    val_ids.update(groups[g])
                for g in test_g:
                    test_ids.update(groups[g])
                for g in train_g:
                    train_ids.update(groups[g])
            else:
                # 分层：按每个“有标注组”的主类（该组所有带标注图片的多数票类别）进行分配
                by_label_groups: Dict[int, List[str]] = defaultdict(list)
                for g, ids in labeled_groups.items():
                    # 统计该组带标注图片的类别计数
                    cnt: Dict[int, int] = defaultdict(int)
                    for iid in ids:
                        pl = primary_label(iid)
                        if pl != -1:
                            cnt[pl] += 1
                    if not cnt:
                        lab = -1
                    else:
                        m = max(cnt.values())
                        cands = [k for k, v in cnt.items() if v == m]
                        lab = int(min(cands))
                    by_label_groups[lab].append(g)

                for lab, gnames in by_label_groups.items():
                    gnames = gnames[:]
                    rng.shuffle(gnames)
                    n = len(gnames)
                    if n == 0:
                        continue
                    n_val_g = int(round(n * val_ratio))
                    # 至少保证每类在 val 中有组（若可行）
                    if lab != -1 and n_val_g < int(min_val_per_class) and n >= (int(min_val_per_class) + 1):
                        n_val_g = int(min_val_per_class)
                    n_test_g = int(round(n * test_ratio))
                    if n_val_g + n_test_g >= n:
                        n_test_g = max(0, n - n_val_g - 1)
                    val_g = gnames[:n_val_g]
                    test_g = gnames[n_val_g:n_val_g + n_test_g]
                    train_g = gnames[n_val_g + n_test_g:]
                    for g in val_g:
                        val_ids.update(labeled_groups[g])
                    for g in test_g:
                        test_ids.update(labeled_groups[g])
                    for g in train_g:
                        train_ids.update(labeled_groups[g])

                # 空组（全为空标注）按比例随机分配
                if keep_empty and len(empty_groups) > 0:
                    gnames = list(empty_groups.keys())
                    rng.shuffle(gnames)
                    n = len(gnames)
                    er = empty_val_ratio if (empty_val_ratio is not None) else val_ratio
                    er = max(0.0, min(1.0, float(er)))
                    n_val_g = int(round(n * er))
                    n_test_g = int(round(n * test_ratio))
                    if n_val_g + n_test_g > n:
                        n_test_g = max(0, n - n_val_g)
                    val_g = gnames[:n_val_g]
                    test_g = gnames[n_val_g:n_val_g + n_test_g]
                    train_g = gnames[n_val_g + n_test_g:]
                    for g in val_g:
                        val_ids.update(empty_groups[g])
                    for g in test_g:
                        test_ids.update(empty_groups[g])
                    for g in train_g:
                        train_ids.update(empty_groups[g])
        else:
            # 原先按图片级别的 deterministic/stratified 逻辑
            if split_mode == 'deterministic':
                imgs_sorted = sorted(images_data, key=lambda x: x['file_name'])
                n_all = len(imgs_sorted)
                n_val = int(round(n_all * val_ratio))
                n_test = int(round(n_all * test_ratio))
                n_val = min(n_val, n_all)
                n_test = min(n_test, max(0, n_all - n_val))
                val_ids = {int(im['id']) for im in imgs_sorted[:n_val]}
                test_ids = {int(im['id']) for im in imgs_sorted[n_val:n_val + n_test]}
                train_ids = {int(im['id']) for im in imgs_sorted[n_val + n_test:]}
            else:
                # 按图片主类分层
                pos_ids = [iid for iid in id_to_img.keys() if len(imgid_to_anns.get(int(iid), [])) > 0]
                empty_ids = [iid for iid in id_to_img.keys() if len(imgid_to_anns.get(int(iid), [])) == 0]
                by_label: Dict[int, List[int]] = defaultdict(list)
                for iid in pos_ids:
                    by_label[primary_label(iid)].append(int(iid))
                for lab, ids in by_label.items():
                    ids = ids[:]
                    rng.shuffle(ids)
                    n = len(ids)
                    if n == 0:
                        continue
                    n_val = int(round(n * val_ratio))
                    if lab != -1 and n_val < int(min_val_per_class) and n >= (int(min_val_per_class) + 1):
                        n_val = int(min_val_per_class)
                    n_test = int(round(n * test_ratio))
                    if n_val + n_test >= n:
                        n_test = max(0, n - n_val - 1)
                    val_sel = ids[:n_val]
                    test_sel = ids[n_val:n_val + n_test]
                    train_sel = ids[n_val + n_test:]
                    val_ids.update(val_sel)
                    test_ids.update(test_sel)
                    train_ids.update(train_sel)
                if keep_empty and empty_ids:
                    ids = empty_ids[:]
                    rng.shuffle(ids)
                    n = len(ids)
                    er = empty_val_ratio if (empty_val_ratio is not None) else val_ratio
                    er = max(0.0, min(1.0, float(er)))
                    tr_emp_n = int(round(n * (1.0 - er - test_ratio)))
                    va_emp_n = int(round(n * er))
                    te_emp_n = max(0, n - tr_emp_n - va_emp_n)
                    tr_emp = ids[:tr_emp_n]
                    va_emp = ids[tr_emp_n:tr_emp_n + va_emp_n]
                    te_emp = ids[tr_emp_n + va_emp_n:tr_emp_n + va_emp_n + te_emp_n]
                    train_ids.update(tr_emp)
                    val_ids.update(va_emp)
                    test_ids.update(te_emp)

        def _filter_split(img_ids: set):
            ims = [id_to_img[i] for i in img_ids]
            anns = [a for a in annotations if int(a['image_id']) in img_ids]
            return ims, anns

        if make_splits:
            tr_ims, tr_anns = _filter_split(train_ids)
            va_ims, va_anns = _filter_split(val_ids)
            te_ims, te_anns = _filter_split(test_ids)
            # 写 JSON
            (base_dir / 'coco_train.json').write_text(json.dumps({
                'info': coco['info'], 'licenses': [], 'images': tr_ims, 'annotations': tr_anns, 'categories': categories
            }, ensure_ascii=False))
            (base_dir / 'coco_val.json').write_text(json.dumps({
                'info': coco['info'], 'licenses': [], 'images': va_ims, 'annotations': va_anns, 'categories': categories
            }, ensure_ascii=False))
            if len(test_ids) > 0:
                (base_dir / 'coco_test.json').write_text(json.dumps({
                    'info': coco['info'], 'licenses': [], 'images': te_ims, 'annotations': te_anns, 'categories': categories
                }, ensure_ascii=False))

        if write_cls_lists:
            cls_dir = Path(cls_lists_dir_arg) if cls_lists_dir_arg else (base_dir / 'cls_lists')
            cls_dir.mkdir(parents=True, exist_ok=True)
            def _write_list(p: Path, img_ids: set):
                lines = [id_to_img[i]['file_name'] for i in sorted(img_ids)]
                p.write_text("\n".join(lines), encoding='utf-8')
            _write_list(cls_dir / 'train.list', train_ids)
            _write_list(cls_dir / 'val.list', val_ids)
            _write_list(cls_dir / 'test.list', test_ids)


def main():
    global args
    args = parse_args()
    # 将部分参数暴露为全局以便函数内部访问
    globals()['_VAL_SPLIT'] = args.val_split
    globals()['_TEST_SPLIT'] = args.test_split
    globals()['_SPLIT_MODE'] = args.split_mode
    globals()['_GROUP_BY'] = args.group_by
    globals()['_SEED'] = args.seed
    globals()['_MIN_VAL_PER_CLASS'] = args.min_val_per_class
    globals()['_KEEP_EMPTY'] = args.keep_empty
    globals()['_EMPTY_VAL_RATIO'] = args.empty_val_ratio
    globals()['_MAKE_SPLITS'] = args.make_splits
    globals()['_WRITE_CLS_LISTS'] = args.write_cls_lists
    globals()['_CLS_LISTS_DIR'] = args.cls_lists_dir

    dirs = [Path(d) for d in args.dirs]
    out_json = Path(args.out_json)
    convert_dirs_to_coco(
        dirs,
        out_json,
        skip_unknown=args.skip_unknown,
        allow_empty=args.allow_empty,
        start_category_id=args.start_category_id,
        infer_label_from_folder_when_missing=args.infer_label_from_folder_when_missing,
        image_ext=args.image_ext,
        add_extra_fields=args.add_extra_fields,
        max_per_class=args.max_per_class,
        gate_classes=_parse_range(args.gate_classes, (1, 23)),
        index_range=_parse_range(args.index_range, (0, 50)),
        include_no_index=args.include_no_index,
        background_label=args.background_label,
        bg_include_all=args.bg_include_all,
    )


if __name__ == '__main__':
    main()
