import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set


def parse_args():
    p = argparse.ArgumentParser("Split a COCO (images+annotations) into stratified train/val JSONs")
    p.add_argument('--coco', required=True, help='Path to source COCO json (e.g., coco_labeled.json)')
    p.add_argument('--out_dir', required=True, help='Output directory to write coco_train.json and coco_val.json')
    p.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio for images with annotations')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    p.add_argument('--min_val_per_class', type=int, default=1, help='Minimum number of images per class in val if possible')
    p.add_argument('--keep_empty', action='store_true', help='Also split and include empty (no-annotation) images')
    p.add_argument('--empty_val_ratio', type=float, default=None, help='Val ratio for empty images (default: same as val_ratio)')
    return p.parse_args()


def load_coco(path: Path) -> Dict:
    return json.loads(path.read_text(encoding='utf-8'))


def index_coco(coco: Dict):
    images = coco.get('images', [])
    anns = coco.get('annotations', [])
    cats = coco.get('categories', [])
    imgid_to_img = {int(im['id']): im for im in images}
    imgid_to_anns: Dict[int, List[Dict]] = defaultdict(list)
    for a in anns:
        imgid_to_anns[int(a['image_id'])].append(a)
    return imgid_to_img, imgid_to_anns, cats


def pick_primary_label(anns: List[Dict]) -> int:
    # majority label among anns; if tie, pick smallest id
    cnt = Counter(int(a['category_id']) for a in anns)
    if not cnt:
        return -1
    m = max(cnt.values())
    cands = [k for k, v in cnt.items() if v == m]
    return min(cands)


def stratified_split(pos_img_ids: List[int], imgid_to_anns: Dict[int, List[Dict]], *, val_ratio: float, seed: int, min_val: int) -> Tuple[Set[int], Set[int]]:
    # group by primary label
    by_label: Dict[int, List[int]] = defaultdict(list)
    for iid in pos_img_ids:
        label = pick_primary_label(imgid_to_anns[iid])
        by_label[label].append(iid)
    rng = random.Random(seed)
    train_ids: Set[int] = set()
    val_ids: Set[int] = set()
    for lab, ids in by_label.items():
        ids = ids[:]
        rng.shuffle(ids)
        n = len(ids)
        n_val = int(round(n * val_ratio))
        if n_val < min_val and n >= (min_val + 1):
            n_val = min_val
        if n_val >= n:  # avoid taking all into val
            n_val = max(1, n - 1)
        val_sel = set(ids[:n_val])
        trn_sel = set(ids[n_val:])
        val_ids |= val_sel
        train_ids |= trn_sel
    return train_ids, val_ids


def random_split(ids: List[int], *, ratio: float, seed: int) -> Tuple[Set[int], Set[int]]:
    rng = random.Random(seed + 999)
    ids = ids[:]
    rng.shuffle(ids)
    n = len(ids)
    n_val = int(round(n * ratio))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def build_coco_subset(src: Dict, img_ids: Set[int]) -> Dict:
    img_ids = set(int(i) for i in img_ids)
    images = [im for im in src.get('images', []) if int(im['id']) in img_ids]
    img_ids_kept = {int(im['id']) for im in images}
    anns = [a for a in src.get('annotations', []) if int(a['image_id']) in img_ids_kept]
    # keep categories/info/licenses as-is for consistency
    return {
        'info': src.get('info', {}),
        'licenses': src.get('licenses', []),
        'images': images,
        'annotations': anns,
        'categories': src.get('categories', []),
    }


def main():
    args = parse_args()
    src_path = Path(args.coco)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco(src_path)
    imgid_to_img, imgid_to_anns, _ = index_coco(coco)

    pos_ids = [iid for iid, al in imgid_to_anns.items() if len(al) > 0]
    empty_ids = [iid for iid in imgid_to_img.keys() if iid not in imgid_to_anns or len(imgid_to_anns[iid]) == 0]

    tr_pos, va_pos = stratified_split(pos_ids, imgid_to_anns, val_ratio=max(0.0, min(1.0, args.val_ratio)), seed=args.seed, min_val=max(0, args.min_val_per_class))

    tr_ids = set(tr_pos)
    va_ids = set(va_pos)

    if args.keep_empty and len(empty_ids) > 0:
        er = args.empty_val_ratio if (args.empty_val_ratio is not None) else args.val_ratio
        tr_emp, va_emp = random_split(empty_ids, ratio=max(0.0, min(1.0, er)), seed=args.seed)
        tr_ids |= tr_emp
        va_ids |= va_emp

    # build subsets
    coco_train = build_coco_subset(coco, tr_ids)
    coco_val = build_coco_subset(coco, va_ids)

    (out_dir / 'coco_train.json').write_text(json.dumps(coco_train, ensure_ascii=False))
    (out_dir / 'coco_val.json').write_text(json.dumps(coco_val, ensure_ascii=False))

    print(f"Split done. train_images={len(coco_train['images'])} val_images={len(coco_val['images'])} | train_anns={len(coco_train['annotations'])} val_anns={len(coco_val['annotations'])}")


if __name__ == '__main__':
    main()
