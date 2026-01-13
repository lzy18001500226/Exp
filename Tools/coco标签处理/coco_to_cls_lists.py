import argparse
import json
from pathlib import Path
from collections import Counter


def pick_primary_label(anns):
    # 空图 → 0（未知/背景）
    if not anns:
        return 0
    cnt = Counter()
    for a in anns:
        cid = int(a.get('category_id', 0) or 0)
        cnt[cid] += 1
    # 多标签取出现次数最多的类；若全为0或异常，回退为0
    if not cnt:
        return 0
    cid, _ = cnt.most_common(1)[0]
    return int(cid)


def build_list(images_dir: Path, coco_json: Path, out_list: Path):
    with coco_json.open('r', encoding='utf-8') as f:
        data = json.load(f)
    # 索引
    imgid_to_file = {int(im['id']): im['file_name'] for im in data.get('images', [])}
    imgid_to_anns = {}
    for ann in data.get('annotations', []):
        iid = int(ann.get('image_id'))
        imgid_to_anns.setdefault(iid, []).append(ann)
    # 逐图生成行："abs_path label"
    out_list.parent.mkdir(parents=True, exist_ok=True)
    with out_list.open('w', encoding='utf-8') as wf:
        for iid, fname in imgid_to_file.items():
            p = images_dir / fname
            label = pick_primary_label(imgid_to_anns.get(iid, []))
            wf.write(f"{str(p.resolve())} {label}\n")
    return out_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', required=True, help='PNG根目录（LabelMePNG）')
    ap.add_argument('--train_json', required=True, help='COCO train json')
    ap.add_argument('--val_json', required=True, help='COCO val json')
    ap.add_argument('--test_json', default='', help='COCO test json（可选，未提供则复用 val）')
    ap.add_argument('--out_dir', required=True, help='输出目录，生成 train.list / val.list / test.list')
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    train_list = build_list(images_dir, Path(args.train_json), out_dir / 'train.list')
    val_list = build_list(images_dir, Path(args.val_json), out_dir / 'val.list')
    test_json = Path(args.test_json) if args.test_json else None
    if test_json and test_json.exists():
        test_list = build_list(images_dir, test_json, out_dir / 'test.list')
    else:
        # 复用验证集作为测试集占位
        (out_dir / 'test.list').write_text((out_dir / 'val.list').read_text(encoding='utf-8'), encoding='utf-8')

    print(f"Written lists to: {str(out_dir.resolve())}")


if __name__ == '__main__':
    main()
