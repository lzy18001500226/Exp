import argparse
import json
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser("Convert COCO images list to DSFNet detection list (LabelMe JSON paths)")
    p.add_argument('--coco_train', type=str, required=True, help='Path to coco_train.json')
    p.add_argument('--coco_val', type=str, required=False, default='', help='Path to coco_val.json')
    p.add_argument('--out_train_list', type=str, required=False, default='', help='Output train list path (default: <coco_dir>/train_json.list)')
    p.add_argument('--out_val_list', type=str, required=False, default='', help='Output val list path (default: <coco_dir>/val_json.list)')
    p.add_argument('--check_json_exists', action='store_true', help='Only include items whose label JSON exists next to the image')
    return p.parse_args()


def coco_to_list(coco_path: Path, out_list: Path, check_json_exists: bool = False) -> int:
    data = json.loads(Path(coco_path).read_text(encoding='utf-8'))
    images = data.get('images', [])
    count = 0
    with out_list.open('w', encoding='utf-8') as f:
        for im in images:
            fn = im.get('file_name')
            if not fn:
                continue
            jp = Path(fn).with_suffix('.json')
            if check_json_exists and not jp.is_file():
                # skip if JSON not found (optional)
                continue
            f.write(str(jp) + "\n")
            count += 1
    return count


def main():
    args = parse_args()
    coco_train = Path(args.coco_train)
    coco_val = Path(args.coco_val) if args.coco_val else None
    out_train = Path(args.out_train_list) if args.out_train_list else (coco_train.parent / 'train_json.list')
    out_val = Path(args.out_val_list) if args.out_val_list else ((coco_val.parent / 'val_json.list') if coco_val else None)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    if out_val:
        out_val.parent.mkdir(parents=True, exist_ok=True)

    ntr = coco_to_list(coco_train, out_train, check_json_exists=args.check_json_exists)
    print(f"[OK] Wrote {ntr} lines to {out_train}")
    if coco_val and out_val:
        nva = coco_to_list(coco_val, out_val, check_json_exists=args.check_json_exists)
        print(f"[OK] Wrote {nva} lines to {out_val}")


if __name__ == '__main__':
    main()
