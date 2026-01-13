import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

# 允许以 `common.io` 形式导入
THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[2]  # D:/Exp/Baseline
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common.io import ensure_yolo_dataset, build_dataset_yaml_only  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='YOLOv8 训练（缓存转换 PNG/TXT）')
    p.add_argument('--data-root', required=True,
                   help='列表文件所在目录（包含 train_json.list 和 val_json.list）')
    p.add_argument('--workdir', default='D:/Exp/Baseline/yolov8/results/fcs',
                   help='训练结果输出目录')
    p.add_argument('--cache-dir', default='D:/Exp/Baseline/yolov8/cache/fcs',
                   help='缓存输出目录（PNG/TXT 和 dataset.yaml）')
    p.add_argument('--epochs', type=int, default=120, help='训练轮数')
    p.add_argument('--imgsz', type=int, default=512, help='输入图像尺寸')
    p.add_argument('--model', default='D:/Exp/Model/Basic/YOLO/YOLOv8/yolov8m.pt', help='预训练模型路径')
    p.add_argument('--exclude-classes', default='13,14', help='排除的类别 ID（逗号分隔）')
    p.add_argument('--seed', type=int, default=3407, help='随机种子')
    p.add_argument('--train-list-name', default='train_json.list', help='训练列表文件名')
    p.add_argument('--val-list-name', default='val_json.list', help='验证列表文件名')
    p.add_argument('--assume-ready', action='store_true', default=False,
                   help='假定 PNG/TXT 已就绪，仅生成 dataset.yaml 并直接训练（不做任何转换）')
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    exclude = set(int(x) for x in args.exclude_classes.split(',') if x.strip() != '')

    if args.assume_ready:
        print("[INFO] 假定数据已就绪：跳过转换，仅生成 YAML...")
        dataset_yaml = build_dataset_yaml_only(cache_dir)
    else:
        print("[INFO] 使用缓存模式（将 JSON/NPZ 转换为 PNG/TXT 文件），仅首次花时间，后续增量跳过...")
        dataset_yaml = ensure_yolo_dataset(
            data_root=data_root,
            cache_dir=cache_dir,
            split_files=(args.train_list_name, args.val_list_name),
            exclude_classes=exclude,
            imgsz=args.imgsz,
        )

    print(f"[INFO] Dataset YAML: {dataset_yaml}")
    print(f"[INFO] 开始 YOLOv8 训练...")
    
    model = YOLO(args.model)
    results = model.train(
        data=str(dataset_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        project=args.workdir,
        name='fcs_yolov8',
        patience=30,
        warmup_epochs=5,
        lr0=0.01,
        workers=4,
        seed=args.seed,
        plots=False,
    )
    
    print("[INFO] 训练完成！")
    return results


if __name__ == '__main__':
    main()

