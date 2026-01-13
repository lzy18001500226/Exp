import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# 配置：目标目录与最多显示文件数
TARGET_DIR = Path(r'G:\Exp\data_processing\Data-512NPZ-merged\1')  # 默认目录，可被命令行覆盖
MAX_FILES = 10
OUTPUT_DIR = Path('npz_preview')
OUTPUT_DIR.mkdir(exist_ok=True)


def add_html_entry(index_lines, text: str, img_name: str) -> None:
    index_lines.append(f'<li><p>{text}</p><img src="{img_name}" style="max-width:300px"></li>')


def load_npz(file_path: Path, allow_pickle: bool = False):
    """加载 npz 文件，返回 NpzFile 对象。某些含 object/string 的压缩字段需要 allow_pickle=True。"""
    return np.load(file_path, allow_pickle=allow_pickle)

def guess_main_array(npz_obj: np.lib.npyio.NpzFile):
    """根据形状和维度猜测最主要的数组键。优先选择维度>=2或包含最大元素数量的数组。"""
    best_key = None
    best_size = -1
    for k in npz_obj.files:
        arr = npz_obj[k]
        size = arr.size
        if arr.ndim >= 2 and size > best_size:
            best_size = size
            best_key = k
    if best_key is None:
        # fallback: 选择最大size的
        for k in npz_obj.files:
            arr = npz_obj[k]
            size = arr.size
            if size > best_size:
                best_size = size
                best_key = k
    return best_key

def visualize_array(arr: np.ndarray, title: str, out_path: Path):
    plt.figure(figsize=(4,4))
    if arr.ndim == 2:
        plt.imshow(arr, cmap='hot')
        plt.colorbar()
    elif arr.ndim == 3:
        # 如果是 (H,W,C) 或 (C,H,W)，做简单判断
        if arr.shape[2] in (1,3,4):
            # 假设 (H,W,C)
            show = arr
            if arr.shape[2] == 1:
                show = arr[:,:,0]
                plt.imshow(show, cmap='hot')
            else:
                # 归一化到0-1
                show = show.astype(float)
                show = (show - show.min()) / (show.max() - show.min() + 1e-8)
                plt.imshow(show)
        else:
            # 取第一通道
            show = arr[0]
            if show.ndim == 2:
                plt.imshow(show, cmap='hot')
            else:
                # 再次兜底：展示第一张二维切片
                while show.ndim > 2:
                    show = show[0]
                plt.imshow(show, cmap='hot')
        plt.colorbar()
    else:
        # 高维或者一维：展示直方图
        flat = arr.reshape(-1)
        plt.hist(flat, bins=50)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def visualize_with_bboxes(image: np.ndarray, bboxes: np.ndarray, title: str, out_path: Path):
    """可视化图像并绘制边界框。bboxes: (N,5) [x_min,y_min,x_max,y_max,label]"""
    plt.figure(figsize=(5,5))
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        show = image
        if show.dtype != np.uint8:
            show = show.astype(float)
            show = (show - show.min()) / (show.max() - show.min() + 1e-8)
        plt.imshow(show)
    for box in bboxes:
        x_min, y_min, x_max, y_max, label = box.tolist()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(x_min, max(0, y_min - 5), str(label), color='yellow', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.5))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()

def visualize_feature_planes(feature_stack: np.ndarray, feature_names: np.ndarray, limit: int, stem: str, out_dir: Path, html_lines) -> None:
    if feature_stack.ndim != 3 or feature_stack.size == 0:
        return
    count = int(min(max(0, limit), feature_stack.shape[0]))
    if count <= 0:
        return
    for idx in range(count):
        arr = feature_stack[idx]
        name = str(feature_names[idx]) if idx < len(feature_names) else f"feat_{idx}"
        safe_name = name.replace('/', '_')
        out_name = f"{stem}_feat_{idx}_{safe_name}.png"
        out_path = out_dir / out_name
        visualize_array(arr, f"{stem}::{name}", out_path)
        add_html_entry(html_lines, f"{stem} - feature[{idx}] {name}", out_name)


def parse_args():
    parser = argparse.ArgumentParser(description='可视化 NPZ 文件（支持频谱与bbox）')
    parser.add_argument('--dir', type=str, default=str(TARGET_DIR), help='NPZ文件目录')
    parser.add_argument('--max', type=int, default=MAX_FILES, help='最多显示文件数')
    parser.add_argument('--out', type=str, default=str(OUTPUT_DIR), help='输出图像与HTML目录')
    parser.add_argument('--prefer-image', action='store_true', help='如果同时存在 image 与其它数组，优先显示 image + bbox')
    parser.add_argument('--allow-pickle', action='store_true', help='当 NPZ 内含 object/string 数组需要开启，否则会报错')
    parser.add_argument('--show-features', action='store_true', help='若存在 feature_stack，则额外渲染部分通道')
    parser.add_argument('--feature-channels', type=int, default=3, help='可视化的 feature_stack 通道数量上限 (默认 3)')
    return parser.parse_args()

def main():
    args = parse_args()
    target_dir = Path(args.dir)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    max_files = args.max

    if not target_dir.exists():
        print(f"目录不存在: {target_dir}")
        return
    files = sorted(target_dir.glob('*.npz'))[:max_files]
    if not files:
        print("未找到 npz 文件")
        return
    print(f"发现 {len(files)} 个文件，开始可视化...")
    index_html_lines = [
        '<html><head><meta charset="utf-8"><title>NPZ预览</title></head><body>',
        '<h1>NPZ 文件预览</h1>',
        '<ul>'
    ]
    for i,f in enumerate(files, start=1):
        try:
            npz_obj = load_npz(f, allow_pickle=args.allow_pickle)
            keys = set(npz_obj.files)
            bboxes_np = np.asarray(npz_obj['bboxes']) if 'bboxes' in keys else np.zeros((0, 5), dtype=np.float32)

            if 'pseudo_rgb_u8' in keys:
                pseudo = npz_obj['pseudo_rgb_u8']
                out_name = f"{f.stem}_pseudo_rgb.png"
                out_path = out_dir / out_name
                visualize_with_bboxes(pseudo, bboxes_np, f"{f.name} pseudo_rgb", out_path)
                print(f"[{i}] {f.name} -> pseudo_rgb saved {out_path}")
                add_html_entry(index_html_lines, f"{f.name} - pseudo_rgb ({len(bboxes_np)} boxes)", out_name)

            if 'gray_norm' in keys:
                gray = npz_obj['gray_norm']
                out_name = f"{f.stem}_gray_norm.png"
                out_path = out_dir / out_name
                visualize_array(gray, f"{f.name} gray_norm", out_path)
                print(f"[{i}] {f.name} -> gray_norm saved {out_path}")
                add_html_entry(index_html_lines, f"{f.name} - gray_norm", out_name)

            if 'corner_mask_clean' in keys:
                corner = npz_obj['corner_mask_clean']
                out_name = f"{f.stem}_corner_mask.png"
                out_path = out_dir / out_name
                visualize_array(corner, f"{f.name} corner_mask_clean", out_path)
                print(f"[{i}] {f.name} -> corner_mask_clean saved {out_path}")
                add_html_entry(index_html_lines, f"{f.name} - corner_mask_clean", out_name)

            if args.show_features and 'feature_stack' in keys:
                feature_stack = npz_obj['feature_stack']
                feature_names = npz_obj['feature_names'] if 'feature_names' in keys else np.array([])
                visualize_feature_planes(feature_stack, feature_names, args.feature_channels, f.stem, out_dir, index_html_lines)

            # 新数据结构适配：如果有 spec_512 + boxes + labels
            if 'spec_512' in keys and 'boxes' in keys:
                spec = npz_obj['spec_512']
                boxes = npz_obj['boxes']
                # boxes 可能是 (N,4) [x_min,y_min,x_max,y_max] 或含 label -> 尝试检测
                labels = npz_obj['labels'] if 'labels' in keys else None
                # 标准化 boxes 为 (N,5)
                boxes_arr = np.array(boxes)
                if boxes_arr.ndim == 2 and boxes_arr.shape[1] >= 4:
                    if boxes_arr.shape[1] == 4:
                        # 无 label, 用 0 占位
                        bboxes_std = np.concatenate([boxes_arr, np.zeros((boxes_arr.shape[0],1), dtype=boxes_arr.dtype)], axis=1)
                    else:
                        bboxes_std = boxes_arr[:, :5]
                else:
                    bboxes_std = np.zeros((0,5), dtype=np.int32)
                # 如果 labels 存在且长度匹配，用 labels 填充最后一列
                if labels is not None:
                    try:
                        labels_arr = np.array(labels).reshape(-1)
                        if labels_arr.shape[0] == bboxes_std.shape[0]:
                            bboxes_std[:,-1] = labels_arr.astype(bboxes_std.dtype)
                    except Exception:
                        pass
                out_name = f"{f.stem}_spec_boxes.png"
                out_path = out_dir / out_name
                visualize_with_bboxes(spec, bboxes_std, f"{f.name} spec_512+boxes", out_path)
                print(f"[{i}] {f.name} -> spec_512+boxes saved {out_path}")
                add_html_entry(index_html_lines, f"{f.name} - spec_512+boxes ({bboxes_std.shape[0]} boxes)", out_name)
            else:
                # 旧逻辑：有 image + bboxes
                has_image = 'image' in keys
                has_bboxes = 'bboxes' in keys and npz_obj['bboxes'].ndim == 2 and npz_obj['bboxes'].shape[1] == 5
                if has_image and has_bboxes and args.prefer_image:
                    image = npz_obj['image']
                    out_name = f"{f.stem}_image_bbox.png"
                    out_path = out_dir / out_name
                    visualize_with_bboxes(image, bboxes_np, f"{f.name} image+bbox", out_path)
                    print(f"[{i}] {f.name} -> image+bboxes saved {out_path}")
                    add_html_entry(index_html_lines, f"{f.name} - image+bboxes ({len(bboxes_np)} boxes)", out_name)
                else:
                    main_key = guess_main_array(npz_obj)
                    arr = npz_obj[main_key]
                    out_name = f"{f.stem}_{main_key}.png"
                    out_path = out_dir / out_name
                    visualize_array(arr, f"{f.name}::{main_key} shape={arr.shape}", out_path)
                    print(f"[{i}] {f.name} -> {main_key} {arr.shape} saved {out_path}")
                    add_html_entry(index_html_lines, f"{f.name} - key: {main_key} shape: {arr.shape}", out_name)
            npz_obj.close()
        except Exception as e:
            print(f"处理 {f} 出错: {e}")
    index_html_lines.append('</ul></body></html>')
    with open(out_dir / 'index.html','w',encoding='utf-8') as fw:
        fw.write('\n'.join(index_html_lines))
    print(f"完成。预览文件: {out_dir / 'index.html'}")

if __name__ == '__main__':
    main()
