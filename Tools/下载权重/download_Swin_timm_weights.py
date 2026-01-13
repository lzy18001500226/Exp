import argparse
import os
from pathlib import Path
import torch
import timm

"""
使用 timm 离线导出预训练权重（state_dict），避免联网时机不可控。
示例：
  python download_timm_weights.py --model swin_base_patch4_window7_224 --img_size 224 --out C:/Users/HP/Desktop/Exp/Model/Pretrain/Swin/timm/swin_base_patch4_window7_224_timm.pth
注意：timm 的 create_model(pretrained=True) 需要网络访问；如果当前环境无法联网，将无法直接下载。
可选：先在可联网环境运行本脚本导出 .pth，再拷贝到目标机使用。
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='swin_base_patch4_window7_224')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    print(f"Creating timm model {args.model} (img_size={args.img_size}) with pretrained=True …")
    model = timm.create_model(args.model, pretrained=True, img_size=args.img_size)
    sd = model.state_dict()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out_path)
    print(f"Saved state_dict to: {out_path}")

if __name__ == '__main__':
    main()
