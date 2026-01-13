import argparse
import os
from pathlib import Path
import torch
import timm

"""
使用 timm 离线导出 ConvNeXt 预训练权重（state_dict）。
示例：
  python download_ConvNeXt_timm_weights.py
或指定其它变体/输出：
  python download_ConvNeXt_timm_weights.py --model convnext_base --out C:/Users/HP/Desktop/Exp/Model/Pretrain/ConvNeXt/timm/convnext_base.pth
注意：ConvNeXt 不接受 img_size 参数，已忽略；需要可联网以便 timm 下载（或使用本地缓存）。
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='convnext_base')
    parser.add_argument('--img_size', type=int, default=224)  # kept for compatibility, ignored
    parser.add_argument('--out', type=str, default='C:/Users/HP/Desktop/Exp/Model/Pretrain/ConvNeXt/timm/convnext_base.pth')
    args = parser.parse_args()

    print(f"Creating timm model {args.model} with pretrained=True … (img_size ignored for ConvNeXt)")
    # ConvNeXt 不支持 img_size 形参
    model = timm.create_model(args.model, pretrained=True)
    sd = model.state_dict()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out_path)
    print(f"Saved state_dict to: {out_path}")

if __name__ == '__main__':
    main()
