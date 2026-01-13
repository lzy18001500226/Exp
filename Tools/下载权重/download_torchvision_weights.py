import argparse
import os
from pathlib import Path
import traceback

import torch
import torchvision
from torchvision import models


def try_build_model(name: str):
    fn = getattr(models, name, None)
    if fn is None:
        return None, None
    # Try multiple ways to get pretrained weights depending on torchvision version
    weights_obj = None
    try:
        get_w = getattr(models, 'get_model_weights', None)
        if get_w is not None:
            W = get_w(name)
            if hasattr(W, 'DEFAULT'):
                weights_obj = W.DEFAULT
    except Exception:
        weights_obj = None

    # Try instantiate with found weights or fallbacks
    m = None
    tried = []
    if weights_obj is not None:
        try:
            m = fn(weights=weights_obj)
            tried.append('weights=DEFAULT')
        except Exception:
            m = None
    if m is None:
        for w in ('DEFAULT', 'IMAGENET1K_V1', 'IMAGENET1K_V2'):
            try:
                m = fn(weights=w)  # type: ignore[arg-type]
                tried.append(f'weights={w}')
                break
            except Exception:
                m = None
    if m is None:
        try:
            m = fn(pretrained=True)  # deprecated in new versions but works in older
            tried.append('pretrained=True')
        except Exception:
            m = None

    return m, (weights_obj.__class__.__name__ if getattr(weights_obj, '__class__', None) else (tried[-1] if tried else ''))


def save_model_weights(model, out_path: Path, meta: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'state_dict': model.state_dict(),
        'meta': meta,
        'torchvision': torchvision.__version__,
        'torch': torch.__version__,
    }
    torch.save(payload, str(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', required=True, help='Output root dir, e.g. C:/.../Model/Pretrain/Backbone/Others')
    args = parser.parse_args()
    root = Path(args.out)

    families = {
        'AlexNet': ['alexnet'],
        'ConvNeXt': ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'],
        'DenseNet': ['densenet121', 'densenet161', 'densenet169', 'densenet201'],
        'EfficientNet': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'],
        'EfficientNetV2': ['efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'],
        'GoogLeNet': ['googlenet'],
        'InceptionV3': ['inception_v3'],
        'MaxVit': ['maxvit_t'],  # 可用性依赖 torchvision 版本
        'MNASNet': ['mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3'],
        'MobileNetV2': ['mobilenet_v2'],
        'MobileNetV3': ['mobilenet_v3_small', 'mobilenet_v3_large'],
        'RegNet': [
            'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf',
            'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf', 'regnet_x_8gf',
        ],
        'ResNet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
        'ResNeXt': ['resnext50_32x4d', 'resnext101_32x8d'],
        'ShuffleNetV2': ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'],
        'SqueezeNet': ['squeezenet1_0', 'squeezenet1_1'],
        'SwinTransformer': ['swin_t', 'swin_s', 'swin_b'],
        'VGG': ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'],
        'VisionTransformer': ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32'],
        'WideResNet': ['wide_resnet50_2', 'wide_resnet101_2'],
    }

    summary_lines = []
    for fam, names in families.items():
        for name in names:
            try:
                m, wtag = try_build_model(name)
                if m is None:
                    summary_lines.append(f"[skip] {fam}/{name}: model or weights not available in this torchvision version")
                    continue
                m.eval()
                meta = {'family': fam, 'model': name, 'weights': wtag}
                out_path = root / fam / f"{name}.pth"
                save_model_weights(m, out_path, meta)
                summary_lines.append(f"[ok]   {fam}/{name} -> {out_path}")
            except Exception:
                summary_lines.append(f"[fail] {fam}/{name}: {traceback.format_exc().splitlines()[-1]}")

    # write summary
    try:
        (root / 'DOWNLOAD_SUMMARY.txt').write_text('\n'.join(summary_lines), encoding='utf-8')
    except Exception:
        pass

    print('\n'.join(summary_lines))


if __name__ == '__main__':
    main()
