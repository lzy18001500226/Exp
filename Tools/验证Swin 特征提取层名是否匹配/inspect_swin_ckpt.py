import argparse
import json
from pathlib import Path
import torch
import timm


def clean_key(k: str) -> str | None:
    if k.startswith('module.'):
        k = k[len('module.'):]
    if k.startswith('model.'):
        k = k[len('model.'):]
    # drop classifier and runtime buffers
    if k.startswith('head') or k.startswith('fc') or 'head.' in k:
        return None
    if 'relative_position_index' in k or 'relative_coords_table' in k or 'attn_mask' in k:
        return None
    return k


def group_prefix(name: str) -> str:
    # coarse grouping for readability
    if name.startswith('patch_embed'):
        return 'patch_embed'
    if name.startswith('layers.'):
        parts = name.split('.')
        if len(parts) >= 2 and parts[1].isdigit():
            return f"layers.{parts[1]}"
        return 'layers'
    if name.startswith('stages.'):
        parts = name.split('.')
        if len(parts) >= 2 and parts[1].isdigit():
            return f"stages.{parts[1]}"
        return 'stages'
    if name.startswith('norm'):
        return 'norm'
    return name.split('.', 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--model_name', default='swin_base_patch4_window7_224')
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--out', default='swin_ckpt_inspect.json')
    args = ap.parse_args()

    model = timm.create_model(args.model_name, pretrained=False, img_size=args.img_size, num_classes=0, global_pool='')
    m_sd = model.state_dict()
    m_keys = set(m_sd.keys())

    try:
        try:
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
    except Exception as e:
        raise SystemExit(f"Failed to load checkpoint: {e}")

    sd_raw = ckpt.get('state_dict', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
    new_sd = {}
    filtered = 0
    for k, v in sd_raw.items():
        k2 = clean_key(k)
        if k2 is None:
            filtered += 1
            continue
        new_sd[k2] = v

    c_keys = set(new_sd.keys())
    matched = sorted(m_keys & c_keys)
    missing = sorted(m_keys - c_keys)
    unexpected = sorted(c_keys - m_keys)

    # group by coarse prefix
    def group_stats(names):
        g = {}
        for n in names:
            p = group_prefix(n)
            g.setdefault(p, 0)
            g[p] += 1
        return g

    report = {
        'model_name': args.model_name,
        'img_size': args.img_size,
        'model_params': len(m_keys),
        'ckpt_params_after_filter': len(c_keys),
        'filtered_keys_count': filtered,
        'coverage_ratio': round(len(matched) / max(1, len(m_keys)), 4),
        'missing_count': len(missing),
        'unexpected_count': len(unexpected),
        'missing_by_group': group_stats(missing),
        'unexpected_by_group': group_stats(unexpected),
        'missing_preview': missing[:30],
        'unexpected_preview': unexpected[:30],
    }

    out = Path(args.out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved inspect report to {out}")


if __name__ == '__main__':
    main()
