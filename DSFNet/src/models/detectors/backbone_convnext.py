import torch
import torch.nn as nn
import os
import glob

class ConvNeXtBackbone(nn.Module):
    def __init__(self, name: str = 'convnext_base', pretrained: bool = False, checkpoint: str = '', img_size: int = 224):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise RuntimeError(f"timm not installed: {e}")
        # ✅ 增加 P2 输出 (stride=4) 用于小目标检测
        self.model = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(1,2,3))
        chs = self.model.feature_info.channels()
        self.proj2 = nn.Conv2d(chs[-3], 128, 1)  # P2: stride=4
        self.proj3 = nn.Conv2d(chs[-2], 256, 1)  # P3: stride=8
        self.proj4 = nn.Conv2d(chs[-1], 512, 1)  # P4: stride=16
        ck = checkpoint
        if ck and os.path.isdir(ck):
            cand = sorted(glob.glob(os.path.join(ck, '*.pth')) + glob.glob(os.path.join(ck, '*.pt')))
            ck = cand[0] if cand else ''
        if ck and os.path.isfile(ck):
            try:
                state = torch.load(ck, map_location='cpu', weights_only=False)
                if isinstance(state, dict):
                    if 'state_dict' in state:
                        state = state['state_dict']
                    elif 'model' in state and isinstance(state['model'], dict):
                        state = state['model']
                # 重命名key: 官方格式(stem.0, stages.0) -> timm格式(stem_0, stages_0)
                # 注意：timm模型中的 downsample 使用点号索引(如 downsample.0)，不要改为下划线
                state_fixed = {}
                for k, v in state.items():
                    import re
                    new_k = re.sub(r'^stem\.(\d+)', r'stem_\1', k)
                    new_k = re.sub(r'^stages\.(\d+)', r'stages_\1', new_k)
                    state_fixed[new_k] = v
                before_keys = set(self.model.state_dict().keys())
                msg = self.model.load_state_dict(state_fixed, strict=False)
                missing = list(msg.missing_keys) if hasattr(msg, 'missing_keys') else []
                unexpected = list(msg.unexpected_keys) if hasattr(msg, 'unexpected_keys') else []
                total = len(before_keys)
                loaded = total - len(missing)
                ratio = loaded / max(1, total)
                print(f"[ConvNeXtBackbone] Loaded ckpt: {os.path.basename(ck)} | loaded={loaded}/{total} ({ratio:.1%})")
                if missing and len(missing) <= 20:
                    print(f"  missing({len(missing)}): {missing}")
                elif missing:
                    print(f"  missing({len(missing)}): {missing[:10]} ...")
                if unexpected and len(unexpected) <= 20:
                    print(f"  unexpected({len(unexpected)}): {unexpected}")
                elif unexpected:
                    print(f"  unexpected({len(unexpected)}): {unexpected[:10]} ...")
            except Exception as e:
                print(f"[ConvNeXtBackbone] failed to load checkpoint {ck}: {e}")

    def forward(self, x):
        feats = self.model(x)  # features_only may return NHWC or NCHW
        # Handle both formats: if 4D and last dim matches expected channels, it's NHWC
        f2, f3, f4 = feats[-3], feats[-2], feats[-1]
        # Check if NHWC (last dim == C)
        if f2.ndim == 4 and f2.shape[-1] == self.proj2.in_channels:
            f2 = f2.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
        if f3.ndim == 4 and f3.shape[-1] == self.proj3.in_channels:
            f3 = f3.permute(0, 3, 1, 2)
        if f4.ndim == 4 and f4.shape[-1] == self.proj4.in_channels:
            f4 = f4.permute(0, 3, 1, 2)
        p2 = self.proj2(f2)
        p3 = self.proj3(f3)
        p4 = self.proj4(f4)
        return p2, p3, p4
