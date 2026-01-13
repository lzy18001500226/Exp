import torch
import torch.nn as nn
import os
import glob

# 占位：从 timm 导入 Swin，适配为特征输出层。后续可对接你提供的本地权重。
class SwinBackbone(nn.Module):
    def __init__(self, name: str = 'swin_base_patch4_window7_224', pretrained: bool = False, checkpoint: str = '', img_size: int = 224):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise RuntimeError(f"timm not installed: {e}")
        # ✅ 增加 P2 输出 (stride=4) 用于小目标检测
        self.model = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(1,2,3), img_size=img_size)
        # 输出 P2/P3/P4 三个阶段,统一通道数
        chs = self.model.feature_info.channels()
        self.proj2 = nn.Conv2d(chs[-3], 128, 1)  # P2: stride=4
        self.proj3 = nn.Conv2d(chs[-2], 256, 1)  # P3: stride=8
        self.proj4 = nn.Conv2d(chs[-1], 512, 1)  # P4: stride=16
        # 尝试加载自定义 checkpoint（文件或目录）
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
                # 重命名key: 官方格式(layers.0) -> timm格式(layers_0)
                # 只替换第一层的点号: layers.N -> layers_N
                state_fixed = {}
                for k, v in state.items():
                    # 只替换 layers.数字 模式,不影响内层的 blocks.N
                    import re
                    new_k = re.sub(r'^layers\.(\d+)', r'layers_\1', k)
                    new_k = re.sub(r'\.downsample\.(\d+)', r'.downsample_\1', new_k)
                    state_fixed[new_k] = v
                before_keys = set(self.model.state_dict().keys())
                msg = self.model.load_state_dict(state_fixed, strict=False)
                missing = list(msg.missing_keys) if hasattr(msg, 'missing_keys') else []
                unexpected = list(msg.unexpected_keys) if hasattr(msg, 'unexpected_keys') else []
                total = len(before_keys)
                loaded = total - len(missing)
                ratio = loaded / max(1, total)
                print(f"[SwinBackbone] Loaded ckpt: {os.path.basename(ck)} | loaded={loaded}/{total} ({ratio:.1%})")
                if missing and len(missing) <= 20:
                    print(f"  missing({len(missing)}): {missing}")
                elif missing:
                    print(f"  missing({len(missing)}): {missing[:10]} ...")
                if unexpected and len(unexpected) <= 20:
                    print(f"  unexpected({len(unexpected)}): {unexpected}")
                elif unexpected:
                    print(f"  unexpected({len(unexpected)}): {unexpected[:10]} ...")
            except Exception as e:
                print(f"[SwinBackbone] failed to load checkpoint {ck}: {e}")

    def forward(self, x):
        feats = self.model(x)  # list: [..., C2, C3, C4], NHWC format for Swin
        # Swin features_only returns NHWC, convert to NCHW for Conv2d
        f2 = feats[-3].permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
        f3 = feats[-2].permute(0, 3, 1, 2)
        f4 = feats[-1].permute(0, 3, 1, 2)
        p2 = self.proj2(f2)
        p3 = self.proj3(f3)
        p4 = self.proj4(f4)
        return p2, p3, p4
