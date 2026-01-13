import torch
import torch.nn as nn
import timm

class SwinEncoder(nn.Module):
    """
    Swin-Base 编码器：支持 512×512 输入。
    - 先构建完整 Swin，并加载 state_dict；
    - 不再使用 FeatureListNet 包装，直接按主干的 forward_features 前半段执行到最后一个 stage + norm，得到 [B, N, C] token；
    - 单通道在前向阶段使用 1x1 Conv 学习映射为 3 通道；
    - 本地 checkpoint 手动加载（strict=False，过滤分类头）。
    """
    def __init__(self, in_ch: int = 1, out_dim: int = 512, pretrained: bool = False, model_name: str = 'swin_base_patch4_window7_224', checkpoint_path: str = None, img_size: int = 512):
        super().__init__()
        self.in_ch = in_ch

        # 1) 构建完整模型
        use_pretrained = pretrained and not checkpoint_path
        full = timm.create_model(model_name, pretrained=use_pretrained, img_size=img_size, num_classes=0, global_pool='')

        # 2) 加载 checkpoint（同原逻辑）
        if checkpoint_path:
            def _clean_key(k: str) -> str | None:
                if k.startswith('module.'):
                    k = k[len('module.') :]
                if k.startswith('model.'):
                    k = k[len('model.') :]
                if k.startswith('head') or k.startswith('fc') or 'head.' in k:
                    return None
                if 'relative_position_index' in k or 'relative_coords_table' in k or 'attn_mask' in k:
                    return None
                return k
            try:
                try:
                    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                except TypeError:
                    ckpt = torch.load(checkpoint_path, map_location='cpu')
                sd_raw = ckpt.get('state_dict', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
                new_sd = {}
                for k, v in sd_raw.items():
                    k2 = _clean_key(k)
                    if k2 is None:
                        continue
                    new_sd[k2] = v
                full.load_state_dict(new_sd, strict=False)
            except Exception as e:
                print(f"[SwinEncoder] failed to load checkpoint '{checkpoint_path}': {e}")

        self.backbone = full
        # 记录主干输出通道
        num_features = getattr(full, 'num_features', None)
        if num_features is None:
            num_features = 1024
        self.in_channels = int(num_features)

        # 输入适配：1->3
        self.input_adapter = None
        if self.in_ch == 1:
            self.input_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            with torch.no_grad():
                self.input_adapter.weight.fill_(1.0/3.0)

        self.out_proj = nn.Linear(self.in_channels, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def _forward_backbone_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # 直接使用 timm 的 forward_features，按我们设置不会做全局池化，返回 [B,N,C]
        feats = self.backbone.forward_features(x)
        # 某些版本可能返回 (x, H, W) 或 list，取第一个
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 单通道适配
        if x.shape[1] == 1 and self.input_adapter is not None:
            x = self.input_adapter(x)
        elif x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        tokens = self._forward_backbone_tokens(x)
        # 统一到 [B, N, C]
        if tokens.dim() == 4:
            B, d1, d2, d3 = tokens.shape
            # 优先判断最后一维是否为通道
            if d3 == self.in_channels:
                # BHWC -> [B, H*W, C]
                tokens = tokens.reshape(B, d1 * d2, d3)
            elif d1 == self.in_channels:
                # BCHW -> [B, H*W, C]
                tokens = tokens.permute(0, 2, 3, 1).reshape(B, d2 * d3, d1)
            else:
                # 回退：将最后一维视为通道
                tokens = tokens.reshape(B, d1 * d2, d3)
        elif tokens.dim() == 3 and tokens.shape[-1] != self.in_channels and tokens.shape[1] == self.in_channels:
            tokens = tokens.transpose(1, 2).contiguous()
        # 投影
        tokens = self.out_proj(tokens)
        tokens = self.out_norm(tokens)
        return tokens
