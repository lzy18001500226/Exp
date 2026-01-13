import torch
import torch.nn as nn
import timm

class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt 编码器：支持 512×512 输入，单通道适配，输出 token 序列并投影到 out_dim。
    - 构建完整 ConvNeXt 模型（num_classes=0, global_pool=''），使用 forward_features 得到 [B,C,H,W]
    - 1x1 Conv 将 1 通道映射为 3 通道，兼容预训练权重
    - 可选本地 checkpoint 加载（过滤分类头），strict=False
    - 输出：tokens [B, N, out_dim] （其中 N = H*W）
    """
    def __init__(
        self,
        in_ch: int = 1,
        out_dim: int = 512,
        pretrained: bool = False,
        model_name: str = 'convnext_base',
        checkpoint_path: str | None = None,
        img_size: int = 512,
    ):
        super().__init__()
        self.in_ch = int(in_ch)

        # 构建完整模型（避免 features_only 键名不一致带来的加载问题）
        use_pretrained = bool(pretrained) and not checkpoint_path
        self.backbone = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=0,
            global_pool='',
            img_size=img_size,
        )
        
        # 单通道输入适配
        self.input_adapter = None
        if self.in_ch == 1:
            self.input_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            with torch.no_grad():
                self.input_adapter.weight.fill_(1.0/3.0)

        # 可选：从本地 checkpoint 加载（在探测通道数之前加载）
        if checkpoint_path:
            def _clean_key(k: str) -> str | None:
                if k.startswith('module.'):
                    k = k[len('module.') :]
                if k.startswith('model.'):
                    k = k[len('model.') :]
                if k.startswith('head') or k.startswith('fc') or 'head.' in k:
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
                missing, unexpected = self.backbone.load_state_dict(new_sd, strict=False)
                max_show = 8
                if len(missing) or len(unexpected):
                    miss_show = ', '.join(list(missing)[:max_show])
                    unexp_show = ', '.join(list(unexpected)[:max_show])
                    print(f"[ConvNeXtEncoder] load summary: missing={len(missing)} [{miss_show}] | unexpected={len(unexpected)} [{unexp_show}]")
                else:
                    print("[ConvNeXtEncoder] load summary: missing=0 unexpected=0")
            except Exception as e:
                print(f"[ConvNeXtEncoder] failed to load checkpoint '{checkpoint_path}': {e}")
        
        # 推断最后通道数（在加载checkpoint之后）
        num_features = getattr(self.backbone, 'num_features', None)
        if num_features is None:
            # 兼容不同实现，做一次前向探测
            with torch.no_grad():
                x = torch.zeros(1, 3, img_size, img_size)
                f = self.backbone.forward_features(x)  # [1,C,H,W]
                num_features = f.shape[1]
        self.in_channels = int(num_features)
        
        # 输出投影（在获取正确的in_channels之后）
        self.out_proj = nn.Linear(self.in_channels, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def _forward_feats(self, x: torch.Tensor) -> torch.Tensor:
        # forward_features → features, ensure [B,C,H,W]
        x = self.backbone.forward_features(x)
        if x.dim() == 4 and x.shape[1] < x.shape[-1]:
            # Likely NHWC -> NCHW
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入适配
        if x.shape[1] == 1 and self.input_adapter is not None:
            x = self.input_adapter(x)
        elif x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feats = self._forward_feats(x)            # [B,C,H,W]
        B, C, H, W = feats.shape
        tokens = feats.permute(0, 2, 3, 1).reshape(B, H*W, C)  # [B,N,C]
        tokens = self.out_proj(tokens)            # [B,N,E]
        tokens = self.out_norm(tokens)
        return tokens
