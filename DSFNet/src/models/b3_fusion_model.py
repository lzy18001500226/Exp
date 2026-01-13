import torch
import torch.nn as nn
from .encoders.swin_encoder import SwinEncoder
from .encoders.convnext_encoder import ConvNeXtEncoder


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, attn_dropout: float = 0.1, mlp_ratio: float = 4.0, dropout: float = 0.1, init_scale: float = 1e-3):
        super().__init__()
        # Pre-LN
        self.ln_q = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln_kv = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        # MHA
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        # MLP
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        # 可学习缩放系数（初值很小，避免梯度断流同时不扰动预训练特征）
        self.s_attn = nn.Parameter(torch.tensor(float(init_scale)))
        self.s_mlp = nn.Parameter(torch.tensor(float(init_scale)))
        # 小权重初始化（而非全0）
        self._init_small_weights(std=float(init_scale))

    def _init_small_weights(self, std: float = 1e-3):
        if hasattr(self.attn, 'out_proj') and isinstance(self.attn.out_proj, nn.Linear):
            nn.init.normal_(self.attn.out_proj.weight, mean=0.0, std=std)
            if self.attn.out_proj.bias is not None:
                nn.init.zeros_(self.attn.out_proj.bias)
        # 初始化 MLP 最后一层
        for m in reversed(list(self.mlp.modules())):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                break

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor):
        # q_tokens, kv_tokens: [B, N, C]
        q = self.ln_q(q_tokens)
        kv = self.ln_kv(kv_tokens)
        # 注意力用 float32 保持数值稳定
        with torch.amp.autocast(device_type='cuda', enabled=False):
            attn_out, _ = self.attn(q.float(), kv.float(), kv.float(), need_weights=False)
        x = q_tokens + self.s_attn * self.attn_drop(attn_out.to(q_tokens.dtype))
        x = x + self.s_mlp * self.mlp(self.ln_mlp(x))
        return x

    def set_scale(self, scale: float):
        self.s_attn.data.fill_(float(scale))
        self.s_mlp.data.fill_(float(scale))


class B3FusionModel(nn.Module):
    """
    B3：跨模态注意力融合（FCS queries 读 VTS keys/values）。
    - 使用可学习缩放系数抑制早期跨注意力扰动，支持训练期调度。
    - 注意力在 float32 计算。
    """
    def __init__(
        self,
        num_classes: int,
        backbone_type: str = 'swin',
        embed_dim: int = 512,
        dropout: float = 0.1,
        fusion_depth: int = 1,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        swin_name: str = 'swin_base_patch4_window7_224',
        pretrained: bool = True,
        swin_checkpoint: str | None = None,
        init_fusion_scale: float = 1e-3,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.embed_dim = embed_dim

        # Encoders for FCS & VTS
        if backbone_type == 'convnext':
            self.encoder_fcs = ConvNeXtEncoder(out_dim=embed_dim, pretrained=pretrained, model_name=swin_name, checkpoint_path=swin_checkpoint, img_size=512)
            self.encoder_vts = ConvNeXtEncoder(out_dim=embed_dim, pretrained=pretrained, model_name=swin_name, checkpoint_path=swin_checkpoint, img_size=512)
        else:
            self.encoder_fcs = SwinEncoder(out_dim=embed_dim, pretrained=pretrained, model_name=swin_name, checkpoint_path=swin_checkpoint, img_size=512)
            self.encoder_vts = SwinEncoder(out_dim=embed_dim, pretrained=pretrained, model_name=swin_name, checkpoint_path=swin_checkpoint, img_size=512)

        # Cross-Attention stack（带缩放系数）
        blocks = []
        for _ in range(max(1, int(fusion_depth))):
            blocks.append(CrossAttentionBlock(embed_dim, num_heads=num_heads, attn_dropout=attn_dropout, mlp_ratio=mlp_ratio, dropout=dropout, init_scale=init_fusion_scale))
        self.xattn = nn.ModuleList(blocks)

        # presence 占位，不参与 head
        self.presence_embed = nn.Embedding(2, embed_dim)

        # Head：仅基于融合后的特征（embed_dim）
        in_dim = embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim, eps=1e-6),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    @staticmethod
    def _to_tokens(x: torch.Tensor) -> torch.Tensor:
        # 统一为 [B, N, C]
        if x.dim() == 4:  # [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif x.dim() == 3:
            pass
        else:  # [B, C] -> [B, 1, C]
            x = x.unsqueeze(1)
        return x

    @staticmethod
    def _avg_tokens(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1) if x.dim() == 3 else x

    # ============== 融合/调度辅助接口 ==============
    def set_fusion_scale(self, scale: float):
        for blk in self.xattn:
            if hasattr(blk, 'set_scale'):
                blk.set_scale(scale)

    def get_fusion_scale(self) -> float:
        if len(self.xattn) == 0:
            return 0.0
        s = 0.0
        for blk in self.xattn:
            s += float(blk.s_attn.item())
        return s / len(self.xattn)

    def freeze_vts_encoder(self):
        for p in self.encoder_vts.parameters():
            p.requires_grad = False

    def unfreeze_vts_encoder(self):
        for p in self.encoder_vts.parameters():
            p.requires_grad = True

    # ============== 前向 (临时跳过VTS/xattn以排查NaN) ==============
    def forward(self, fcs: torch.Tensor, vts: torch.Tensor | None = None, vts_valid: torch.Tensor | None = None):
        # 仅使用 FCS 主干，强制跳过跨注意力/VTS 分支
        f_tokens = self._to_tokens(self.encoder_fcs(fcs))  # [B, Nf, C]
        fused_tokens = f_tokens
        fused_feat = self._avg_tokens(fused_tokens)  # [B, C]
        logits = self.head(fused_feat)
        return logits
