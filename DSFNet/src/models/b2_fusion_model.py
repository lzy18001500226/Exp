import torch
import torch.nn as nn
from .encoders.swin_encoder import SwinEncoder
from .encoders.convnext_encoder import ConvNeXtEncoder

class BaseFusionModel(nn.Module):
    """
    最小可跑骨架（B0/B1：单分支分类）。
    - 输入：fcs 或 vts [B,1,512,512]
    - 输出：logits [B,num_classes]
    通过 backbone_type 选择 'swin'（B0）或 'convnext'（B1）。
    """
    def __init__(
        self,
        num_classes: int,
        backbone_type: str = 'swin',
        mode: str = 'single',
        embed_dim: int = 512,
        dropout: float = 0.1,
        swin_name: str = 'swin_base_patch4_window7_224',
        pretrained: bool = True,
        swin_checkpoint: str = None,
    ):
        super().__init__()
        self.mode = mode
        self.backbone_type = backbone_type
        
        # 单流主干（FCS 用于 B0，或 VTS 用于 B1；在 B2 中作为 FCS 编码器）
        if backbone_type == 'convnext':
            self.encoder_fcs = ConvNeXtEncoder(
                out_dim=embed_dim,
                pretrained=pretrained,
                model_name=swin_name,
                checkpoint_path=swin_checkpoint,
                img_size=512
            )
        else:
            self.encoder_fcs = SwinEncoder(
                out_dim=embed_dim,
                pretrained=pretrained,
                model_name=swin_name,
                checkpoint_path=swin_checkpoint,
                img_size=512
            )

        if mode == 'concat':
            # VTS 分支默认与 FCS 相同类型
            if backbone_type == 'convnext':
                self.encoder_vts = ConvNeXtEncoder(
                    out_dim=embed_dim,
                    pretrained=pretrained,
                    model_name=swin_name,
                    checkpoint_path=swin_checkpoint,
                    img_size=512
                )
            else:
                self.encoder_vts = SwinEncoder(
                    out_dim=embed_dim,
                    pretrained=pretrained,
                    model_name=swin_name,
                    checkpoint_path=swin_checkpoint,
                    img_size=512
                )
            # presence embedding（vts 缺失时提供一个可学习标志）
            self.presence_embed = nn.Embedding(2, embed_dim)
            in_dim = embed_dim * 3  # fcs + vts + presence
        else:
            self.encoder_vts = None
            self.presence_embed = None
            in_dim = embed_dim

        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def _avg_tokens(self, x):
        """将 token 序列平均池化为单个向量"""
        if x.dim() == 3:  # [B, N, C]
            return x.mean(dim=1)
        if x.dim() == 4:  # [B, C, H, W]
            return x.mean(dim=(2, 3))
        return x

    def forward(self, fcs, vts=None, vts_valid=None):
        # B0/B1 单流
        if self.mode != 'concat':
            tokens = self.encoder_fcs(fcs)
            feat = self._avg_tokens(tokens)
            return self.head(feat)

        # B2 双流
        f_tokens = self.encoder_fcs(fcs)
        f_feat = self._avg_tokens(f_tokens)

        # 跳过 VTS 前向：当 batch 的 vts_valid 全 0 时
        if vts is None or vts_valid is None:
            v_feat = torch.zeros_like(f_feat)
            presence = torch.zeros((f_feat.size(0),), dtype=torch.long, device=f_feat.device)
        else:
            presence = (vts_valid > 0).long().to(f_feat.device)
            if torch.sum(presence).item() == 0:
                v_feat = torch.zeros_like(f_feat)
            else:
                v_tokens = self.encoder_vts(vts)
                v_feat = self._avg_tokens(v_tokens)

        p_emb = self.presence_embed(presence)
        feat = torch.cat([f_feat, v_feat, p_emb], dim=-1)
        return self.head(feat)
