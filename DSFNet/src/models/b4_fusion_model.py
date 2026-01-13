import torch
import torch.nn as nn
from .encoders.swin_encoder import SwinEncoder
from .encoders.convnext_encoder import ConvNeXtEncoder


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, attn_dropout: float = 0.1, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor):
        # q_tokens, kv_tokens: [B, N, C]
        q = self.ln_q(q_tokens)
        kv = self.ln_kv(kv_tokens)
        x, _ = self.attn(q, kv, kv, need_weights=False)
        x = q_tokens + self.drop(x)
        x = x + self.mlp(x)
        return x


class B4FusionModel(nn.Module):
    """
    B4：Hybrid 融合（Cross-Attn + Concat + Gating）并保留 presence embedding。
    - 当 batch 内 VTS 全缺失时，退化为 FCS-only；
    - 否则并行得到 cross-attn 特征与 concat 特征，通过门控融合；
    - 分类头输入 [fused_final || presence_emb]。
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
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Encoders
        if backbone_type == 'convnext':
            self.encoder_fcs = ConvNeXtEncoder(out_dim=embed_dim, pretrained=pretrained, model_name=swin_name, checkpoint_path=swin_checkpoint, img_size=512)
            self.encoder_vts = ConvNeXtEncoder(out_dim=embed_dim, pretrained=pretrained, model_name=swin_name, checkpoint_path=swin_checkpoint, img_size=512)
        else:
            self.encoder_fcs = SwinEncoder(out_dim=embed_dim, pretrained=pretrained, model_name=swin_name, checkpoint_path=swin_checkpoint, img_size=512)
            self.encoder_vts = SwinEncoder(out_dim=embed_dim, pretrained=pretrained, model_name=swin_name, checkpoint_path=swin_checkpoint, img_size=512)

        # Cross-Attention stack
        self.xattn = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads=num_heads, attn_dropout=attn_dropout, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(max(1, int(fusion_depth)))
        ])

        # Concat -> proj to embed_dim
        self.concat_proj = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # presence embedding（0 缺失 / 1 存在）
        self.presence_embed = nn.Embedding(2, embed_dim)

        # Gating: 输入 [f_feat || v_feat || presence_emb] -> scalar gate in (0,1)
        self.gating = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

        # Classifier on [fused_final || presence]
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, num_classes),
        )

    @staticmethod
    def _to_tokens(x: torch.Tensor) -> torch.Tensor:
        # 将 encoder 输出统一为 [B, N, C]
        if x.dim() == 4:
            # [B, C, H, W] -> [B, H*W, C]
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, x.size(-1))
        elif x.dim() == 3:
            pass
        else:
            x = x.unsqueeze(1)
        return x

    @staticmethod
    def _avg_tokens(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1) if x.dim() == 3 else x

    def _cross_attend(self, f_tokens: torch.Tensor, v_tokens: torch.Tensor) -> torch.Tensor:
        x = f_tokens
        for blk in self.xattn:
            x = blk(x, v_tokens)
        return x

    def extract_feats(self, fcs: torch.Tensor, vts: torch.Tensor | None, vts_valid: torch.Tensor | None):
        """返回中间特征，供训练脚本做辅助损失：
        - f_feat, v_feat: 池化后的单模态向量 [B,C]
        - fused_xattn: Cross-Attn 后池化向量 [B,C]
        - concat_feat: Concat+proj 后向量 [B,C]
        - fused_final: 门控融合后向量 [B,C]
        - presence: [B] 0/1
        """
        f_tokens = self._to_tokens(self.encoder_fcs(fcs))
        f_feat = self._avg_tokens(f_tokens)

        if vts is None or vts_valid is None:
            presence = torch.zeros((f_feat.size(0),), dtype=torch.long, device=f_feat.device)
            v_feat = torch.zeros_like(f_feat)
            fused_xattn = f_feat
            concat_feat = f_feat
            fused_final = f_feat
        else:
            presence = (vts_valid > 0).long().to(f_feat.device)
            if torch.sum(presence).item() == 0:
                v_feat = torch.zeros_like(f_feat)
                fused_xattn = f_feat
                concat_feat = f_feat
                fused_final = f_feat
            else:
                v_tokens = self._to_tokens(self.encoder_vts(vts))
                v_feat = self._avg_tokens(v_tokens)
                # Cross-Attn 分支
                x_tokens = self._cross_attend(f_tokens, v_tokens)
                fused_xattn = self._avg_tokens(x_tokens)
                # Concat 分支
                concat_feat = self.concat_proj(torch.cat([f_feat, v_feat], dim=-1))
                # Gating 融合
                p_emb = self.presence_embed(presence)
                gate = self.gating(torch.cat([f_feat, v_feat, p_emb], dim=-1)).squeeze(-1)  # [B]
                gate = gate.unsqueeze(-1)  # [B,1]
                fused_final = gate * fused_xattn + (1.0 - gate) * concat_feat

        return {
            'f_feat': f_feat,
            'v_feat': v_feat,
            'fused_xattn': fused_xattn,
            'concat_feat': concat_feat,
            'fused_final': fused_final,
            'presence': presence,
        }

    def forward(self, fcs: torch.Tensor, vts: torch.Tensor | None = None, vts_valid: torch.Tensor | None = None):
        feats = self.extract_feats(fcs, vts, vts_valid)
        fused_final = feats['fused_final']
        presence = feats['presence']
        p_emb = self.presence_embed(presence)
        logits = self.head(torch.cat([fused_final, p_emb], dim=-1))
        return logits
