"""
Position Feature (PF) Aware Module - Inspired by SMNet
用于提取时频域的全局位置特征,专为 FCS 频率跳变特性设计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention2D(nn.Module):
    """2D Self-Attention for Time-Frequency Position Features
    
    Args:
        in_channels: 输入特征通道数
        reduction: Query/Key/Value 维度缩减比例
        num_heads: 多头注意力头数
    """
    def __init__(self, in_channels: int, reduction: int = 8, num_heads: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.dim_per_head = max(1, in_channels // (reduction * num_heads))
        self.hidden_dim = self.dim_per_head * num_heads
        
        # Query, Key, Value 投影
        self.query = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, bias=False)
        self.key = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, bias=False)
        self.value = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, bias=False)
        
        # 输出投影
        self.out_proj = nn.Conv2d(self.hidden_dim, in_channels, kernel_size=1, bias=False)
        self.scale = math.sqrt(self.dim_per_head)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attn_out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 投影到 Q, K, V
        q = self.query(x)  # [B, hidden_dim, H, W]
        k = self.key(x)    # [B, hidden_dim, H, W]
        v = self.value(x)  # [B, hidden_dim, H, W]
        
        # Reshape for multi-head attention
        # [B, num_heads, dim_per_head, H*W]
        q = q.view(B, self.num_heads, self.dim_per_head, H * W)
        k = k.view(B, self.num_heads, self.dim_per_head, H * W)
        v = v.view(B, self.num_heads, self.dim_per_head, H * W)
        
        # Scaled Dot-Product Attention
        # [B, num_heads, H*W, H*W]
        attn = torch.matmul(q.transpose(-2, -1), k) / self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        # [B, num_heads, dim_per_head, H*W]
        out = torch.matmul(v, attn.transpose(-2, -1))
        
        # Concatenate heads and reshape
        # [B, hidden_dim, H, W]
        out = out.contiguous().view(B, self.hidden_dim, H, W)
        
        # Output projection
        out = self.out_proj(out)  # [B, C, H, W]
        return out


class PFAwareModule(nn.Module):
    """Position Feature (PF) Aware Module
    
    SMNet 论文中的核心创新:同时从时间域和频率域提取全局位置相关性
    
    Args:
        in_channels: 输入特征通道数
        reduction: Self-Attention 维度缩减比例
        num_heads: 多头注意力头数
    """
    def __init__(self, in_channels: int, reduction: int = 8, num_heads: int = 1):
        super().__init__()
        self.in_channels = in_channels
        
        # 时间域 Self-Attention (沿频率轴聚合)
        self.attn_time = SelfAttention2D(in_channels, reduction, num_heads)
        
        # 频率域 Self-Attention (沿时间轴聚合)
        self.attn_freq = SelfAttention2D(in_channels, reduction, num_heads)
        
        # 融合权重 (可学习)
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - 特征图 (H=时间,W=频率)
        Returns:
            pf_out: [B, C, H, W] - 位置感知增强后的特征
        """
        # 时间域注意力 (捕捉频率跳变的时间模式)
        x_time = self.attn_time(x)
        
        # 频率域注意力 (捕捉频率分布的空间模式)
        # 转置 H <-> W,应用注意力,再转置回来
        x_freq = x.transpose(-2, -1)  # [B, C, W, H]
        x_freq = self.attn_freq(x_freq)
        x_freq = x_freq.transpose(-2, -1)  # [B, C, H, W]
        
        # 融合时间和频率域的位置特征
        pf_out = self.fusion(torch.cat([x_time, x_freq], dim=1))
        
        # 残差连接
        return x + pf_out


class LightweightPFAware(nn.Module):
    """轻量级 PF-Aware 模块 (降低计算复杂度)
    
    使用下采样 + Self-Attention + 上采样减少计算量
    """
    def __init__(self, in_channels: int, downsample_ratio: int = 4, reduction: int = 8):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        
        # 下采样
        self.down = nn.Conv2d(in_channels, in_channels, 
                             kernel_size=downsample_ratio, 
                             stride=downsample_ratio, 
                             groups=in_channels)
        
        # PF-Aware 在低分辨率特征上执行
        self.pfaware = PFAwareModule(in_channels, reduction=reduction, num_heads=1)
        
        # 上采样
        self.up = nn.ConvTranspose2d(in_channels, in_channels,
                                    kernel_size=downsample_ratio,
                                    stride=downsample_ratio,
                                    groups=in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        # 下采样到低分辨率
        x_down = self.down(x)  # [B, C, H//r, W//r]
        
        # PF-Aware 处理
        pf_down = self.pfaware(x_down)
        
        # 上采样回原分辨率
        pf_up = self.up(pf_down)  # [B, C, H, W]
        
        # 确保尺寸匹配 (处理不能整除的情况)
        if pf_up.shape[-2:] != x.shape[-2:]:
            pf_up = F.interpolate(pf_up, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return x + pf_up


def test_pfaware_module():
    """测试 PF-Aware 模块"""
    print("Testing PFAware Module...")
    
    # 标准模块
    module = PFAwareModule(in_channels=256, reduction=8, num_heads=4)
    x = torch.randn(2, 256, 32, 32)
    out = module(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    
    # 轻量级模块
    lite_module = LightweightPFAware(in_channels=256, downsample_ratio=4, reduction=8)
    out_lite = lite_module(x)
    print(f"Lightweight - Input: {x.shape} -> Output: {out_lite.shape}")
    assert out_lite.shape == x.shape, "Shape mismatch!"
    
    # 计算参数量
    params_std = sum(p.numel() for p in module.parameters())
    params_lite = sum(p.numel() for p in lite_module.parameters())
    print(f"Standard PFAware params: {params_std:,}")
    print(f"Lightweight PFAware params: {params_lite:,}")
    print(f"Reduction ratio: {params_std / params_lite:.2f}x")
    
    print("✅ All tests passed!")


if __name__ == '__main__':
    test_pfaware_module()
