from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .vit_mae import DropPath


def conv_3xnxn(inp: int, oup: int, kernel_size: int = 3, stride: int = 3, groups: int = 1) -> nn.Conv3d:
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 0, 0), groups=groups)


def conv_1xnxn(inp: int, oup: int, kernel_size: int = 3, stride: int = 3, groups: int = 1) -> nn.Conv3d:
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups)


def conv_1x1x1(inp: int, oup: int, groups: int = 1) -> nn.Conv3d:
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)


def conv_3x3x3(inp: int, oup: int, groups: int = 1) -> nn.Conv3d:
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)


def conv_5x5x5(inp: int, oup: int, groups: int = 1) -> nn.Conv3d:
    return nn.Conv3d(inp, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)


def bn_3d(dim: int) -> nn.BatchNorm3d:
    return nn.BatchNorm3d(dim)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1x1x1(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = conv_1x1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)
        self.attn = conv_5x5x5(dim, dim, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = bn_3d(dim)
        self.mlp = CMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        batch_size, channels, frames, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(batch_size, channels, frames, height, width)
        return x


class SpecialPatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_3xnxn(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        batch_size, _, frames, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.reshape(batch_size, frames, height, width, -1).permute(0, 4, 1, 2, 3).contiguous()


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_1xnxn(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        batch_size, _, frames, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.reshape(batch_size, frames, height, width, -1).permute(0, 4, 1, 2, 3).contiguous()


class UniFormer(nn.Module):
    def __init__(
        self,
        depth: Sequence[int] = (5, 8, 20, 7),
        img_size: int = 224,
        in_chans: int = 3,
        embed_dim: Sequence[int] = (64, 128, 320, 512),
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed1 = SpecialPatchEmbed(img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([CBlock(dim=embed_dim[0], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i]) for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([CBlock(dim=embed_dim[1], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i + depth[0]]) for i in range(depth[1])])
        self.blocks3 = nn.ModuleList(
            [
                SABlock(
                    dim=embed_dim[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i + depth[0] + depth[1]],
                )
                for i in range(depth[2])
            ]
        )
        self.blocks4 = nn.ModuleList(
            [
                SABlock(
                    dim=embed_dim[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i + depth[0] + depth[1] + depth[2]],
                )
                for i in range(depth[3])
            ]
        )
        self.norm = bn_3d(embed_dim[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        return self.norm(x)
