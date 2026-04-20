from __future__ import annotations

from functools import lru_cache, reduce
from operator import mul
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import build_activation_layer, build_norm_layer
from .vit_mae import DropPath


def window_partition(x: torch.Tensor, window_size: Sequence[int]) -> torch.Tensor:
    batch_size, depth, height, width, channels = x.shape
    x = x.view(
        batch_size,
        depth // window_size[0],
        window_size[0],
        height // window_size[1],
        window_size[1],
        width // window_size[2],
        window_size[2],
        channels,
    )
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), channels)


def window_reverse(
    windows: torch.Tensor,
    window_size: Sequence[int],
    batch_size: int,
    depth: int,
    height: int,
    width: int,
) -> torch.Tensor:
    x = windows.view(
        batch_size,
        depth // window_size[0],
        height // window_size[1],
        width // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(batch_size, depth, height, width, -1)


def get_window_size(
    x_size: Sequence[int],
    window_size: Sequence[int],
    shift_size: Sequence[int] | None = None,
) -> tuple[int, ...] | tuple[tuple[int, ...], tuple[int, ...]]:
    use_window_size = list(window_size)
    use_shift_size = list(shift_size) if shift_size is not None else None
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if use_shift_size is not None:
                use_shift_size[i] = 0
    if use_shift_size is None:
        return tuple(use_window_size)
    return tuple(use_window_size), tuple(use_shift_size)


@lru_cache()
def compute_mask(
    depth: int,
    height: int,
    width: int,
    window_size: Sequence[int],
    shift_size: Sequence[int],
    device: str | torch.device,
) -> torch.Tensor:
    img_mask = torch.zeros((1, depth, height, width, 1), device=device)
    cnt = 0
    for d in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
        for h in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
            for w in (slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class WindowAttention3D(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        window_size: Sequence[int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = tuple(window_size)
        self.num_heads = int(num_heads)
        head_dim = embed_dims // self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        table_size = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, self.num_heads))
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_windows, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, num_tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        rel_bias = self.relative_position_bias_table[self.relative_position_index[:num_tokens, :num_tokens].reshape(-1)]
        rel_bias = rel_bias.reshape(num_tokens, num_tokens, -1).permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(batch_windows // num_windows, num_windows, self.num_heads, num_tokens, num_tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, num_tokens, num_tokens)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_windows, num_tokens, channels)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation_layer({"type": "GELU"})
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class SwinTransformerBlock3D(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: Sequence[int] = (8, 7, 7),
        shift_size: Sequence[int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = tuple(window_size)
        self.shift_size = tuple(shift_size)
        self.norm1 = build_norm_layer({"type": "LN"}, embed_dims)[1]
        self.attn = WindowAttention3D(embed_dims, window_size, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = build_norm_layer({"type": "LN"}, embed_dims)[1]
        self.mlp = Mlp(embed_dims, int(embed_dims * mlp_ratio), drop=drop)

    def forward_part1(self, x: torch.Tensor, mask_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, depth, height, width, channels = x.shape
        window_size, shift_size = get_window_size((depth, height, width), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_d1 = (window_size[0] - depth % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - height % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - width % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (channels,)))
        shifted_x = window_reverse(attn_windows, window_size, batch_size, dp, hp, wp)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=shift_size, dims=(1, 2, 3))
        else:
            x = shifted_x
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :depth, :height, :width, :].contiguous()
        return x

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = shortcut + self.drop_path(self.forward_part1(x, mask_matrix))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, embed_dims: int) -> None:
        super().__init__()
        self.out_embed_dims = 2 * embed_dims
        self.reduction = nn.Linear(4 * embed_dims, self.out_embed_dims, bias=False)
        self.norm = build_norm_layer({"type": "LN"}, 4 * embed_dims)[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, depth, height, width, channels = x.shape
        if (height % 2 == 1) or (width % 2 == 1):
            x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int] = (8, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_paths: float | Sequence[float] = 0.0,
        downsample: type[PatchMerging] | None = None,
    ) -> None:
        super().__init__()
        self.window_size = tuple(window_size)
        self.shift_size = tuple(i // 2 for i in window_size)
        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=float(drop_paths[i]),
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample(embed_dims=embed_dims) if downsample is not None else None
        self.out_embed_dims = self.downsample.out_embed_dims if self.downsample is not None else embed_dims

    def forward(self, x: torch.Tensor, do_downsample: bool = True) -> torch.Tensor:
        batch_size, channels, depth, height, width = x.shape
        window_size, shift_size = get_window_size((depth, height, width), self.window_size, self.shift_size)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        dp = int(np.ceil(depth / window_size[0])) * window_size[0]
        hp = int(np.ceil(height / window_size[1])) * window_size[1]
        wp = int(np.ceil(width / window_size[2])) * window_size[2]
        attn_mask = compute_mask(dp, hp, wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        if self.downsample is not None and do_downsample:
            x = self.downsample(x)
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size: Sequence[int] | int = (2, 4, 4), in_channels: int = 3, embed_dims: int = 96, patch_norm: bool = True) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = tuple(patch_size)
        self.embed_dims = int(embed_dims)
        self.proj = nn.Conv3d(in_channels, embed_dims, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = build_norm_layer({"type": "LN"}, embed_dims)[1] if patch_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, depth, height, width = x.size()
        if width % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - width % self.patch_size[2]))
        if height % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - height % self.patch_size[1]))
        if depth % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - depth % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            dp, hp, wp = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dims, dp, hp, wp)
        return x


class SwinTransformer3D(nn.Module):
    arch_zoo = {
        **dict.fromkeys(["t", "tiny"], {"embed_dims": 96, "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24]}),
        **dict.fromkeys(["s", "small"], {"embed_dims": 96, "depths": [2, 2, 18, 2], "num_heads": [3, 6, 12, 24]}),
        **dict.fromkeys(["b", "base"], {"embed_dims": 128, "depths": [2, 2, 18, 2], "num_heads": [4, 8, 16, 32]}),
        **dict.fromkeys(["l", "large"], {"embed_dims": 192, "depths": [2, 2, 18, 2], "num_heads": [6, 12, 24, 48]}),
    }

    def __init__(
        self,
        arch: str = "base",
        patch_size: Sequence[int] = (2, 4, 4),
        in_channels: int = 3,
        window_size: Sequence[int] = (8, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        patch_norm: bool = True,
        out_indices: Sequence[int] = (3,),
        out_after_downsample: bool = False,
    ) -> None:
        super().__init__()
        arch_settings = self.arch_zoo[str(arch).lower()]
        self.embed_dims = arch_settings["embed_dims"]
        self.depths = arch_settings["depths"]
        self.num_heads = arch_settings["num_heads"]
        self.num_layers = len(self.depths)
        self.out_indices = tuple(out_indices)
        self.out_after_downsample = bool(out_after_downsample)
        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_channels=in_channels, embed_dims=self.embed_dims, patch_norm=patch_norm)
        self.pos_drop = nn.Dropout(p=drop_rate)
        total_depth = sum(self.depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        self.layers = nn.ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth, num_heads) in enumerate(zip(self.depths, self.num_heads)):
            downsample = PatchMerging if i < self.num_layers - 1 else None
            layer = BasicLayer(
                embed_dims=embed_dims[-1],
                depth=depth,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[:depth],
                downsample=downsample,
            )
            self.layers.append(layer)
            dpr = dpr[depth:]
            embed_dims.append(layer.out_embed_dims)
        self.num_features = embed_dims[1:] if self.out_after_downsample else embed_dims[:-1]
        for i in self.out_indices:
            self.add_module(f"norm{i}", build_norm_layer({"type": "LN"}, self.num_features[i])[1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...] | torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        outs: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x = layer(x.contiguous(), do_downsample=self.out_after_downsample)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                out = norm_layer(x)
                out = out.permute(0, 4, 1, 2, 3).contiguous()
                outs.append(out)
            if layer.downsample is not None and not self.out_after_downsample:
                x = layer.downsample(x)
            if i < self.num_layers - 1:
                x = x.permute(0, 4, 1, 2, 3).contiguous()
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)
