from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple

from .common import build_activation_layer, build_norm_layer
from .vit_mae import DropPath


def resize_pos_embed(
    pos_embed: torch.Tensor,
    src_shape: tuple[int, int, int],
    dst_shape: tuple[int, int, int],
    mode: str = "trilinear",
    num_extra_tokens: int = 1,
) -> torch.Tensor:
    if src_shape == dst_shape:
        return pos_embed
    extra_tokens = pos_embed[:, :num_extra_tokens]
    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, *src_shape, pos_embed.shape[-1]).permute(0, 4, 1, 2, 3)
    dst_weight = F.interpolate(src_weight, size=dst_shape, mode=mode, align_corners=False)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    return torch.cat((extra_tokens, dst_weight), dim=1)


def resize_decomposed_rel_pos(rel_pos: torch.Tensor, q_size: int, k_size: int) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos = F.interpolate(rel_pos.transpose(0, 1).unsqueeze(0), size=max_rel_dist, mode="linear")
        rel_pos = rel_pos.squeeze(0).transpose(0, 1)
    q_ratio = max(k_size / q_size, 1.0)
    k_ratio = max(q_size / k_size, 1.0)
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * q_ratio
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * k_ratio
    relative_coords = (q_coords - k_coords) + (k_size - 1) * k_ratio
    return rel_pos[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    q_shape: Sequence[int],
    k_shape: Sequence[int],
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    rel_pos_t: torch.Tensor,
    with_cls_token: bool = True,
) -> torch.Tensor:
    sp_idx = 1 if with_cls_token else 0
    bsz, num_heads, _, channels = q.shape
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    rt = resize_decomposed_rel_pos(rel_pos_t, q_t, k_t)
    rh = resize_decomposed_rel_pos(rel_pos_h, q_h, k_h)
    rw = resize_decomposed_rel_pos(rel_pos_w, q_w, k_w)
    r_q = q[:, :, sp_idx:].reshape(bsz, num_heads, q_t, q_h, q_w, channels)
    rel_t = torch.einsum("bythwc,tkc->bythwk", r_q, rt)
    rel_h = torch.einsum("bythwc,hkc->bythwk", r_q, rh)
    rel_w = torch.einsum("bythwc,wkc->bythwk", r_q, rw)
    rel_pos_embed = (
        rel_t[:, :, :, :, :, :, None, None]
        + rel_h[:, :, :, :, :, None, :, None]
        + rel_w[:, :, :, :, :, None, None, :]
    )
    attn_map = attn[:, :, sp_idx:, sp_idx:].view(bsz, -1, q_t, q_h, q_w, k_t, k_h, k_w)
    attn_map += rel_pos_embed
    attn[:, :, sp_idx:, sp_idx:] = attn_map.view(bsz, -1, q_t * q_h * q_w, k_t * k_h * k_w)
    return attn


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 768,
        kernel_size: int | tuple[int, int, int] = (3, 7, 7),
        stride: int | tuple[int, int, int] = (2, 4, 4),
        padding: int | tuple[int, int, int] = (1, 3, 3),
        input_size: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        self.projection = nn.Conv3d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.init_out_size = None
        if input_size is not None:
            input_size = _triple(input_size)
            t_out = (input_size[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
            h_out = (input_size[1] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
            w_out = (input_size[2] + 2 * padding[2] - kernel_size[2]) // stride[2] + 1
            self.init_out_size = (t_out, h_out, w_out)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3], x.shape[4])
        x = x.flatten(2).transpose(1, 2)
        return x, out_size


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        out_channels: int | None = None,
        act_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = build_activation_layer(act_cfg or {"type": "GELU"})
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def attention_pool(
    x: torch.Tensor,
    pool: nn.Module | None,
    in_size: tuple[int, int, int],
    with_cls_token: bool = True,
    norm: nn.Module | None = None,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    if pool is None:
        return x, in_size
    ndim = x.ndim
    if ndim == 3:
        x = x.unsqueeze(1)
    elif ndim != 4:
        raise RuntimeError(f"Unsupported input shape {tuple(x.shape)}")
    batch_size, num_heads, _, channels = x.shape
    if with_cls_token:
        cls_tok, x = x[:, :, :1, :], x[:, :, 1:, :]
    t, h, w = in_size
    x = x.reshape(batch_size * num_heads, t, h, w, channels).permute(0, 4, 1, 2, 3).contiguous()
    x = pool(x)
    out_size = (x.shape[2], x.shape[3], x.shape[4])
    x = x.reshape(batch_size, num_heads, channels, -1).transpose(2, 3)
    if with_cls_token:
        x = torch.cat((cls_tok, x), dim=2)
    if norm is not None:
        x = norm(x)
    if ndim == 3:
        x = x.squeeze(1)
    return x, out_size


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        num_heads: int,
        qkv_bias: bool = True,
        norm_cfg: dict | None = None,
        pool_kernel: tuple[int, int, int] = (3, 3, 3),
        stride_q: tuple[int, int, int] = (1, 1, 1),
        stride_kv: tuple[int, int, int] = (1, 1, 1),
        rel_pos_embed: bool = True,
        residual_pooling: bool = True,
        input_size: tuple[int, int, int] | None = None,
        with_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.with_cls_token = bool(with_cls_token)
        self.out_dims = int(out_dims)
        head_dim = self.out_dims // self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(in_dims, out_dims * 3, bias=qkv_bias)
        self.proj = nn.Linear(out_dims, out_dims)
        pool_padding = tuple(k // 2 for k in pool_kernel)
        pool_dims = out_dims // num_heads

        def build_pool(stride: tuple[int, int, int]) -> tuple[nn.Module, nn.Module]:
            pool = nn.Conv3d(
                pool_dims,
                pool_dims,
                pool_kernel,
                stride=stride,
                padding=pool_padding,
                groups=pool_dims,
                bias=False,
            )
            norm = build_norm_layer(norm_cfg or {"type": "LN"}, pool_dims)[1]
            return pool, norm

        self.pool_q, self.norm_q = build_pool(stride_q)
        self.pool_k, self.norm_k = build_pool(stride_kv)
        self.pool_v, self.norm_v = build_pool(stride_kv)
        self.residual_pooling = bool(residual_pooling)
        self.rel_pos_embed = bool(rel_pos_embed)
        if self.rel_pos_embed:
            assert input_size is not None
            rel_dim = 2 * max(input_size[1] // stride_q[1], input_size[1] // stride_kv[1]) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_dim, head_dim))
            self.rel_pos_t = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
            nn.init.trunc_normal_(self.rel_pos_w, std=0.02)
            nn.init.trunc_normal_(self.rel_pos_t, std=0.02)

    def forward(self, x: torch.Tensor, in_size: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
        batch_size, num_tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, q_shape = attention_pool(q, self.pool_q, in_size, norm=self.norm_q, with_cls_token=self.with_cls_token)
        k, k_shape = attention_pool(k, self.pool_k, in_size, norm=self.norm_k, with_cls_token=self.with_cls_token)
        v, _ = attention_pool(v, self.pool_v, in_size, norm=self.norm_v, with_cls_token=self.with_cls_token)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos_embed:
            attn = add_decomposed_rel_pos(
                attn,
                q,
                q_shape,
                k_shape,
                self.rel_pos_h,
                self.rel_pos_w,
                self.rel_pos_t,
                self.with_cls_token,
            )
        attn = attn.softmax(dim=-1)
        x = attn @ v
        if self.residual_pooling:
            if self.with_cls_token:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q
        x = x.transpose(1, 2).reshape(batch_size, -1, self.out_dims)
        return self.proj(x), q_shape


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        qkv_pool_kernel: tuple[int, int, int] = (3, 3, 3),
        stride_q: tuple[int, int, int] = (1, 1, 1),
        stride_kv: tuple[int, int, int] = (1, 1, 1),
        rel_pos_embed: bool = True,
        residual_pooling: bool = True,
        with_cls_token: bool = True,
        dim_mul_in_attention: bool = True,
        input_size: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.with_cls_token = bool(with_cls_token)
        self.dim_mul_in_attention = bool(dim_mul_in_attention)
        self.norm1 = build_norm_layer(norm_cfg or {"type": "LN", "eps": 1e-6}, in_dims)[1]
        attn_dims = out_dims if self.dim_mul_in_attention else in_dims
        self.attn = MultiScaleAttention(
            in_dims=in_dims,
            out_dims=attn_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            pool_kernel=qkv_pool_kernel,
            stride_q=stride_q,
            stride_kv=stride_kv,
            rel_pos_embed=rel_pos_embed,
            residual_pooling=residual_pooling,
            input_size=input_size,
            with_cls_token=with_cls_token,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg or {"type": "LN", "eps": 1e-6}, attn_dims)[1]
        self.mlp = MLP(
            in_channels=attn_dims,
            hidden_channels=int(attn_dims * mlp_ratio),
            out_channels=out_dims,
            act_cfg=act_cfg or {"type": "GELU"},
        )
        self.proj = nn.Linear(in_dims, out_dims) if in_dims != out_dims else None
        if np.prod(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            padding_skip = [skip // 2 for skip in kernel_skip]
            self.pool_skip = nn.MaxPool3d(kernel_skip, stride_q, padding_skip, ceil_mode=False)
            self.init_out_size = [size // s for size, s in zip(input_size or (1, 1, 1), stride_q)]
        else:
            self.pool_skip = None
            self.init_out_size = input_size

    def forward(self, x: torch.Tensor, in_size: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x_norm = self.norm1(x)
        x_attn, out_size = self.attn(x_norm, in_size)
        skip = self.proj(x_norm) if self.dim_mul_in_attention and self.proj is not None else x
        if self.pool_skip is not None:
            skip, _ = attention_pool(skip, self.pool_skip, in_size, with_cls_token=self.with_cls_token)
        x = skip + self.drop_path(x_attn)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        skip = self.proj(x_norm) if (not self.dim_mul_in_attention and self.proj is not None) else x
        return skip + self.drop_path(x_mlp), out_size


class MViT(nn.Module):
    arch_zoo = {
        "tiny": {"embed_dims": 96, "num_layers": 10, "num_heads": 1, "downscale_indices": [1, 3, 8]},
        "small": {"embed_dims": 96, "num_layers": 16, "num_heads": 1, "downscale_indices": [1, 3, 14]},
        "base": {"embed_dims": 96, "num_layers": 24, "num_heads": 1, "downscale_indices": [2, 5, 21]},
        "large": {"embed_dims": 144, "num_layers": 48, "num_heads": 2, "downscale_indices": [2, 8, 44]},
    }
    num_extra_tokens = 1

    def __init__(
        self,
        arch: str = "base",
        spatial_size: int = 224,
        temporal_size: int = 16,
        in_channels: int = 3,
        out_scales: int | Sequence[int] = -1,
        drop_path_rate: float = 0.0,
        use_abs_pos_embed: bool = False,
        interpolate_mode: str = "trilinear",
        pool_kernel: tuple[int, int, int] = (3, 3, 3),
        dim_mul: int = 2,
        head_mul: int = 2,
        adaptive_kv_stride: tuple[int, int, int] = (1, 8, 8),
        rel_pos_embed: bool = True,
        residual_pooling: bool = True,
        dim_mul_in_attention: bool = True,
        with_cls_token: bool = True,
        output_cls_token: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_cfg: dict | None = None,
        patch_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        arch_settings = self.arch_zoo[str(arch).lower()]
        self.embed_dims = int(arch_settings["embed_dims"])
        self.num_layers = int(arch_settings["num_layers"])
        self.num_heads = int(arch_settings["num_heads"])
        self.downscale_indices = list(arch_settings["downscale_indices"])
        self.dim_mul_indices = list(self.downscale_indices)
        self.num_scales = len(self.downscale_indices) + 1
        self.stage_indices = {index - 1: i for i, index in enumerate(self.downscale_indices)}
        self.stage_indices[self.num_layers - 1] = self.num_scales - 1
        self.use_abs_pos_embed = bool(use_abs_pos_embed)
        self.interpolate_mode = str(interpolate_mode)
        self.with_cls_token = bool(with_cls_token)
        self.output_cls_token = bool(output_cls_token)
        if isinstance(out_scales, int):
            out_scales = [out_scales]
        out_scale_list = list(out_scales)
        for i, index in enumerate(out_scale_list):
            if index < 0:
                out_scale_list[i] = self.num_scales + index
        self.out_scales = sorted(out_scale_list)
        patch_args = {
            "kernel_size": (3, 7, 7),
            "stride": (2, 4, 4),
            "padding": (1, 3, 3),
            "input_size": (temporal_size, spatial_size, spatial_size),
        }
        if patch_cfg:
            patch_args.update(patch_cfg)
        self.patch_embed = PatchEmbed3D(in_channels=in_channels, embed_dims=self.embed_dims, **patch_args)
        self.patch_resolution = self.patch_embed.init_out_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        if self.use_abs_pos_embed and self.patch_resolution is not None:
            num_patches = int(np.prod(self.patch_resolution))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_extra_tokens, self.embed_dims))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.blocks = nn.ModuleList()
        out_dims_list = [self.embed_dims]
        num_heads = self.num_heads
        stride_kv = list(adaptive_kv_stride)
        input_size = self.patch_resolution
        for i in range(self.num_layers):
            if i in self.downscale_indices or i in self.dim_mul_indices:
                num_heads *= head_mul
            if i in self.downscale_indices:
                stride_q = [1, 2, 2]
                stride_kv = [max(s // 2, 1) for s in stride_kv]
            else:
                stride_q = [1, 1, 1]
            if dim_mul_in_attention and i in self.dim_mul_indices:
                out_dims = out_dims_list[-1] * dim_mul
            elif (not dim_mul_in_attention) and (i + 1 in self.dim_mul_indices):
                out_dims = out_dims_list[-1] * dim_mul
            else:
                out_dims = out_dims_list[-1]
            block = MultiScaleBlock(
                in_dims=out_dims_list[-1],
                out_dims=out_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=float(dpr[i]),
                norm_cfg=norm_cfg or {"type": "LN", "eps": 1e-6},
                act_cfg={"type": "GELU"},
                qkv_pool_kernel=pool_kernel,
                stride_q=tuple(stride_q),
                stride_kv=tuple(stride_kv),
                rel_pos_embed=rel_pos_embed,
                residual_pooling=residual_pooling,
                with_cls_token=with_cls_token,
                dim_mul_in_attention=dim_mul_in_attention,
                input_size=tuple(input_size) if input_size is not None else None,
            )
            self.blocks.append(block)
            input_size = tuple(block.init_out_size) if block.init_out_size is not None else input_size
            out_dims_list.append(out_dims)
            if i in self.stage_indices:
                stage_index = self.stage_indices[i]
                if stage_index in self.out_scales:
                    self.add_module(f"norm{stage_index}", build_norm_layer(norm_cfg or {"type": "LN", "eps": 1e-6}, out_dims)[1])
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor | None], ...]:
        batch_size = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(batch_size, -1, -1), x), dim=1)
        if self.use_abs_pos_embed and hasattr(self, "pos_embed"):
            x = x + resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=self.num_extra_tokens,
            )
        if not self.with_cls_token:
            x = x[:, 1:]
        outs: list[list[torch.Tensor | None]] = []
        for i, block in enumerate(self.blocks):
            x, patch_resolution = block(x, patch_resolution)
            if i in self.stage_indices:
                stage_index = self.stage_indices[i]
                if stage_index in self.out_scales:
                    _, _, channels = x.shape
                    x = getattr(self, f"norm{stage_index}")(x)
                    tokens = x.transpose(1, 2)
                    patch_token = tokens[:, :, 1:].reshape(batch_size, channels, *patch_resolution)
                    cls_token = tokens[:, :, 0] if self.with_cls_token else None
                    outs.append([patch_token, cls_token] if self.output_cls_token else [patch_token])
        return tuple(outs)
