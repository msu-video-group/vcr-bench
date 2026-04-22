from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoid_encoding(n_position: int, embed_dims: int) -> torch.Tensor:
    vec = torch.arange(embed_dims, dtype=torch.float64)
    vec = (vec - vec % 2) / embed_dims
    vec = torch.pow(10000, -vec).view(1, -1)
    sinusoid_table = torch.arange(n_position).view(-1, 1) * vec
    sinusoid_table[:, 0::2].sin_()
    sinusoid_table[:, 1::2].cos_()
    return sinusoid_table.to(torch.float32).unsqueeze(0)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 768,
        conv_type: str = "Conv2d",
        kernel_size: int | Sequence[int] = 16,
        stride: int | Sequence[int] = 16,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        bias: bool = True,
        input_size: int | None = None,
    ) -> None:
        super().__init__()
        if conv_type == "Conv3d":
            self.projection = nn.Conv3d(
                in_channels,
                embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
            self.init_out_size = None
        else:
            self.projection = nn.Conv2d(
                in_channels,
                embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
            self.init_out_size = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        x = self.projection(x)
        out_size = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x, out_size


class Attention(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.num_heads = int(num_heads)
        head_embed_dims = self.embed_dims // self.num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(self.embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(self.embed_dims))
        else:
            self.q_bias = None
            self.v_bias = None
        self.qkv = nn.Linear(self.embed_dims, self.embed_dims * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        if self.q_bias is not None and self.v_bias is not None:
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(x, self.qkv.weight, qkv_bias)
        else:
            qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FFN(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        num_fcs: int = 2,
        ffn_drop: float = 0.0,
        add_identity: bool = True,
    ) -> None:
        super().__init__()
        assert num_fcs >= 2
        layers: list[nn.Module] = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    nn.GELU(),
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = nn.Identity()
        self.add_identity = bool(add_identity)

    def forward(self, x: torch.Tensor, identity: torch.Tensor | None = None) -> torch.Tensor:
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class Block(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims, eps=1e-6)
        self.attn = Attention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dims, eps=1e-6)
        self.mlp = FFN(
            embed_dims=embed_dims,
            feedforward_channels=int(embed_dims * mlp_ratio),
            num_fcs=2,
            ffn_drop=drop_rate,
            add_identity=False,
        )
        if isinstance(init_values, float) and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(embed_dims), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(embed_dims), requires_grad=True)
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma_1 is not None and self.gamma_2 is not None:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values: float = 0.0,
        use_learnable_pos_emb: bool = False,
        num_frames: int = 16,
        tubelet_size: int = 2,
        use_mean_pooling: bool = True,
        return_feat_map: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.patch_size = int(patch_size)
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type="Conv3d",
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
        )
        grid_size = img_size // patch_size
        num_patches = grid_size**2 * (num_frames // tubelet_size)
        self.grid_size = (grid_size, grid_size)
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_buffer("pos_embed", get_sinusoid_encoding(num_patches, embed_dims))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        if use_mean_pooling:
            self.norm = nn.Identity()
            self.fc_norm = nn.LayerNorm(embed_dims, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(embed_dims, eps=1e-6)
            self.fc_norm = None
        self.return_feat_map = bool(return_feat_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, height, width = x.shape
        height //= self.patch_size
        width //= self.patch_size
        x = self.patch_embed(x)[0]
        if (height, width) != self.grid_size:
            pos_embed = self.pos_embed.reshape(-1, *self.grid_size, self.embed_dims).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(height, width), mode="bicubic", align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2).reshape(1, -1, self.embed_dims)
        else:
            pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.return_feat_map:
            x = x.reshape(batch_size, -1, height, width, self.embed_dims).permute(0, 4, 1, 2, 3).contiguous()
            return x
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        return x[:, 0]


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_prob: float = 0.0,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.num_heads = int(num_heads)
        self.batch_first = bool(batch_first)
        self.attn = nn.MultiheadAttention(self.embed_dims, self.num_heads, attn_drop, batch_first=self.batch_first)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        key_pos: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        elif query_pos is not None and query_pos.shape == key.shape:
            key = key + query_pos
        out = self.attn(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        return identity + self.dropout_layer(self.proj_drop(out))


class BaseTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        drop_prob: float = 0.0,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.batch_first = bool(batch_first)
        self.pre_norm = True
        self.embed_dims = int(embed_dims)
        self.operation_order = ("norm", "self_attn", "norm", "ffn")
        self.attentions = nn.ModuleList(
            [
                MultiheadAttention(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    drop_prob=drop_prob,
                    batch_first=batch_first,
                )
            ]
        )
        self.ffns = nn.ModuleList(
            [
                FFN(
                    embed_dims=embed_dims,
                    feedforward_channels=embed_dims * 4,
                    num_fcs=2,
                    ffn_drop=0.0,
                    add_identity=True,
                )
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims, eps=1e-6), nn.LayerNorm(embed_dims, eps=1e-6)])

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        key_pos: torch.Tensor | None = None,
        attn_masks: torch.Tensor | None = None,
        query_key_padding_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        identity = query
        query = self.norms[0](query)
        key_input = query if key is None else key
        value_input = key_input if value is None else value
        query = self.attentions[0](
            query,
            key_input,
            value_input,
            identity,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=attn_masks,
            key_padding_mask=query_key_padding_mask if key is None else key_padding_mask,
        )
        identity = query
        query = self.norms[1](query)
        query = self.ffns[0](query, identity)
        return query


class TransformerLayerSequence(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.num_layers = len(layers)
        self.embed_dims = layers[0].embed_dims
        self.pre_norm = True

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        query_pos: torch.Tensor | None = None,
        key_pos: torch.Tensor | None = None,
        attn_masks: torch.Tensor | None = None,
        query_key_padding_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            query = layer(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
            )
        return query


class PatchEmbed2D(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int = 3, embed_dims: int = 768) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.projection = nn.Conv2d(in_channels, embed_dims, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        return self.projection(x).flatten(2).transpose(1, 2)


class TimeSformer(nn.Module):
    supported_attention_types = ["divided_space_time", "space_only", "joint_space_time"]

    def __init__(
        self,
        num_frames: int,
        img_size: int,
        patch_size: int,
        embed_dims: int = 768,
        num_heads: int = 12,
        num_transformer_layers: int = 12,
        in_channels: int = 3,
        dropout_ratio: float = 0.0,
        attention_type: str = "space_only",
    ) -> None:
        super().__init__()
        if attention_type not in self.supported_attention_types:
            raise AssertionError(f"Unsupported Attention Type {attention_type}!")
        if attention_type == "divided_space_time":
            raise NotImplementedError("Local TimeSformer only supports the benchmark's space_only/joint_space_time variants.")
        self.num_frames = int(num_frames)
        self.embed_dims = int(embed_dims)
        self.num_transformer_layers = int(num_transformer_layers)
        self.attention_type = attention_type
        self.patch_embed = PatchEmbed2D(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dims=embed_dims)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=dropout_ratio)
        if self.attention_type != "space_only":
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dims))
            self.drop_after_time = nn.Dropout(p=dropout_ratio)
        dpr = torch.linspace(0, 0.1, num_transformer_layers).tolist()
        self.transformer_layers = TransformerLayerSequence(
            nn.ModuleList(
                [
                    BaseTransformerLayer(
                        embed_dims=embed_dims,
                        num_heads=num_heads,
                        drop_prob=float(dpr[i]),
                        batch_first=True,
                    )
                    for i in range(num_transformer_layers)
                ]
            )
        )
        self.norm = nn.LayerNorm(embed_dims, eps=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.attention_type != "space_only":
            nn.init.trunc_normal_(self.time_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batches = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.drop_after_pos(x + self.pos_embed)
        if self.attention_type != "space_only":
            cls_tokens = x[:batches, 0, :].unsqueeze(1)
            x = x[:, 1:, :].reshape(batches, self.num_frames, -1, self.embed_dims).permute(0, 2, 1, 3)
            x = x.reshape(-1, self.num_frames, self.embed_dims)
            x = self.drop_after_time(x + self.time_embed)
            x = x.reshape(batches, -1, self.num_frames, self.embed_dims).permute(0, 2, 1, 3)
            x = x.reshape(batches, -1, self.embed_dims)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer_layers(x, None, None)
        if self.attention_type == "space_only":
            x = x.view(-1, self.num_frames, *x.size()[-2:]).mean(1)
        x = self.norm(x)
        return x[:, 0]


class TimeSformerHead(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, dropout_ratio: float = 0.0) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.dropout = nn.Dropout(p=float(dropout_ratio)) if dropout_ratio and dropout_ratio > 0 else None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc_cls(x)
