from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.layers import trunc_normal_


class Adapter(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 0.25, skip_connect: bool = True) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.skip_connect = bool(skip_connect)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(self.act(self.fc1(x)))
        return x + y if self.skip_connect else y


def make_image_bucket_position(bucket_size: int, num_relative_distance: int) -> torch.Tensor:
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += bucket_size - 1
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 0] *= 2 * bucket_size - 1
    relative_position_index = torch.zeros(
        size=(bucket_size * bucket_size + 1, bucket_size * bucket_size + 1),
        dtype=relative_coords.dtype,
    )
    relative_position_index[1:, 1:] = relative_coords.sum(-1)
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (1, x.shape[1], 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def embedding(num_embeddings: int, embedding_dim: int, zero_init: bool = False) -> nn.Embedding:
    layer = nn.Embedding(num_embeddings, embedding_dim)
    nn.init.normal_(layer.weight, mean=0.0, std=embedding_dim ** -0.5)
    if zero_init:
        nn.init.constant_(layer.weight, 0)
    return layer


class LayerNorm2D(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class GeGLU(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.wi_1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.wi_0(x)) * self.wi_1(x)


class ImageAdaptor(nn.Module):
    def __init__(
        self,
        *,
        attention_heads: int,
        bucket_size: int,
        num_frames: int,
        dropout: float,
        embed_dim: int,
        shared_rp_bias: bool,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed_images = nn.Sequential(
            nn.Conv2d(3, embed_dim // 4, kernel_size=4, stride=4),
            LayerNorm2D(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2D(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
        )
        scale = embed_dim ** -0.5
        self.cls_embedding = nn.Parameter(scale * torch.randn(1, 1, embed_dim))
        self.bucket_size = int(bucket_size)
        self.num_frames = int(num_frames)
        self.pos_embed = nn.Parameter(scale * torch.randn(bucket_size ** 2 + 1, embed_dim))
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.shared_rp_bias = bool(shared_rp_bias)
        if self.shared_rp_bias:
            num_rel_dis = (2 * bucket_size - 1) * (2 * bucket_size - 1) + 3
            self.rel_pos_table = embedding(num_rel_dis, attention_heads, zero_init=True)
            self.register_buffer("rp_bucket", make_image_bucket_position(bucket_size, num_rel_dis))

    def get_rel_pos_bias(self) -> Optional[torch.Tensor]:
        if not self.shared_rp_bias:
            return None
        values = F.embedding(self.rp_bucket, self.rel_pos_table.weight)
        return values.permute(2, 0, 1).contiguous()

    def forward(self, src_images: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = src_images.size(0)
        x = self.embed_images(src_images).flatten(2).transpose(1, 2)
        cls_embedding = self.cls_embedding.expand(batch_size, -1, -1)
        x = torch.cat([cls_embedding, x], dim=1)
        x = x + self.pos_embed.unsqueeze(0)
        x = self.dropout(x)
        n = x.shape[1]
        x = rearrange(x, "(b t) n d -> (b n) t d", t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, "(b n) t d -> (b t) n d", n=n)
        return x, self.get_rel_pos_bias()


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, query: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt_len, batch_size, _ = query.size()
        q = self.q_proj(query).view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(query).view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(query).view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        q = q * self.scaling
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias
        attn_probs = self.dropout(F.softmax(attn_weights, dim=-1))
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, batch_size, self.embed_dim)
        return self.out_proj(self.ln(attn))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        *,
        activation_dropout: float,
        attention_dropout: float,
        attention_heads: int,
        bucket_size: int,
        dropout: float,
        drop_path_rate: float,
        embed_dim: int,
        ffn_embed_dim: int,
        layer_scale_init_value: float,
        num_tadapter: int,
        num_frames: int,
        scale: float,
        rp_bias: bool,
    ) -> None:
        super().__init__()
        self.rp_bias = bool(rp_bias)
        self.num_frames = int(num_frames)
        self.self_attn = MultiheadAttention(embed_dim, attention_heads, dropout=attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones(embed_dim))
        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones(embed_dim))
        if self.rp_bias:
            num_rel_dis = (2 * bucket_size - 1) * (2 * bucket_size - 1) + 3
            self.rel_pos_table = embedding(num_rel_dis, attention_heads, zero_init=True)
            self.register_buffer("rp_bucket", make_image_bucket_position(bucket_size, num_rel_dis))
        self.image_ffn = nn.Sequential(
            GeGLU(embed_dim, ffn_embed_dim),
            self.activation_dropout,
            nn.LayerNorm(ffn_embed_dim),
            nn.Linear(ffn_embed_dim, embed_dim),
        )
        self.mlp_adapter = Adapter(embed_dim, skip_connect=False)
        self.s_adapter = Adapter(embed_dim)
        self.t_adapter = Adapter(embed_dim, skip_connect=False)
        self.t_adapter_in = Adapter(embed_dim) if num_tadapter == 2 else None
        self.scale = float(scale)

    def get_rel_pos_bias(self) -> torch.Tensor:
        values = F.embedding(self.rp_bucket, self.rel_pos_table.weight)
        return values.permute(2, 0, 1).contiguous()

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        n, _, _ = x.shape
        residual = x
        xt = rearrange(x, "n (b t) d -> t (b n) d", t=self.num_frames)
        xt_input = self.self_attn_layer_norm(xt)
        if self.t_adapter_in is not None:
            xt_input = self.t_adapter_in(xt_input)
        xt = self.t_adapter(self.self_attn(xt_input))
        xt = rearrange(xt, "t (b n) d -> n (b t) d", n=n)
        x = x + self.drop_path(xt)
        if self.rp_bias:
            batch = x.shape[1]
            attn_bias = self.get_rel_pos_bias().unsqueeze(0).expand(batch, -1, -1, -1).flatten(0, 1)
        x = self.s_adapter(self.self_attn(self.self_attn_layer_norm(x), attn_bias))
        x = residual + self.drop_path(self.gamma_1 * x)
        residual = x
        xn = self.final_layer_norm(x)
        x = residual + self.gamma_2 * self.dropout(self.image_ffn(xn)) + self.drop_path(self.scale * self.mlp_adapter(xn))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        activation_dropout: float,
        attention_dropout: float,
        attention_heads: int,
        bucket_size: int,
        dropout: float,
        drop_path_rate: float,
        embed_dim: int,
        ffn_embed_dim: int,
        layers: int,
        layer_scale_init_value: float,
        num_tadapter: int,
        num_frames: int,
        scale: float,
        rp_bias: bool,
        use_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(
                activation_dropout=activation_dropout,
                attention_dropout=attention_dropout,
                attention_heads=attention_heads,
                bucket_size=bucket_size,
                dropout=dropout,
                drop_path_rate=dpr[i],
                embed_dim=embed_dim,
                ffn_embed_dim=ffn_embed_dim,
                layer_scale_init_value=layer_scale_init_value,
                num_tadapter=num_tadapter,
                num_frames=num_frames,
                scale=scale,
                rp_bias=rp_bias,
            )
            for i in range(layers)
        )
        self.image_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, image_info: tuple[torch.Tensor, Optional[torch.Tensor]]) -> torch.Tensor:
        x, attn_bias = image_info
        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(0).expand(x.size(0), -1, -1, -1).flatten(0, 1)
        x = x.transpose(0, 1)
        for layer in self.layers:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x, attn_bias)
            else:
                x = layer(x, attn_bias)
        return self.image_layer_norm(x)


class OnePeaceViT(nn.Module):
    def __init__(
        self,
        *,
        activation_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_heads: int = 24,
        adapter_scale: float = 0.5,
        bucket_size: int = 16,
        num_tadapter: int = 1,
        num_frames: int = 32,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_dim: int = 1536,
        ffn_embed_dim: int = 6144,
        layers: int = 40,
        layer_scale_init_value: float = 1e-2,
        rp_bias: bool = False,
        shared_rp_bias: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.image_adapter = ImageAdaptor(
            attention_heads=attention_heads,
            bucket_size=bucket_size,
            num_frames=num_frames,
            dropout=dropout,
            embed_dim=embed_dim,
            shared_rp_bias=shared_rp_bias,
        )
        self.encoder = TransformerEncoder(
            activation_dropout=activation_dropout,
            attention_dropout=attention_dropout,
            attention_heads=attention_heads,
            bucket_size=bucket_size,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            layers=layers,
            layer_scale_init_value=layer_scale_init_value,
            num_tadapter=num_tadapter,
            num_frames=num_frames,
            scale=adapter_scale,
            rp_bias=rp_bias,
            use_checkpoint=use_checkpoint,
        )
        self.apply(self._init_weights)
        for module in self.modules():
            if isinstance(module, Adapter):
                nn.init.constant_(module.fc2.weight, 0)
                nn.init.constant_(module.fc2.bias, 0)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c t h w -> (b t) c h w")
        return self.encoder(self.image_adapter(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, frames, _, _ = x.shape
        x = self.forward_features(x).transpose(0, 1)
        x = x[:, 0]
        x = rearrange(x, "(b t) d -> b d t", b=batch_size, t=frames)
        return x.unsqueeze(-1).unsqueeze(-1)
