"""InternVideo2 single-modality architecture.

This mirrors the OpenGVLab InternVideo2 classifier head used by the K400
fine-tuned checkpoints: a 1B video encoder followed by an attentive CLIP
projector and a 768-wide classification head.

NOTE: The model is fine-tuned on K400 starting from a K710-pretrained checkpoint.
The head has 400 outputs, but the K400 class IDs inside K710 are not contiguous —
they are listed (in K400 label order) by ``_K400_SORTED_INDICES`` below.
The forward pass must apply this permutation so logit[i] corresponds to K400
class i, not K710 class i.
"""
from __future__ import annotations

# Maps K400 label index (0-399) → column index in the head's 400-output weight
# matrix.  Taken verbatim from OpenGVLab/InternVideo2 single_modality/models/internvideo2.py.
_K400_SORTED_INDICES: list[int] = [
    341, 158, 189, 16, 398, 302, 202, 318, 80, 323, 249, 315, 18, 88, 365, 52, 257, 103, 113,
    162, 75, 338, 388, 352, 308, 125, 159, 82, 10, 44, 92, 396, 185, 258, 383, 178, 71, 260,
    15, 335, 192, 326, 58, 133, 172, 120, 334, 280, 306, 101, 337, 173, 203, 356, 4, 209, 332,
    7, 65, 115, 95, 81, 232, 344, 303, 201, 342, 351, 165, 397, 252, 368, 285, 244, 363, 355,
    79, 268, 110, 343, 72, 219, 321, 208, 345, 340, 84, 61, 206, 188, 62, 55, 29, 237, 2, 286,
    245, 90, 8, 372, 325, 380, 226, 274, 346, 354, 97, 28, 246, 194, 212, 26, 281, 147, 215,
    264, 30, 14, 301, 275, 66, 265, 224, 104, 121, 357, 117, 54, 107, 279, 109, 122, 289, 78,
    59, 241, 179, 291, 349, 142, 152, 220, 311, 386, 145, 239, 392, 99, 266, 100, 176, 314, 167,
    64, 160, 216, 49, 207, 222, 184, 171, 22, 234, 148, 339, 218, 294, 324, 233, 262, 9, 377,
    41, 390, 53, 150, 361, 73, 247, 96, 60, 364, 298, 70, 395, 143, 236, 336, 196, 385, 33,
    144, 1, 307, 393, 256, 263, 375, 235, 273, 243, 106, 366, 271, 186, 287, 51, 299, 175, 276,
    369, 57, 11, 373, 35, 163, 297, 195, 399, 290, 382, 319, 134, 40, 310, 223, 151, 270, 3,
    387, 137, 31, 309, 217, 17, 374, 190, 277, 327, 135, 394, 50, 284, 177, 67, 379, 141, 353,
    108, 37, 136, 197, 272, 21, 312, 213, 164, 182, 250, 91, 89, 253, 199, 333, 248, 63, 119,
    0, 130, 102, 32, 227, 362, 296, 23, 47, 156, 180, 183, 313, 5, 350, 389, 328, 112, 93, 378,
    359, 83, 282, 174, 371, 48, 360, 24, 376, 68, 42, 221, 140, 181, 118, 116, 381, 94, 77,
    27, 45, 87, 230, 292, 76, 39, 169, 131, 19, 126, 367, 105, 114, 193, 210, 305, 149, 98,
    259, 200, 12, 320, 254, 146, 278, 242, 261, 36, 293, 251, 214, 25, 304, 204, 157, 255, 111,
    229, 283, 128, 161, 170, 86, 74, 138, 6, 198, 384, 187, 155, 348, 154, 166, 124, 205, 132,
    13, 34, 225, 43, 347, 228, 358, 38, 127, 231, 316, 269, 288, 139, 168, 46, 238, 317, 69,
    211, 123, 391, 330, 295, 322, 329, 129, 240, 153, 267, 85, 300, 20, 191, 56, 370, 331,
]

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: int | None = None,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.num_heads = num_heads
        head_dim = attn_head_dim or dim // num_heads
        all_head_dim = head_dim * num_heads
        if all_head_dim != dim:
            raise ValueError("CrossAttention requires all_head_dim == dim")
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        key_tokens = k.shape[1]
        value_tokens = v.shape[1]

        q = F.linear(x, self.q.weight, self.q_bias)
        k = F.linear(k, self.k.weight, self.k_bias)
        v = F.linear(v, self.v.weight, self.v_bias)

        q = q.reshape(batch, tokens, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(batch, key_tokens, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(batch, value_tokens, self.num_heads, -1).transpose(1, 2)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, -1)
        return self.proj_drop(self.proj(x))


class AttentiveBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        attn_head_dim: int | None = None,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
            out_dim=out_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        x_q = self.norm1_q(x_q)
        x_k = self.norm1_k(x_kv)
        x_v = self.norm1_v(x_kv)
        return self.drop_path(self.cross_attn(x_q, k=x_k, v=x_v))


class AttentionPoolingBlock(AttentiveBlock):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = x.mean(1, keepdim=True)
        return super().forward(x_q, x).squeeze(1)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        qk_normalization: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Attention dim must be divisible by num_heads")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if self.qk_normalization:
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(batch, tokens, self.num_heads, -1)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(batch, tokens, self.num_heads, -1)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        bias: bool | tuple[bool, bool] = True,
        drop: float | tuple[float, float] = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        qk_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
            qk_normalization=qk_normalization,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        num_frames: int = 8,
        tubelet_size: int = 1,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1]),
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(3).permute(0, 2, 3, 1)
        return self.norm(x)


class InternVideo2(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        patch_size: int = 14,
        img_size: int = 224,
        qkv_bias: bool = False,
        drop_path_rate: float = 0.25,
        embed_dim: int = 1408,
        head_drop_path_rate: float = 0.0,
        num_heads: int = 16,
        mlp_ratio: float = 48 / 11,
        init_values: float = 1e-5,
        qk_normalization: bool = True,
        depth: int = 40,
        attn_pool_num_heads: int = 16,
        clip_embed_dim: int = 768,
        num_frames: int = 8,
        tubelet_size: int = 1,
        fc_drop_rate: float = 0.0,
        num_classes: int = 1000,
        init_scale: float = 0.001,
        **_: object,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer_for_blocks,
                    drop_path=dpr[i],
                    init_values=init_values,
                    qk_normalization=qk_normalization,
                )
                for i in range(depth)
            ]
        )
        self.clip_projector = AttentionPoolingBlock(
            dim=embed_dim,
            num_heads=attn_pool_num_heads,
            qkv_bias=True,
            drop_path=head_drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-5),
            out_dim=clip_embed_dim,
        )
        self.fc_norm = nn.LayerNorm(clip_embed_dim)
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0.0 else nn.Identity()
        self.head = nn.Linear(clip_embed_dim, num_classes)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def fix_init_weight(self) -> None:
        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks, start=1):
            rescale(layer.attn.proj.weight.data, layer_id)
            rescale(layer.mlp.fc2.weight.data, layer_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x.type(self.dtype))
        batch, frames, patches, channels = x.shape
        x = x.view(batch, frames * patches, channels)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.clip_projector(x)
        x = self.fc_norm(x)
        x = self.head(self.fc_dropout(x))
        # Re-map head columns to K400 label order.  The checkpoint was fine-tuned
        # starting from K710 weights; the 400 K400 classes are not at consecutive
        # indices in the weight matrix.  Only apply when this is a 400-class head.
        if x.shape[1] == len(_K400_SORTED_INDICES):
            x = x[:, _K400_SORTED_INDICES]
        return x


def internvideo2_1B_patch14_224(**kwargs: object) -> InternVideo2:
    return InternVideo2(
        img_size=224,
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        **kwargs,
    )
