# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import torch

from torch import Tensor
from torch import nn

import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        x = out.permute(0, 2, 1, 3).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, y=None) -> Tensor:
        if attn_bias is not None:
            raise NotImplementedError("attn_bias is not supported without xformers")

        if y is not None:
            B, Nq, C = x.shape
            context = x if y is None else y
            Nk = context.shape[1]

            if not hasattr(self, 'q_proj') or self.q_proj is None:
                self.q_proj, self.kv_proj = convert_qkv_to_q_and_kv_proj(self.qkv)

            q = self.q_proj(x).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv_proj(context).reshape(B, Nk, 2, self.num_heads, C // self.num_heads)
            k, v = kv.unbind(dim=2)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            out = F.scaled_dot_product_attention(q, k, v)
            out = out.permute(0, 2, 1, 3).reshape(B, Nq, C)
            out = self.proj(out)
            out = self.proj_drop(out)
            return out

        else:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

            q, k, v = torch.unbind(qkv, 2)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            x = F.scaled_dot_product_attention(q, k, v)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x


def convert_qkv_to_q_and_kv_proj(qkv_layer: nn.Linear):
    """
    Convert a self-attention qkv projection layer (dim -> 3*dim) into
    separate q_proj (dim -> dim) and kv_proj (dim -> 2*dim) layers.

    Returns:
        q_proj (nn.Linear): projection for query
        kv_proj (nn.Linear): projection for key and value
    """
    assert isinstance(qkv_layer, nn.Linear), "Expected nn.Linear for qkv_layer"
    in_features = qkv_layer.in_features
    out_features = qkv_layer.out_features
    assert out_features % 3 == 0, "Output features must be divisible by 3"

    dim = out_features // 3
    device = qkv_layer.weight.device
    dtype = qkv_layer.weight.dtype

    q_proj = nn.Linear(in_features, dim, bias=qkv_layer.bias is not None).to(device=device, dtype=dtype)
    kv_proj = nn.Linear(in_features, dim * 2, bias=qkv_layer.bias is not None).to(device=device, dtype=dtype)

    # Split weights and biases
    q_weight, k_weight, v_weight = qkv_layer.weight.chunk(3, dim=0)
    q_proj.weight.data.copy_(q_weight)
    kv_proj.weight.data.copy_(torch.cat([k_weight, v_weight], dim=0))

    if qkv_layer.bias is not None:
        q_bias, k_bias, v_bias = qkv_layer.bias.chunk(3, dim=0)
        q_proj.bias.data.copy_(q_bias)
        kv_proj.bias.data.copy_(torch.cat([k_bias, v_bias], dim=0))

    return q_proj, kv_proj
