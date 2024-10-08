import math
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn import Dropout, Parameter
from ..utils import logging_info

# N, L, H, E: batch, length, head, dim

# https://github.com/JunnYu/FLASHQuad_pytorch/blob/main/flash/gau.py
def rope(x, dim):
    """RoPE position embedding."""
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]
    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i
    position = torch.reshape(
        torch.arange(total_len, dtype=x.dtype, device=x.device), spatial_shape
    )
    for i in range(dim[-1] + 1, len(shape) - 1, 1):
        position = position.unsqueeze(-1)
    half_size = shape[-1] // 2
    freq_seq = -torch.arange(half_size, dtype=x.dtype, device=x.device) / float(
        half_size
    )
    inv_freq = 10000**freq_seq
    sinusoid = torch.einsum("...,d->...d", position, inv_freq)
    sin = sinusoid.sin()
    cos = sinusoid.cos()
    x1, x2 = torch.chunk(x, 2, dim=-1)

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class ScaleNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scala = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean_square = (x**2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps) * self.scala
        return x


# Flash attention
class FlashAttention(nn.Module):
    def __init__(
        self,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        # add
        s=128,
        norm_type="layer_norm",
        eps=1e-5,
        max_position_embeddings=2048,
        expansion_factor=2,
    ):
        super().__init__()
        self.s = s
        self.d_output = d_model
        self.embed_dim = d_model
        self.e = int(self.embed_dim * expansion_factor)
        self.u_proj = nn.Linear(d_model, self.e)
        self.v_proj = nn.Linear(d_model, self.e)
        self.base_proj = nn.Linear(d_model, self.s)
        self.q_weight = nn.Parameter(torch.randn(1, self.s))
        self.q_bias = nn.Parameter(torch.zeros(1, self.s))
        self.k_weight = nn.Parameter(torch.randn(1, self.s))
        self.k_bias = nn.Parameter(torch.zeros(1, self.s))
        self.o = nn.Linear(self.e, self.embed_dim)

        self.norm = (
            nn.LayerNorm(self.embed_dim, eps=eps)
            if norm_type == "layer_norm"
            else ScaleNorm(eps=eps)
        )
        self.w = nn.Parameter(torch.randn(2 * max_position_embeddings - 1))
        self.a = nn.Parameter(torch.randn(1, self.s))
        self.b = nn.Parameter(torch.randn(1, self.s))
        self.act_fn = F.silu
        self.max_position_embeddings = max_position_embeddings

        nn.init.normal_(self.q_weight, std=0.02)
        nn.init.normal_(self.k_weight, std=0.02)
        nn.init.normal_(self.w, std=0.02)
        nn.init.normal_(self.a, std=0.02)
        nn.init.normal_(self.b, std=0.02)

        print("flash attention")
        print(f"s {self.s}")
        print(f"norm_type {norm_type}")
        print(f"eps {eps}")
        print(f"max_position_embeddings {max_position_embeddings}")
        print(f"expansion_factor {expansion_factor}")

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.v_proj.weight)

        if self.has_out:
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)

    def rel_pos_bias(self, seq_len):
        """Relative position bias."""
        if seq_len <= self.max_position_embeddings:
            # Construct Toeplitz matrix directly when the sequence length is less than 512
            t = F.pad(self.w[: 2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            # Construct Toeplitz matrix using RoPE when the sequence length is over 512.
            a = rope(self.a.repeat(seq_len, 1), dim=0)
            b = rope(self.b.repeat(seq_len, 1), dim=0)
            t = torch.einsum("mk,nk ->mn", a, b)

        return t

    def forward(
        self,
        query,
        # key: Optional[Tensor],
        # value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        state=None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        key = query
        value = query
        if need_head_weights:
            need_weights = True

        assert key is not None and value is not None
        bsz, tgt_len, embed_dim = query.size()
        # bsz, tgt_len, embed_dim

        shortcut, x = query, self.norm(query)
        # bsz, tgt_len, e
        u = self.act_fn(self.u_proj(x))
        # bsz, tgt_len, e
        v = self.act_fn(self.v_proj(x))
        # bsz, tgt_len, s
        base = self.act_fn(self.base_proj(x))
        # base = base * weight + bias
        q_base = base * self.q_weight + self.q_bias
        k_base = base * self.k_weight + self.k_bias
        # base = torch.einsum("...r,hr->...hr", base, self.weight) + self.bias
        q = rope(q_base, dim=1)
        k = rope(k_base, dim=1)
        # bsz, tgt_len, tgt_len
        qk = torch.bmm(q, k.transpose(1, 2))
        bias = self.rel_pos_bias(self.max_position_embeddings)[:, :tgt_len, :tgt_len]
        kernel = torch.square(torch.relu(qk / self.max_position_embeddings + bias))
        if attn_mask is not None:
            kernel = kernel.masked_fill(attn_mask == float("-inf"), 0)

        x = u * torch.bmm(kernel, v)
        x = self.o(x)
        output = x
        output = output.contiguous()
        return output, None
