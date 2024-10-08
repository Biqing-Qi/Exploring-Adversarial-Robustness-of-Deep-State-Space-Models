import math
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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


class FlashLinearAttention(nn.Module):
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
        norm_type="scale_norm",
        eps=1e-5,
        max_position_embeddings=512,
        expansion_factor=2,
        chunk_size=64,
    ):
        super().__init__()
        self.s = s
        self.d_output = d_model
        self.embed_dim = d_model
        self.e = int(self.embed_dim * expansion_factor)
        self.u_proj = nn.Linear(d_model, self.e)
        self.v_proj = nn.Linear(d_model, self.e)
        self.base_proj = nn.Linear(d_model, self.s)
        self.quad_q_weight = nn.Parameter(torch.randn(1, self.s))
        self.quad_q_bias = nn.Parameter(torch.zeros(1, self.s))
        self.lin_q_weight = nn.Parameter(torch.randn(1, self.s))
        self.lin_q_bias = nn.Parameter(torch.zeros(1, self.s))
        self.quad_k_weight = nn.Parameter(torch.randn(1, self.s))
        self.quad_k_bias = nn.Parameter(torch.zeros(1, self.s))
        self.lin_k_weight = nn.Parameter(torch.randn(1, self.s))
        self.lin_k_bias = nn.Parameter(torch.zeros(1, self.s))
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
        self.chunk_size = chunk_size

        nn.init.normal_(self.quad_q_weight, std=0.02)
        nn.init.normal_(self.quad_k_weight, std=0.02)
        nn.init.normal_(self.lin_q_weight, std=0.02)
        nn.init.normal_(self.lin_k_weight, std=0.02)
        nn.init.normal_(self.w, std=0.02)
        nn.init.normal_(self.a, std=0.02)
        nn.init.normal_(self.b, std=0.02)

        print("flash attention")
        print(f"s {self.s}")
        print(f"norm_type {norm_type}")
        print(f"eps {eps}")
        print(f"max_position_embeddings {max_position_embeddings}")
        print(f"expansion_factor {expansion_factor}")
        print(f"chunk_size {self.chunk_size}")

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

    def get_mask(self, num_chunks, causal_flag):
        mask = torch.ones(num_chunks, num_chunks)
        if causal_flag:
            mask = torch.tril(mask, diagonal=-1)
        return mask

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
        causal_flag = False
        # pad
        d = tgt_len
        len_pad = (self.chunk_size - d % self.chunk_size) % self.chunk_size
        query = F.pad(query, (0, 0, 0, len_pad))
        pad_tgt_len = query.shape[1]
        num_chunks = pad_tgt_len // self.chunk_size
        # pad
        # bsz, tgt_len, embed_dim
        shortcut, x = query, self.norm(query)
        # bsz, tgt_len, e
        u = self.act_fn(self.u_proj(x))
        # bsz, tgt_len, e
        v = self.act_fn(self.v_proj(x))
        # bsz, tgt_len, s
        base = self.act_fn(self.base_proj(x))
        # base = base * weight + bias
        quad_q_base = base * self.quad_q_weight + self.quad_q_bias
        quad_k_base = base * self.quad_k_weight + self.quad_k_bias
        lin_q_base = base * self.lin_q_weight + self.lin_q_bias
        lin_k_base = base * self.lin_k_weight + self.lin_k_bias
        # reshape
        # bsz, tgt_len, e -> bsz, chunk_num, chunk_size, e
        # b, g, n, e
        quad_q_base = quad_q_base.contiguous().reshape(bsz, -1, self.chunk_size, self.s)
        quad_k_base = quad_k_base.contiguous().reshape(bsz, -1, self.chunk_size, self.s)
        lin_q_base = lin_q_base.contiguous().reshape(bsz, -1, self.chunk_size, self.s)
        lin_k_base = lin_k_base.contiguous().reshape(bsz, -1, self.chunk_size, self.s)
        v = v.contiguous().reshape(bsz, -1, self.chunk_size, self.e)
        u = u.contiguous().reshape(bsz, -1, self.chunk_size, self.e)
        # base = torch.einsum("...r,hr->...hr", base, self.weight) + self.bias
        quad_q = rope(quad_q_base, dim=[1, 2])
        quad_k = rope(quad_k_base, dim=[1, 2])
        lin_q = rope(lin_q_base, dim=[1, 2])
        lin_k = rope(lin_k_base, dim=[1, 2])
        # bsz, tgt_len, e -> bsz, chunk_num, chunk_size, e
        # quad
        quad_qk = torch.einsum("bgne,bgme->bgnm", quad_q, quad_k)
        bias = self.rel_pos_bias(self.max_position_embeddings)[
            :, : self.chunk_size, : self.chunk_size
        ]  # [0].repeat(bsz, num_chunks, 1, 1)
        kernel = torch.square(torch.relu(quad_qk / self.chunk_size + bias))

        if attn_mask is not None:
            causal_flag = True
            attn_mask = (
                torch.tril(torch.ones(self.chunk_size, self.chunk_size)) == 0
            ).to(v.device)
            kernel = kernel.masked_fill(attn_mask, 0)
        # bsz, chunk_num, chunk_size, e
        quadratic = torch.einsum("bgnm,bgme->bgne", kernel, v)

        # linear
        lin_kv = torch.einsum("bgnk,bgne->bgke", lin_k, v) / self.chunk_size
        # chunk_size1 * chunk_size2的矩阵
        mask = self.get_mask(num_chunks, causal_flag).to(lin_kv)
        # bsz, chunk_size1, chunk_size2
        # mask = mask.repeat(bsz, 1, 1)
        lin_kv = torch.einsum("bhke,gh->bgke", lin_kv, mask)
        linear = torch.einsum("bgnk,bgke->bgne", lin_q, lin_kv)

        # fusion
        x = u * (linear + quadratic)
        # reshape
        # bsz, chunk_num, chunk_size, e -> sz, tgt_len, e
        x = x.contiguous().reshape(bsz, pad_tgt_len, -1)
        x = self.o(x)

        # bsz, tgt_len, s
        output = x
        output = output.contiguous()
        output = output[:, :tgt_len, :]

        return output, None
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
