import math
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import src.models.nn.utils as U
import torch
import torch.nn.functional as F
from einops import rearrange
from models.sequence.x_attention.lrpe import Lrpe
from src.models.nn.components import Normalization
from torch import Tensor, logit, nn
from torch.nn import Dropout, Parameter
from ..utils import logging_info

# N, L, H, E: batch, length, head, dim


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [
        padded_x[:, ind : (ind + t), ...] for ind in range(forward + backward + 1)
    ]
    return torch.cat(tensors, dim=dim)


class NormLocalAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
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
        index=0,
        use_relu=True,
        use_elu=False,
        use_leak=False,
        use_bound=False,
        max_l=1024,
        has_out=False,
        causal=False,
        # use reweighting
        weight_type=-1,
        c=1.0,
        v_act=False,
        use_dropout=False,
        p=0.5,
        use_layer_norm=False,
        qk_layer_norm=False,
        seq_dropout=False,
        seq_p=0.3,
        act_fun="relu",
        negative_slope=0.1,
        # lrpe
        use_lrpe=False,
        core_matrix=1,
        p_matrix=1,
        max_positions=512,
        theta_type="a",
        theta_learned=False,
        householder_learned=False,
        # chunk_size
        chunk_size=32,
        left_window=1,
        right_window=1,
        group_type="chunk",
        use_softmax=False,
        norm_type="gatedrmsnorm",
        # final dropout
        use_final_dropout=False,
        final_dropout=0.0,
    ):
        # add
        self.index = index

        super().__init__()
        self.embed_dim = d_model
        self.d_output = d_model
        self.kdim = kdim if kdim is not None else d_model
        self.vdim = vdim if vdim is not None else d_model
        self.qkv_same_dim = self.kdim == d_model and self.vdim == d_model

        self.num_heads = n_heads
        self.dropout_module = Dropout(dropout)

        self.head_dim = d_model // n_heads
        assert (
            self.head_dim * n_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, d_model, bias=bias)
        self.v_proj = nn.Linear(self.vdim, d_model, bias=bias)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.norm_type = norm_type
        self.normalize = Normalization(d_model, _name_=norm_type)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, d_model))
            self.bias_v = Parameter(torch.Tensor(1, 1, d_model))
        else:
            self.bias_k = self.bias_v = None

        self.reset_parameters()

        # for test
        self.onnx_trace = False

        # add
        self.act_fun = act_fun
        self.negative_slope = negative_slope
        self.act = self.get_act_fun()
        self.use_softmax = use_softmax

        # lrpe add
        self.core_matrix = core_matrix
        self.p_matrix = p_matrix
        self.max_positions = max_positions
        self.causal = causal
        self.use_lrpe = use_lrpe
        self.theta_learned = theta_learned
        self.householder_learned = householder_learned
        if self.use_lrpe:
            self.lrpe = Lrpe(
                self.core_matrix,
                self.p_matrix,
                embedding_dim=self.head_dim,
                theta_type=theta_type,
                theta_learned=theta_learned,
                householder_learned=householder_learned,
            )
            # self.lrpe = lrpe(self.core_matrix, self.p_matrix, embedding_dim=self.head_dim, theta_type=theta_type, theta_learned=theta_learned, householder_learned=householder_learned)

        self.causal = causal
        self.left_window = left_window
        self.right_window = right_window
        self.group_type = group_type
        self.weight_type = weight_type
        self.use_final_dropout = use_final_dropout
        # chunk
        self.chunk_size = chunk_size
        logging_info("use relu sparse")
        logging_info(f"use lrpe {self.use_lrpe}")
        logging_info(f"num_heads {self.num_heads}")
        logging_info(f"add_bias_kv {add_bias_kv}")
        logging_info(f"act_fun {self.act_fun}")
        logging_info(f"negative_slope {self.negative_slope}")
        logging_info(f"chunk_size {self.chunk_size}")
        logging_info(f"causal {self.causal}")
        logging_info(f"self.left_window {self.left_window}")
        logging_info(f"self.right_window {self.right_window}")
        logging_info(f"self.group_type {self.group_type}")
        logging_info(f"self.use_softmax {self.use_softmax}")
        logging_info(f"self.weight_type {self.weight_type}")
        logging_info(f"self.use_final_dropout {self.use_final_dropout}")
        logging_info(f"self.final_dropout {final_dropout}")

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def get_act_fun(self):
        logging_info(self.act_fun)
        if self.act_fun == "gelu":
            return F.gelu
        elif self.act_fun == "relu":
            return F.relu
        elif self.act_fun == "elu":
            return F.elu
        elif self.act_fun == "sigmoid":
            return F.sigmoid
        elif self.act_fun == "exp":
            return torch.exp
        elif self.act_fun == "1+elu":

            def f(x):
                return F.elu(x) + 1

            return f
        elif self.act_fun == "1+relu":

            def f(x):
                return F.relu(x) + 1

            return f
        elif self.act_fun == "2+elu":

            def f(x):
                return F.elu(x) + 2

            return f
        elif self.act_fun == "relu2":

            def f(x):
                return torch.square(torch.relu(x))

            return f
        elif self.act_fun == "leak":

            def f(x):
                return F.leaky_relu(x, negative_slope=self.negative_slope)

            return f
        elif self.act_fun == "silu":
            return F.silu
        else:

            def f(x):
                return x

            return f

    def forward(
        self,
        query,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        state=None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        query = rearrange(query, "n l e -> l n e")
        key = query
        value = query
        return self.forward_chunk(
            query,
            key,
            value,
            key_padding_mask,
            incremental_state,
            need_weights,
            static_kv,
            attn_mask,
            before_softmax,
            need_head_weights,
        )

    def transform(self, q):
        q = rearrange(q, "l b (h e) -> l (b h) e", h=self.num_heads)
        q = rearrange(q, "l n e -> n l e")
        q = rearrange(q, "n (l c) e -> n l c e", c=self.chunk_size)

        return q

    # 分组版本
    def forward_chunk(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        assert key is not None and value is not None

        """
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """
        # ! transpose
        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        tgt_len, bsz, embed_dim = query.size()

        scaling = float(head_dim) ** -0.5
        # L, N, E1
        q = self.q_proj(query)
        # scale
        q *= scaling
        # S, N, E1
        k = self.k_proj(key)
        # S, N, E2
        v = self.v_proj(value)

        # 保持q, k, v seq_len维度相同
        if tgt_len < src_len:
            q = F.pad(q, (0, 0, 0, 0, 0, src_len - tgt_len))
        else:
            k = F.pad(k, (0, 0, 0, 0, 0, tgt_len - src_len))
            v = F.pad(v, (0, 0, 0, 0, 0, tgt_len - src_len))

        d = max(tgt_len, src_len)
        len_pad = (self.chunk_size - d % self.chunk_size) % self.chunk_size
        q = F.pad(q, (0, 0, 0, 0, 0, len_pad))
        k = F.pad(k, (0, 0, 0, 0, 0, len_pad))
        v = F.pad(v, (0, 0, 0, 0, 0, len_pad))

        q = self.transform(q)
        k = self.transform(k)
        v = self.transform(v)
        # n, l, c, d

        if self.use_lrpe:
            q = self.lrpe(q)
            k = self.lrpe(k)

        logits = torch.einsum("bgle,bgse->bgls", q, k)
        if not self.use_softmax:
            prob = self.act(logits)
        else:
            # logits *= scaling
            prob = F.softmax(logits, dim=-1)

        if self.causal:
            attn_mask = (
                (torch.triu(torch.ones(self.chunk_size, self.chunk_size)) == 1)
                .transpose(0, 1)
                .to(q)
            )
            prob = prob.masked_fill(attn_mask == 0, 0)
        weights = self.dropout_module(prob)

        # (N * h, g, l, s), (N * h, g, s, e2) -> (N * h, g, l, e2)
        output = torch.einsum("bgls,bgsd->bgld", weights, v)
        # (N * h, g, l, e2) -> (N * h, L, e2) -> (L, N * h, e2) -> (L, N, E2)
        output = rearrange(output, "n l c e -> n (l c) e", c=self.chunk_size)
        output = rearrange(output, "n l e -> l n e")
        output = rearrange(output, "l (b h) e -> l b (h e)", h=self.num_heads)
        output = output[:tgt_len, ...]
        # perform RMSNorm to stabilize running
        if not self.use_softmax:
            output = self.normalize(output)
        # outprojection
        output = self.out_proj(output)

        output = rearrange(output, "l n e -> n l e")
        return output, prob


NormLocalAttention = U.Transpose(NormLocalAttention)
