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
from torch import Tensor, nn
from torch.nn import Dropout, Parameter
from ..utils import logging_info

# N, L, H, E: batch, length, head, dim


class NormLinearAttention(nn.Module):
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
        weight_type=-1,
        c=1.0,
        v_act=False,
        use_dropout=False,
        p=0.5,
        use_layer_norm=False,
        qk_layer_norm=False,
        seq_dropout=False,
        seq_p=0.3,
        lambda_=0.001,
        use_gelu=False,
        mem_use_gelu=False,
        mem_use_grad=True,
        mem_use_q=True,
        mem_use_k=False,
        attention_use_layer_norm=True,
        model_update_freq=1,
        act_fun="elu",
        out_use_act=True,
        init_type="default",
        norm_type="layernorm",
        use_rope=False,
        rope_type="a",
        use_v=False,
        negative_slope=0.1,
        # lrpe
        use_lrpe=False,
        core_matrix=1,
        p_matrix=1,
        max_positions=512,
        theta_type="a",
        theta_learned=False,
        householder_learned=False,
        kv_act="identity",
        # final dropout
        use_final_dropout=False,
        final_dropout=0.0,
    ):
        # add
        self.index = index

        super().__init__()
        self.d_output = d_model
        self.embed_dim = d_model
        self.kdim = kdim if kdim is not None else d_model
        self.vdim = vdim if vdim is not None else d_model
        self.qkv_same_dim = self.kdim == d_model and self.vdim == d_model

        self.num_heads = n_heads

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
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attention_use_layer_norm = attention_use_layer_norm
        # self.layer_norm = SimpleRMSNorm(d_model)
        self.layer_norm = Normalization(d_model, _name_=norm_type)

        # add begin
        self.use_relu = use_relu
        self.use_elu = use_elu
        self.use_leak = use_leak
        self.use_bound = use_bound
        self.bound = d_model**-0.5
        self.causal = causal
        self.use_gelu = use_gelu
        self.mem_use_gelu = mem_use_gelu
        self.has_out = has_out
        self.mem_use_q = mem_use_q
        self.mem_use_k = mem_use_k
        self.act_fun = act_fun
        self.out_use_act = out_use_act
        self.init_type = init_type
        self.seq_dropout = seq_dropout
        self.seq_p = seq_p
        self.use_v = use_v
        self.negative_slope = negative_slope
        self.use_dropout = use_dropout

        # TODO dropout
        if self.use_dropout:
            self.dropout_module = Dropout(dropout)
        self.use_final_dropout = use_final_dropout
        if use_final_dropout:
            self.final_dropout_module = Dropout()
        # lrpe
        self.core_matrix = core_matrix
        self.p_matrix = p_matrix
        self.max_positions = max_positions
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

        logging_info("qk_act")
        self.act = self.get_act_fun(self.act_fun)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.norm_type = norm_type
        self.weight_type = weight_type
        logging_info(f"causal {self.causal}")
        logging_info(f"has_out {self.has_out}")
        logging_info(f"attention_use_layer_norm {self.attention_use_layer_norm}")
        logging_info(f"num_heads {self.num_heads}")
        logging_info(f"act_fun_type: {act_fun}")
        logging_info(f"norm_type {self.norm_type}")
        logging_info(f"init_type {self.init_type}")
        logging_info(f"use_lrpe {self.use_lrpe}")
        logging_info(f"use_dropout {self.use_dropout}")
        logging_info(f"kv_act {kv_act}")
        logging_info(f"self.weight_type {self.weight_type}")
        logging_info(f"self.use_final_dropout {self.use_final_dropout}")
        logging_info(f"self.final_dropout {final_dropout}")

        if self.init_type == "gelu":
            self.gelu_reset()
        elif self.init_type == "default":
            self.reset_parameters()

    def get_act_fun(self, act_fun):
        logging_info(act_fun)
        if act_fun == "gelu":
            return F.gelu
        elif act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return F.elu
        elif act_fun == "sigmoid":
            return F.sigmoid
        elif act_fun == "exp":
            return torch.exp
        elif act_fun == "leak":

            def f(x):
                return F.leaky_relu(x, negative_slope=self.negative_slope)

            return f
        elif act_fun == "1+elu":

            def f(x):
                return 1 + F.elu(x)

            return f
        elif act_fun == "silu":
            return F.silu
        elif self.act_fun == "relu2":

            def f(x):
                return torch.square(torch.relu(x))

            return f
        else:
            return lambda x: x

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        logging_info("normal init")
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)

        if self.has_out:
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)

            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)

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
        # ! transpose
        query = rearrange(query, "n l e -> l n e")
        key = query
        value = query
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
        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        eps = 1e-4
        # self.i += 1

        # q *= self.scaling
        # L, N, E1
        q = self.q_proj(query)
        # S, N, E1
        k = self.k_proj(key)
        v = self.v_proj(value)

        # N, L, e1
        head_dim = embed_dim // num_heads

        l = max(src_len, tgt_len)

        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        q = self.act(q)
        k = self.act(k)

        if self.use_lrpe:
            q = self.lrpe(q)
            k = self.lrpe(k)

        if self.causal:
            if attn_mask == None:
                attn_mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(
                    0, 1
                )
                attn_mask = (
                    attn_mask.float().masked_fill(attn_mask == 0, float("-inf")).to(q)
                )
            weights = torch.bmm(q, k.transpose(1, 2))
            weights = weights.masked_fill(attn_mask == float("-inf"), 0)
            output = torch.bmm(weights, v)
        else:
            o1 = torch.matmul(k.transpose(1, 2), v)
            output = torch.bmm(q, o1)

        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        output = output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # B, N, e2
        output = self.layer_norm(output)

        # L, N, e1
        output = self.out_proj(output)
        output = rearrange(output, "l n e -> n l e")

        return output, None


NormLinearAttention = U.Transpose(NormLinearAttention)
