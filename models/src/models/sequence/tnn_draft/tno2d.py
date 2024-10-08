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
from .dpb_v4 import SimpleRMSNorm
from .dynamic_toeplitz_encoding_multihead_v4 import DynamicToepliztMultiheadV4


class TNO(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.0,
        bias=True,
        # add
        index=0,
        act_fun="silu",
        causal=False,
        tno_expand_ratio=2,
        shrink_ratio=1,
        resi_param=False,
        # norm
        use_norm=False,
        norm_type="layernorm",
        # Toeplizt
        use_exp=False,
        use_neg_exp=False,
        tno_max_l=512,
        use_decay=False,
        use_multi_decay=False,
        dpb_embedding=512,
        dpb_act="relu",
        dpb_use_pad=False,
        normalize=False,
        par_type=1,
        dpb_type=1,
        dpb_layers=3,
        residual=False,
        l=1,
        transform_type=1,
        gamma=0.999,
        # token shift
        token_shift_type=-1,
        # tno
        tno_type=1,
        tno_H=32,
        tno_W=32,
    ):
        # add
        self.index = index

        super().__init__()
        logging_info(f"drop {dropout}")
        self.p = dropout
        if self.p > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.H = tno_H
        self.W = tno_W
        self.d_output = d_model
        self.embed_dim = d_model
        self.num_heads = n_heads
        self.head_dim = d_model // n_heads

        self.expand_ratio = tno_expand_ratio
        self.resi_param = resi_param
        logging_info(f"self.expand_ratio {self.expand_ratio}")
        logging_info(f"self.resi_param {self.resi_param}")
        if self.resi_param:
            self.d = nn.Parameter(torch.randn(self.embed_dim))

        d1 = int(self.expand_ratio * d_model)
        d1 = (d1 // self.num_heads) * self.num_heads
        d2 = d_model
        self.head_dim = d1 // n_heads
        # d^2
        self.v_proj = nn.Linear(d_model, d1, bias=bias)
        # d^2
        self.u_proj = nn.Linear(d_model, d1, bias=bias)
        # d^2
        self.o = nn.Linear(d1, d_model, bias=bias)

        self.causal = causal
        self.act = self.get_act_fun(act_fun)
        logging_info(f"act_fun {act_fun}")
        logging_info(f"causal {self.causal}")

        # toep
        self.max_l = tno_max_l
        self.use_exp = use_exp
        self.use_neg_exp = use_neg_exp
        self.use_decay = use_decay
        self.use_multi_decay = use_multi_decay
        self.dpb_embedding = dpb_embedding
        self.dpb_act = dpb_act
        self.dpb_use_pad = dpb_use_pad
        self.normalize = normalize
        self.par_type = par_type
        self.dpb_type = dpb_type
        self.residual = residual
        self.l = l
        self.transform_type = transform_type
        self.gamma = gamma
        self.bias = bias
        self.tno_type = tno_type
        self.dpb_layers = dpb_layers
        if self.tno_type == 1 or self.tno_type == 5:
            if self.tno_type == 1:
                self.forward = self.forward1
            else:
                self.forward = self.forward5
            self.toep = DynamicToepliztMultiheadV4(
                h=self.num_heads,
                n=self.max_l,
                dim=self.head_dim,
                dpb_dim=self.dpb_embedding,
                causal=self.causal,
                use_exp=self.use_exp,
                use_neg_exp=self.use_neg_exp,
                use_decay=self.use_decay,
                use_multi_decay=self.use_multi_decay,
                use_pad=self.dpb_use_pad,
                act=self.dpb_act,
                par_type=self.par_type,
                residual=self.residual,
                dpb_type=self.dpb_type,
                layers=self.dpb_layers,
                l=self.l,
                transform_type=self.transform_type,
                gamma=self.gamma,
                bias=self.bias,
                use_norm1=use_norm1,
                use_norm2=use_norm2,
            )
        elif (
            self.tno_type == 2
            or self.tno_type == 3
            or self.tno_type == 4
            or self.tno_type == 6
        ):
            if self.tno_type == 2:
                self.forward = self.forward2
            elif self.tno_type == 3:
                self.forward = self.forward3
            elif self.tno_type == 4:
                self.forward = self.forward4
            elif self.tno_type == 6:
                self.forward = self.forward6
            else:
                self.forward = self.forward4
            self.toep1 = DynamicToepliztMultiheadV4(
                h=self.num_heads,
                n=self.max_l,
                dim=self.head_dim,
                dpb_dim=self.dpb_embedding,
                causal=self.causal,
                use_exp=self.use_exp,
                use_neg_exp=self.use_neg_exp,
                use_decay=self.use_decay,
                use_multi_decay=self.use_multi_decay,
                use_pad=self.dpb_use_pad,
                act=self.dpb_act,
                par_type=self.par_type,
                residual=self.residual,
                dpb_type=self.dpb_type,
                layers=self.dpb_layers,
                l=self.l,
                transform_type=self.transform_type,
                gamma=self.gamma,
                bias=self.bias,
            )
            self.toep2 = DynamicToepliztMultiheadV4(
                h=self.num_heads,
                n=self.max_l,
                dim=self.head_dim,
                dpb_dim=self.dpb_embedding,
                causal=self.causal,
                use_exp=self.use_exp,
                use_neg_exp=self.use_neg_exp,
                use_decay=self.use_decay,
                use_multi_decay=self.use_multi_decay,
                use_pad=self.dpb_use_pad,
                act=self.dpb_act,
                par_type=self.par_type,
                residual=self.residual,
                dpb_type=self.dpb_type,
                layers=self.dpb_layers,
                l=self.l,
                transform_type=self.transform_type,
                gamma=self.gamma,
                bias=self.bias,
            )
        logging_info(f"self.num_heads {self.num_heads}")
        logging_info(f"self.max_l {self.max_l}")
        logging_info(f"self.use_exp {self.use_exp}")
        logging_info(f"self.use_neg_exp {self.use_neg_exp}")
        logging_info(f"self.use_decay {self.use_decay}")
        logging_info(f"self.use_multi_decay {self.use_multi_decay}")
        logging_info(f"self.dpb_embedding {self.dpb_embedding}")
        logging_info(f"self.dpb_act {self.dpb_act}")
        logging_info(f"self.dpb_use_pad {self.dpb_use_pad}")
        logging_info(f"self.normalize {self.normalize}")
        logging_info(f"self.par_type {self.par_type}")
        logging_info(f"self.dpb_type {self.dpb_type}")
        logging_info(f"self.residual {self.residual}")
        logging_info(f"self.l {self.l}")
        logging_info(f"self.transform_type {self.transform_type}")
        logging_info(f"self.gamma {self.gamma}")
        logging_info(f"bias {bias}")
        logging_info(f"tno_type {tno_type}")
        logging_info(f"dpb_layers {dpb_layers}")

        # norm
        self.norm_type = norm_type
        self.pre_norm = self.get_norm_fun(self.norm_type, d2)

        self.use_norm = use_norm
        if self.use_norm:
            self.norm = self.get_norm_fun(norm_type, d1)
        logging_info(f"use_norm {self.use_norm}")
        logging_info(f"norm_type {self.norm_type}")

        self.token_shift_type = token_shift_type
        logging_info(f"self.token_shift_type {self.token_shift_type}")
        if self.token_shift_type == 1:
            self.token_shift = nn.ZeroPad2d((0, 0, 1, -1))
        elif self.token_shift_type == 2:
            self.token_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.coef = 0.5

        self.par_init()

    def par_init(self):
        nn.init.normal_(self.u_proj.weight, std=0.02)
        nn.init.normal_(self.u_proj.bias, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.bias, std=0.02)
        nn.init.normal_(self.o.weight, std=0.02)
        nn.init.normal_(self.o.bias, std=0.02)

    def get_norm_fun(self, norm_type, embed_dim):
        if norm_type == "rmsnorm":
            logging_info("here! rmsnorm")
            return RMSNorm(embed_dim)
        elif norm_type == "gatedrmsnorm":
            logging_info("here! gatedrmsnorm")
            return GatedRMSNorm(embed_dim)
        elif norm_type == "simplermsnorm":
            logging_info("here! simple rmsnorm")
            return SimpleRMSNorm(embed_dim)
        elif norm_type == "scalenorm":
            logging_info("here! scale norm")
            return ScaleNorm(embed_dim)
        else:
            logging_info("here! layer norm")
            return nn.LayerNorm(embed_dim)

    def get_act_fun(self, act_fun):
        logging_info(act_fun)
        if act_fun == "gelu":
            return F.gelu
        elif act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return F.elu
        elif act_fun == "sigmoid":
            return torch.sigmoid
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

    # row, col, no share
    def forward1(self, x, state=None):
        # x: b, h * w, d
        H = self.H
        W = self.W
        num_heads = self.num_heads

        if self.token_shift_type == 1:
            x = self.token_shift(x)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(x)
            x = self.coef * q1 + (1 - self.coef) * x

        shortcut, x = x, self.pre_norm(x)
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, "b (H W) (h d) -> b h H W d", h=num_heads, H=H, W=W)
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = self.toep(output, dim=-3, normalize=self.normalize)
        output = rearrange(output, "b h H W d -> b (H W) (h d)")
        output = u * output
        if self.use_norm:
            output = self.norm(output)

        output = self.o(output) + shortcut

        return output, None

    # row, col
    def forward2(self, x, state=None):
        # x: b, h * w, d
        H = self.H
        W = self.W
        num_heads = self.num_heads

        if self.token_shift_type == 1:
            x = self.token_shift(x)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(x)
            x = self.coef * q1 + (1 - self.coef) * x

        shortcut, x = x, self.pre_norm(x)
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, "b (H W) (h d) -> b h H W d", h=num_heads, H=H, W=W)
        output = self.toep1(v, dim=-2, normalize=self.normalize)
        output = self.toep2(output, dim=-3, normalize=self.normalize)
        output = rearrange(output, "b h H W d -> b (H W) (h d)")
        output = u * output
        if self.use_norm:
            output = self.norm(output)

        output = self.o(output) + shortcut

        return output, None

    # col, row
    def forward3(self, x, state=None):
        # x: b, h * w, d
        H = self.H
        W = self.W
        num_heads = self.num_heads

        if self.token_shift_type == 1:
            x = self.token_shift(x)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(x)
            x = self.coef * q1 + (1 - self.coef) * x

        shortcut, x = x, self.pre_norm(x)
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, "b (H W) (h d) -> b h H W d", h=num_heads, H=H, W=W)
        output = self.toep2(v, dim=-3, normalize=self.normalize)
        output = self.toep1(output, dim=-2, normalize=self.normalize)
        output = rearrange(output, "b h H W d -> b (H W) (h d)")
        output = u * output
        if self.use_norm:
            output = self.norm(output)

        output = self.o(output) + shortcut

        return output, None

    # (row, col) + (col, row)
    # def forward4(self, x, state = None):
    #     # x: b, h * w, d
    #     H = self.H
    #     W = self.W
    #     num_heads = self.num_heads

    #     if self.token_shift_type == 1:
    #         x = self.token_shift(x)
    #     elif self.token_shift_type == 2:
    #         q1 = self.token_shift(x)
    #         x = self.coef * q1 + (1 - self.coef) * x

    #     shortcut, x = x, self.pre_norm(x)
    #     if self.resi_param:
    #         shortcut = shortcut * self.d
    #     u = self.act(self.u_proj(x))
    #     v = self.act(self.v_proj(x))
    #     # reshape
    #     v = rearrange(v, 'b (H W) (h d) -> b h H W d', h=num_heads, H=H, W=W)
    #     o1 = self.toep1(v, dim=-2, normalize=self.normalize)
    #     o1 = self.toep2(o1, dim=-3, normalize=self.normalize)
    #     o2 = self.toep2(v, dim=-3, normalize=self.normalize)
    #     o2 = self.toep1(o2, dim=-2, normalize=self.normalize)
    #     output = o1 + o2
    #     output = rearrange(output, 'b h H W d -> b (H W) (h d)')
    #     output = u * output
    #     if self.use_norm:
    #         output = self.norm(output)

    #     output = self.o(output) + shortcut

    #     return output, None

    def forward4(self, x, state=None):
        # x: b, h * w, d
        n = x.shape[1]
        H = int(np.sqrt(n))
        W = n // H
        num_heads = self.num_heads

        if self.token_shift_type == 1:
            x = self.token_shift(x)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(x)
            x = self.coef * q1 + (1 - self.coef) * x

        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, "b (H W) (h d) -> b h H W d", h=num_heads, H=H, W=W)
        o1 = self.toep1(v, dim=-2, normalize=self.normalize)
        o1 = self.toep2(o1, dim=-3, normalize=self.normalize)
        o2 = self.toep2(v, dim=-3, normalize=self.normalize)
        o2 = self.toep1(o2, dim=-2, normalize=self.normalize)
        output = o1 + o2
        output = rearrange(output, "b h H W d -> b (H W) (h d)")
        # dropout
        if self.p > 0:
            output = self.dropout(output)
        output = u * output
        if self.use_norm:
            output = self.norm(output)

        output = self.o(output)

        return output, None

    # 1D
    def forward5(self, x, state=None):
        # x: b, h * w, d
        H = self.H
        W = self.W
        num_heads = self.num_heads

        if self.token_shift_type == 1:
            x = self.token_shift(x)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(x)
            x = self.coef * q1 + (1 - self.coef) * x

        shortcut, x = x, self.pre_norm(x)
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, "b n (h d) -> b h n d", h=num_heads)
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = rearrange(output, "b h n d -> b n (h d)")
        output = u * output
        if self.use_norm:
            output = self.norm(output)

        output = self.o(output) + shortcut

        return output, None

    # col + row
    # def forward6(self, x, state = None):
    #     # x: b, h * w, d
    #     H = self.H
    #     W = self.W
    #     num_heads = self.num_heads

    #     if self.token_shift_type == 1:
    #         x = self.token_shift(x)
    #     elif self.token_shift_type == 2:
    #         q1 = self.token_shift(x)
    #         x = self.coef * q1 + (1 - self.coef) * x

    #     shortcut, x = x, self.pre_norm(x)
    #     if self.resi_param:
    #         shortcut = shortcut * self.d
    #     u = self.act(self.u_proj(x))
    #     v = self.act(self.v_proj(x))
    #     # reshape
    #     v = rearrange(v, 'b (H W) (h d) -> b h H W d', h=num_heads, H=H, W=W)
    #     output = self.toep2(v, dim=-3, normalize=self.normalize) + self.toep1(v, dim=-2, normalize=self.normalize)
    #     output = rearrange(output, 'b h H W d -> b (H W) (h d)')
    #     output = u * output
    #     if self.use_norm:
    #         output = self.norm(output)

    #     output = self.o(output) + shortcut

    #     return output, None
    def forward6(self, x, state=None):
        # x: b, h * w, d
        n = x.shape[1]
        H = int(np.sqrt(n))
        W = n // H
        num_heads = self.num_heads

        if self.token_shift_type == 1:
            x = self.token_shift(x)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(x)
            x = self.coef * q1 + (1 - self.coef) * x

        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, "b (H W) (h d) -> b h H W d", h=num_heads, H=H, W=W)
        output = self.toep2(v, dim=-3, normalize=self.normalize) + self.toep1(
            v, dim=-2, normalize=self.normalize
        )
        output = rearrange(output, "b h H W d -> b (H W) (h d)")
        output = u * output
        if self.use_norm:
            output = self.norm(output)

        output = self.o(output)

        return output, None
