# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch
# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..utils import logging_info
from .dpb_v4 import DynamicPosBiasV4
from .dpb_v8 import DynamicPosBiasV8


class DynamicToepliztMultiheadV4(nn.Module):
    def __init__(
        self,
        h,
        n,
        dim,
        dpb_dim,
        causal=False,
        use_exp=False,
        use_neg_exp=False,
        use_decay=False,
        use_multi_decay=False,
        residual=False,
        act="relu",
        use_pad=False,
        par_type=1,
        dpb_type=4,
        l=10,
        transform_type=1,
        gamma=0.999,
        bias=True,
        act_type="none",
        layers=3,
    ):
        super().__init__()
        self.h = h
        self.n = n
        self.dim = dim
        self.causal = causal
        self.use_exp = use_exp
        self.use_neg_exp = use_neg_exp
        self.use_pad = use_pad
        self.par_type = par_type
        self.dpb_type = dpb_type
        if self.use_exp:
            self.zero_value = float("-inf")
        else:
            self.zero_value = 0

        self.use_decay = use_decay
        if self.use_decay:
            self.gamma = nn.Parameter(
                torch.ones(self.h, 1, self.dim) * gamma, requires_grad=False
            )
        self.use_multi_decay = use_multi_decay
        if self.use_multi_decay:
            self.lambda_ = gamma
            self.gamma = nn.Parameter(torch.randn(self.h, 1, self.dim))

        if self.dpb_type == 4:
            self.dpb = DynamicPosBiasV4(
                dim=dpb_dim, outdim=self.h * self.dim, residual=residual, bias=bias
            )
        elif self.dpb_type == 8:
            self.dpb = DynamicPosBiasV8(
                dim=dpb_dim,
                outdim=self.h * self.dim,
                residual=residual,
                bias=bias,
                layers=layers,
            )
        else:
            self.dpb = DynamicPosBiasV4(
                dim=dpb_dim, outdim=self.h * self.dim, residual=residual, bias=bias
            )

        if self.causal:
            self.forward = self.forward_causal
        else:
            self.forward = self.forward_non_causal

        self.act_type = act_type
        self.act_fun = self.get_act_fun(self.act_type)

        # ##### for inference
        # # shape of x: b, h, n, e
        # n = 2048
        # # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        # ##### coef
        # # 1, d, 1 -> h, 1, d
        # zero = self.dpb_transform(self.get_zero())
        # if self.use_pad:
        #     l1 = min(n - 1, self.n - 1)
        #     l2 = max(0, n - 1 - l1)
        #     # n, d, 1 -> h, n, d
        #     pos_dpb = self.dpb_transform(self.get_pos(l1))
        #     neg_index = self.get_neg(l1)
        #     if self.causal:
        #         neg_dpb = neg_index
        #     else:
        #         neg_dpb = self.dpb_transform(neg_index)
        #     # padding to seq len
        #     pos = torch.cat([pos_dpb, torch.ones(self.h, l2, self.dim) * self.zero_value], dim=-2)
        #     neg = torch.cat([torch.ones(self.h, l2, self.dim) * self.zero_value, neg_dpb], dim=-2)
        # else:
        #     pos = self.dpb_transform(self.get_pos(n - 1))
        #     neg_index = self.get_neg(n - 1)
        #     if self.causal:
        #         neg = neg_index
        #     else:
        #         neg = self.dpb_transform(neg_index)

        # if self.use_exp and self.use_neg_exp:
        #     zero = -torch.exp(zero)
        #     pos = -torch.exp(pos)
        #     if not self.causal:
        #         neg = -torch.exp(neg)

        # if self.use_decay or self.use_multi_decay:
        #     coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
        #     if self.use_decay:
        #         gamma = self.gamma
        #     else:
        #         gamma = torch.sigmoid(self.gamma)
        #         gamma = self.lambda_ + (1 - self.lambda_) * gamma
        #     if self.use_exp:
        #         gamma = torch.log(gamma) * coef
        #         pos = gamma + pos
        #         neg = torch.flip(gamma, dims=[1]) + neg
        #     else:
        #         gamma = gamma ** coef
        #         pos = gamma * pos
        #         neg = torch.flip(gamma, dims=[1]) * neg
        # if self.use_exp:
        #     a = torch.exp(torch.clamp(torch.cat([zero, pos, zero, neg], dim=1), max=30, min=-60))
        # else:
        #     a = torch.cat([zero, pos, zero, neg], dim=1)
        #     a = self.act_fun(a)
        # self.a = nn.Parameter(a, requires_grad=False)

        # self.forward = self.inference
        ##### for inference

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
                return F.leaky_relu(x)

            return f
        elif act_fun == "1+elu":

            def f(x):
                return 1 + F.elu(x)

            return f
        elif act_fun == "silu":
            return F.silu
        elif act_fun == "relu2":

            def f(x):
                return torch.square(torch.relu(x))

            return f
        elif act_fun == "cos":
            return torch.cos
        elif act_fun == "sin":
            return torch.sin
        else:
            return lambda x: x

    def get_pos(self, n):
        if self.par_type == 1:
            index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0
        elif self.par_type == 2:
            index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n
        elif self.par_type == 3:
            index = torch.exp(torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n)

        return index

    def get_zero(self):
        index = torch.zeros(1).reshape(1, -1) * 1.0
        if self.par_type == 3:
            index = torch.exp(index)

        return index

    def get_neg(self, n):
        if self.causal:
            index = (
                torch.ones(self.h * n * self.dim).reshape(self.h, n, self.dim)
                * self.zero_value
            )
        else:
            if self.par_type == 1:
                index = -torch.arange(1, 1 + n).flip(0).reshape(n, -1) * 1.0
            elif self.par_type == 2:
                index = -torch.arange(1, 1 + n).flip(0).reshape(n, -1) * 1.0 / n

        return index

    def dpb_transform(self, x):
        # n, 1 -> n, (d * h)
        res = self.dpb(x)
        # n, (d * h) -> h, n, d
        res = rearrange(res, "n (h d) -> h n d", h=self.h)

        return res

    def inference(self, x, dim=-2, normalize=False):
        n = x.shape[dim]
        # a: h, n, d
        # x: ..., h, n, d
        output = self.compute(x, self.a, dim, n)

        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, self.a, dim, n)
            output = output / denorm

        return output

    def forward_causal(self, x, dim=-2, normalize=False):
        # shape of x: b, h, n, e
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # 1, d, 1 -> h, 1, d
        zero = self.dpb_transform(self.get_zero().to(x))
        if self.use_pad:
            l1 = min(n - 1, self.n - 1)
            l2 = max(0, n - 1 - l1)
            # n, d, 1 -> h, n, d
            pos_dpb = self.dpb_transform(self.get_pos(l1).to(x))
            # padding to seq len
            pos = torch.cat(
                [pos_dpb, torch.ones(self.h, l2, self.dim).to(x) * self.zero_value],
                dim=-2,
            )
        else:
            pos = self.dpb_transform(self.get_pos(n - 1).to(x))

        if self.use_exp and self.use_neg_exp:
            zero = -torch.exp(zero)
            pos = -torch.exp(pos)

        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            if self.use_exp:
                gamma = torch.log(gamma) * coef
                pos = gamma + pos
            else:
                gamma = gamma**coef
                pos = gamma * pos
        if self.use_exp:
            a = torch.exp(
                torch.clamp(torch.cat([zero, pos, zero], dim=1), max=30, min=-60)
            )
        else:
            a = torch.cat([zero, pos, zero], dim=1)
            a = self.act_fun(a)

        # a = F.pad(a, (0, 0, 0, n - 1, 0, 0, ))
        # a: h, n, d
        # x: ..., h, n, d
        output = self.compute(x, a, dim, n)

        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, a, dim, n)
            output = output / denorm

        return output
        #### for test
        # matrix = self.toeplizt_matrix(n)
        # res = torch.einsum('...nme,...me->...ne', matrix, x)
        # logging_info(torch.norm(res - output))
        ##### for test

    def forward_non_causal(self, x, dim=-2, normalize=False):
        # shape of x: b, h, n, e
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # 1, d, 1 -> h, 1, d
        zero = self.dpb_transform(self.get_zero().to(x))
        if self.use_pad:
            l1 = min(n - 1, self.n - 1)
            l2 = max(0, n - 1 - l1)
            # n, d, 1 -> h, n, d
            pos_dpb = self.dpb_transform(self.get_pos(l1).to(x))
            neg_index = self.get_neg(l1).to(x)
            if self.causal:
                neg_dpb = neg_index
            else:
                neg_dpb = self.dpb_transform(neg_index)
            # padding to seq len
            pos = torch.cat(
                [pos_dpb, torch.ones(self.h, l2, self.dim).to(x) * self.zero_value],
                dim=-2,
            )
            neg = torch.cat(
                [torch.ones(self.h, l2, self.dim).to(x) * self.zero_value, neg_dpb],
                dim=-2,
            )
        else:
            pos = self.dpb_transform(self.get_pos(n - 1).to(x))
            neg_index = self.get_neg(n - 1).to(x)
            if self.causal:
                neg = neg_index
            else:
                neg = self.dpb_transform(neg_index)

        if self.use_exp and self.use_neg_exp:
            zero = -torch.exp(zero)
            pos = -torch.exp(pos)
            if not self.causal:
                neg = -torch.exp(neg)

        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            if self.use_exp:
                gamma = torch.log(gamma) * coef
                pos = gamma + pos
                neg = torch.flip(gamma, dims=[1]) + neg
            else:
                gamma = gamma**coef
                pos = gamma * pos
                neg = torch.flip(gamma, dims=[1]) * neg
        if self.use_exp:
            a = torch.exp(
                torch.clamp(torch.cat([zero, pos, zero, neg], dim=1), max=30, min=-60)
            )
        else:
            a = torch.cat([zero, pos, zero, neg], dim=1)
            a = self.act_fun(a)
        # a: h, n, d
        # x: ..., h, n, d
        output = self.compute(x, a, dim, n)

        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, a, dim, n)
            output = output / denorm

        return output
        ##### for test
        # matrix = self.toeplizt_matrix(n)
        # res = torch.einsum('...nme,...me->...ne', matrix, x)
        # logging_info(torch.norm(res - output))
        ##### for test

    # def compute(self, x, a, dim, n):
    #     # x: b, h, n, d
    #     # a: h, n, d
    #     y = torch.fft.rfft(x, 2 * n, dim=dim)
    #     v = torch.fft.rfft(a, 2 * n, dim=dim).unsqueeze(0)
    #     logging_info(y.shape, v.shape)
    #     u = v * y
    #     output = torch.fft.irfft(u, 2 * n, dim=dim)[:, :, :n, :]

    #     return output

    def compute(self, x, a, dim, n):
        # x: b h H W d
        # a: h, n, d
        
        y = torch.fft.rfft(x, 2 * n, dim=dim)
        # ..., n, d

        if dim == -2:
            if len(y.shape) == 4:
                v = torch.fft.rfft(a, 2 * n, dim=-2).unsqueeze(0)
            else:
                v = torch.fft.rfft(a, 2 * n, dim=-2).unsqueeze(0).unsqueeze(-3)
        else:
            v = torch.fft.rfft(a, 2 * n, dim=-2).unsqueeze(0).unsqueeze(-2)
        u = v * y
        if dim == -2:
            output = torch.fft.irfft(u, 2 * n, dim=dim)[..., :n, :]
        elif dim == -3:
            output = torch.fft.irfft(u, 2 * n, dim=dim)[..., :n, :, :]

        return output

    def toeplizt_matrix(self, n):
        # c: first col, r: first row
        # 1, d, 1 -> h, 1, d
        zero = self.dpb_transform(self.get_zero())
        if self.use_pad:
            l1 = min(n - 1, self.n - 1)
            l2 = max(0, n - 1 - l1)
            # n, d, 1 -> h, n, d
            pos_dpb = self.dpb_transform(self.get_pos(l1))
            neg_index = self.get_neg(l1)
            if self.causal:
                neg_dpb = neg_index
            else:
                neg_dpb = self.dpb_transform(neg_index)
            # padding to seq len
            pos = torch.cat(
                [pos_dpb, torch.ones(self.h, l2, self.dim) * self.zero_value], dim=-2
            )
            neg = torch.cat(
                [torch.ones(self.h, l2, self.dim) * self.zero_value, neg_dpb], dim=-2
            )
        else:
            pos = self.dpb_transform(self.get_pos(n - 1))
            neg_index = self.get_neg(n - 1)
            if self.causal:
                neg = neg_index
            else:
                neg = self.dpb_transform(neg_index)

        if self.use_exp and self.use_neg_exp:
            zero = -torch.exp(zero)
            pos = -torch.exp(pos)
            if not self.causal:
                neg = -torch.exp(neg)

        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            if self.use_exp:
                gamma = torch.log(gamma) * coef
                pos = gamma + pos
                neg = torch.flip(gamma, dims=[1]) + neg
            else:
                gamma = gamma**coef
                pos = gamma * pos
                neg = torch.flip(gamma, dims=[1]) * neg
        if self.use_exp:
            c = torch.exp(torch.cat([zero, pos], dim=-2))
            r = torch.exp(torch.cat([zero, neg.flip(1)], dim=-2))
        else:
            zero = self.act_fun(zero)
            pos = self.act_fun(pos)
            if not self.causal:
                neg = self.act_fun(neg)
            c = torch.cat([zero, pos], dim=-2)
            r = torch.cat([zero, neg.flip(1)], dim=-2)
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-2)
        n = c.shape[-2]
        shape = self.h, n, n
        i, j = torch.ones(n, n).nonzero().T
        res = vals[:, j - i].reshape(self.h, n, n, -1)

        return res
