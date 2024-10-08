import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import logging_info


class GLU(nn.Module):
    def __init__(
        self,
        d1,
        glu_expand_ratio=2,
        act_fun="None",
        fina_act="None",
        dropout=0.0,
        bias=True,
    ):
        super().__init__()

        d2 = int(d1 * glu_expand_ratio)
        self.d_output = d1
        self.l1 = nn.Linear(d1, d2, bias=bias)
        self.l2 = nn.Linear(d1, d2, bias=bias)
        self.l3 = nn.Linear(d2, d1, bias=bias)
        self.act_fun = self.get_act_fun(act_fun)
        self.p = dropout
        if self.p > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        self.fina_act = self.get_act_fun(fina_act)

        logging_info(f"act_fun {act_fun}")
        logging_info(f"dropout {self.p}")
        logging_info(f"final {fina_act}")

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
        elif act_fun == "swish":
            return F.silu
        else:
            return lambda x: x

    def forward(self, x, state):
        o1 = self.l1(x)
        weight = self.act_fun(o1)
        if self.p > 0.0:
            weight = self.dropout(weight)
        o2 = self.l2(x)
        output = weight * o2
        output = self.l3(output)
        output = self.fina_act(output)

        return output, None
