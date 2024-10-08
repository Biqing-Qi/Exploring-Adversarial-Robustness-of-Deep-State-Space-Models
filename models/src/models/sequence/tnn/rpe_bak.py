import torch.nn as nn


class Rpe(nn.Module):
    def __init__(
        self,
        dim,
        outdim,
        residual,
        act="relu",
        bias=True,
        layers=2,
        norm_type="simplermsnorm",
    ):
        super().__init__()
        self.residual = residual
        self.outdim = outdim
        self.pos_dim = dim
        self.act = act
        self.pos_proj = nn.Linear(1, self.pos_dim, bias=bias)
        self.layers = nn.ModuleList([])
        for i in range(layers):
            self.layers.append(
                nn.Sequential(
                    self.get_act(),
                    nn.Linear(self.pos_dim, self.pos_dim, bias=bias),
                )
            )
        self.out = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            self.get_act(),
            nn.Linear(self.pos_dim, self.outdim, bias=bias),
        )

    def get_act(self):
        if self.act == "silu":
            return nn.SiLU(inplace=True)
        elif self.act == "relu":
            return nn.ReLU(inplace=True)
        else:
            return nn.Identity()

    def forward(self, biases):
        x = self.pos_proj(biases)
        if self.residual:
            for m in self.layers:
                x = m(x) + x
        else:
            for m in self.layers:
                x = m(x)
        x = self.out(x)

        return x
