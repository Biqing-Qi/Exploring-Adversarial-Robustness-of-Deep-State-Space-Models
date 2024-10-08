import torch.nn as nn

from tnn import Gtu

gtu = Gtu


class Gtu(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads=1,
        dropout=0.0,
        bias=True,
        act_fun="silu",
        causal=False,
        expand_ratio=3,
        resi_param=False,
        use_norm=False,
        norm_type="simplermsnorm",
        use_decay=True,
        use_multi_decay=False,
        rpe_layers=3,
        rpe_embedding=512,
        rpe_act="relu",
        normalize=False,
        par_type=1,
        residual=False,
        gamma=0.99,
        act_type="none",
    ):
        super().__init__()
        self.d_output = d_model
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.token_mixer = gtu(
            embed_dim=d_model,
            num_heads=num_heads,
            bias=bias,
            act_fun=act_fun,
            causal=causal,
            expand_ratio=expand_ratio,
            resi_param=resi_param,
            use_norm=use_norm,
            norm_type=norm_type,
            use_decay=use_decay,
            use_multi_decay=use_multi_decay,
            rpe_layers=rpe_layers,
            rpe_embedding=rpe_embedding,
            rpe_act=rpe_act,
            normalize=normalize,
            par_type=par_type,
            residual=residual,
            gamma=gamma,
            act_type=act_type,
        )

    def forward(self, x, state=None):
        output = self.token_mixer(x)

        return output, None
