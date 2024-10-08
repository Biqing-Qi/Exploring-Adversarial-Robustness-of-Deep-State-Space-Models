import torch
import torch.nn as nn
import torch.nn.functional as F
from .s4 import S4
from ..glu import GLU

class S4_GLU(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        l_max=1,  # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
        channels=1,  # maps 1-dim to C-dim
        bidirectional=False,
        # Arguments for FF
        activation="gelu",  # activation in between SS and FF
        ln=False,  # Extra normalization
        postact=None,  # activation after FF
        initializer=None,  # initializer on FF
        weight_norm=False,  # weight normalization on FF
        hyper_act=None,  # Use a "hypernetwork" multiplication
        dropout=0.0,
        transposed=True,  # axis ordering (B, L, D) or (B, D, L)
        verbose=False,
        shift=False,
        linear=False,
        glu_expand_ratio=2,
        act_fun="sigmoid",
        fina_act="silu",
        bias=True,
        # SSM Kernel arguments
        **kernel_args,
    ):
        super().__init__()
        self.d_output = d_model
        self.s4 = S4(d_model,
                     d_state,
                     dropout = dropout,
                     transposed = transposed,
                     bidirectional = bidirectional,
                     hyper_act = hyper_act,
                     postact = postact,
                     **kernel_args ) 
        self.glu = GLU(d_model,
                       glu_expand_ratio = glu_expand_ratio,
                       act_fun = act_fun,
                       fina_act = fina_act)
        
        def forward(self, x, state = None):
            x = self.S4