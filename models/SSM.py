import torch
import torch.nn as nn
from .src.models.sequence.ss.s4 import S4
from .src.models.sequence.ss.fairseq.models.lra.mega_lra_block import MegaLRAEncoder
from s5 import S5
from mamba_ssm.modules.mamba_simple import Mamba

class SSM(nn.Module):
    def __init__(self,
                 d_input:int = 1,
                 d_model:int = 128,
                 d_output:int = 10, 
                 n_layers:int = 4, 
                 d_state:int = 64,
                 dropout:int = 0.2,
                 l_max:int = 1,
                 lr:float = 1e-3,
                 use_lyap:bool = False,
                 mode:str = 'nplr',
                 use_AdSS = False,
                 AdSS_Type='relu',
                 patch_size=None
                 ):
        super(SSM, self).__init__()  
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.dropout = dropout
        self.l_max = l_max
        mapping_layers = []
        self.norms = nn.ModuleList()
        self.post_norm = nn.LayerNorm(d_model)
        
        self.use_AdSS = use_AdSS
        self.AdSS_Type=AdSS_Type
        self.patch_size = patch_size
        if patch_size is not None:
            self.input = nn.Conv2d(d_input, d_model, kernel_size=patch_size, stride=patch_size)
        else:
            self.input = nn.Linear(d_input, d_model)

        if use_lyap: 
            adjusts = []
        for _ in range(self.n_layers):  
            encoder_layer = S4(self.d_model,
                               self.d_state,
                               transposed = False,
                               activation = 'glu',
                               bidirectional=False,
                               mode = mode,
                               lr = lr,
                               use_AdSS=use_AdSS,
                               AdSS_Type=self.AdSS_Type)
            mapping_layers.append(encoder_layer)
            self.norms.append(nn.LayerNorm(d_model))
        self.mapping_layers = nn.Sequential(*mapping_layers)
        self.output = nn.Linear(d_model, d_output)


    def forward(self, x, state = None, ret_lyap = False):
        if self.patch_size is not None:
            x = self.input(x)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
        else:
            # (B C H W) -> (B L=H*W C)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
            # (B L C) -> (B L D)
            x = self.input(x)
        self.input_out = x
        self.input_out.requires_grad_()
        self.input_out.retain_grad()
        x = self.input_out
        x = self.post_norm(x)
        if ret_lyap:
            lyap = 0.0

        num_layer = 0
        for layer in self.mapping_layers:
            residual = x
            if ret_lyap:
                if num_layer == 0:
                    x, state, y_ori = layer(x, state = state, rety = True)
                    lyap += self.adjusts[num_layer](y_ori, x)
                else:
                    x, state, y_ori = layer(x, state = state, rety = True)
                    lyap += self.adjusts[num_layer](y_ori, x)
            else:
                x, state = layer(x, state = state)
            x = residual + x
            x = self.norms[num_layer](x)
            num_layer += 1
        # (B L D) -> (B D L) -> (B L D)
        x = self.output(x.mean(dim=1))
        return x
                
class S5_SSM(nn.Module):
    def __init__(self,
                 d_input:int = 1,
                 d_model:int = 128,
                 d_output:int = 10, 
                 n_layers:int = 4, 
                 d_state:int = 64,
                 dropout:int = 0.2,
                 patch_size:int=None
                 ):
        super(S5_SSM, self).__init__()  
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.dropout = dropout
        mapping_layers = []
        self.norms = nn.ModuleList()
        self.post_norm = nn.LayerNorm(d_model)
        self.patch_size = patch_size
        if patch_size is not None:
            self.input = nn.Conv2d(d_input, d_model, kernel_size=patch_size, stride=patch_size)
        else:
            self.input = nn.Linear(d_input, d_model)

        for _ in range(self.n_layers):  
            encoder_layer = S5(d_model, d_model)
            mapping_layers.append(encoder_layer)
            
            self.norms.append(nn.LayerNorm(d_model))
        
        self.mapping_layers = nn.Sequential(*mapping_layers)
        self.output = nn.Linear(d_model, d_output)


    def forward(self, x, state = None, ret_lyap = False):
        if self.patch_size is not None:
            x = self.input(x)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
        else:
            # (B C H W) -> (B L=H*W C)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
            x = self.input(x)
        self.input_out = x
        self.input_out.requires_grad_()
        self.input_out.retain_grad()
        x = self.input_out
        x = self.post_norm(x)
        num_layer = 0
        for layer in self.mapping_layers:
            residual = x
            x = layer(x)
            x = residual + x
            x = self.norms[num_layer](x)
            num_layer += 1
        # (B L D) -> (B D L) -> (B L D)
        x = self.output(x.mean(dim=1))
        return x
    

class Mega(nn.Module):
    def __init__(self, 
                 d_input:int = 3,
                 d_model:int = 128,
                 hidden_dim:int = 256,
                 d_output = 10,
                 n_layers:int = 4,
                 seq_len:int = 1024,
                 patch_size:int=None):
        super(Mega, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        mapping_layers = []
        for _ in range(self.n_layers):
            encoder_layer = MegaLRAEncoder(self.d_model,
                                   hidden_dim = hidden_dim,
                                   ffn_hidden_dim = hidden_dim,
                                   num_encoder_layers = 1,
                                   max_seq_len=seq_len,
                                   dropout = 0.0,
                                   chunk_size = -1,
                                   activation = 'silu'
                                   )
            mapping_layers.append(encoder_layer)
        self.patch_size = patch_size
        if patch_size is not None:
            self.input = nn.Conv2d(d_input, d_model, kernel_size=patch_size, stride=patch_size)
        else:
            self.input = nn.Linear(d_input, d_model)
        self.mapping_layers = nn.Sequential(*mapping_layers)
        self.mapping_layers = nn.Sequential(*mapping_layers)
        self.output = nn.Linear(d_model, d_output)
    
    def forward(self, x, state = None):
        if self.patch_size is not None:
            x = self.input(x)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
        else:
            # (B C H W) -> (B L=H*W C)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
            x = self.input(x)
        self.input_out = x
        self.input_out.requires_grad_()
        self.input_out.retain_grad()
        x = self.input_out
        for layer in self.mapping_layers:
            x, _ = layer(x, state)
        output = self.output(x.mean(dim=1))
        return output
    
    

class S6_SSM(nn.Module):
    def __init__(self, 
                 d_input:int = 3,
                 d_model:int = 128,
                 d_state:int = 16,
                 d_output = 10,
                 n_layers:int = 4,
                 patch_size:int=None):
        super(S6_SSM, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_state = d_state
        mapping_layers = []
        self.norms = nn.ModuleList()
        for _ in range(self.n_layers):
            encoder_layer = Mamba(self.d_model,
                               self.d_state,
                               device='cuda:0',
                               dtype=torch.float32,)
            mapping_layers.append(encoder_layer)
            self.norms.append(nn.LayerNorm(d_model))
            
        self.post_norm = nn.LayerNorm(d_model)
        self.patch_size = patch_size
        if patch_size is not None:
            self.input = nn.Conv2d(d_input, d_model, kernel_size=patch_size, stride=patch_size)
        else:
            self.input = nn.Linear(d_input, d_model)
        self.mapping_layers = nn.Sequential(*mapping_layers)
        self.output = nn.Linear(d_model, d_output)
    
    def forward(self, x, state = None):
        if self.patch_size is not None:
            x = self.input(x)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
        else:
            # (B C H W) -> (B L=H*W C)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
            x = self.input(x)
        self.input_out = x
        self.input_out.requires_grad_()
        self.input_out.retain_grad()
        x = self.input_out
        x = self.post_norm(x)
        num_layer = 0
        
        for layer in self.mapping_layers:
            residual = x
            x = layer(x, state)
            x = residual + x
            x = self.norms[num_layer](x)
            num_layer += 1          
        
        output = self.output(x.mean(dim=1))
        
        return output