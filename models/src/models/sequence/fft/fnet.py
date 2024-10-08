# https://github.com/erksch/fnet-pytorch/blob/master/fnet.py
import torch
from scipy import linalg
from torch import nn

# only for test
class FourierMMLayer(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.dft_mat_seq = nn.Parameter(
            torch.tensor(linalg.dft(max_position_embeddings), requires_grad=False)
        )
        self.dft_mat_hidden = nn.Parameter(
            torch.tensor(linalg.dft(hidden_size), requires_grad=False)
        )

    def forward(self, hidden_states):
        hidden_states_complex = hidden_states.type(torch.complex128)
        return torch.einsum(
            "...ij,...jk,...ni->...nk",
            hidden_states_complex,
            self.dft_mat_hidden,
            self.dft_mat_seq,
        ).real.type(torch.float32)


class FourierFFTLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, hidden_states):
        return torch.fft.fft(torch.fft.fft(hidden_states.float(), dim=-1), dim=-2).real


class FNetLayer(nn.Module):
    def __init__(
        self,
        max_position_embeddings,
        hidden_size,
        expand_ratio=1,
        fourier="false",
        dropout_rate=0.1,
    ):
        super().__init__()
        self.fft = (
            FourierMMLayer(max_position_embeddings, hidden_size)
            if fourier == "matmul"
            else FourierFFTLayer()
        )
        self.mixing_layer_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Linear(hidden_size, hidden_size * expand_ratio)
        self.output_dense = nn.Linear(hidden_size * expand_ratio, hidden_size)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        fft_output = self.fft(hidden_states)
        fft_output = self.mixing_layer_norm(fft_output + hidden_states)
        intermediate_output = self.feed_forward(fft_output)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)
        output = self.dropout(output)
        output = self.output_layer_norm(output + fft_output)
        return output


class FNetFairseqLayer(nn.Module):
    def __init__(
        self, d_model, max_pos_embeddings, expand_ratio=1, fourier="torch", dropout=0.1
    ):
        super().__init__()
        self.d_output = d_model
        self.fnet = FNetLayer(
            max_position_embeddings=max_pos_embeddings,
            hidden_size=d_model,
            expand_ratio=expand_ratio,
            fourier=fourier,
            dropout_rate=dropout,
        )

    def forward(self, x, state=None):
        x = self.fnet(x)

        return x, None
