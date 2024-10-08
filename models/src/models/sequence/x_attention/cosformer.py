import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from src.models.nn.components import Normalization
from torch import Tensor, nn


class CosformerAttention(nn.Module):
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
        # add
        index=0,
        use_relu=True,
        use_elu=False,
        use_leak=False,
        max_l=1024,
        has_out=False,
        causal=False,
        resi=False,
        norm_type="batch",
    ):
        # add
        self.d_output = d_model
        self.index = index

        super().__init__()
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
        self.v_proj = nn.Linear(self.vdim, d_model, bias=bias)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.layer_norm = Normalization(d_model, _name_=norm_type)

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

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
        eps=1e-4,
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
        head_dim = embed_dim // num_heads

        tgt_len, bsz, embed_dim = query.size()
        m = max(src_len, tgt_len)

        scaling = float(embed_dim) ** -0.5
        # q *= self.scaling
        # L, N, E1
        q = self.q_proj(query)
        # S, N, E1
        k = self.k_proj(key)
        # S, N, E2
        v = self.v_proj(value)

        # activation
        q = F.relu(q)
        k = F.relu(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat(
            [
                q * torch.sin(weight_index[:, :tgt_len, :] / m),
                q * torch.cos(weight_index[:, :tgt_len, :] / m),
            ],
            dim=-1,
        )
        # (N * h, S, 2 * d)
        k_ = torch.cat(
            [
                k * torch.sin(weight_index[:, :src_len, :] / m),
                k * torch.cos(weight_index[:, :src_len, :] / m),
            ],
            dim=-1,
        )

        # (N * b, e1, e2)
        # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
        kv_ = torch.einsum("nld,nlm->ndm", k_, v)
        # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
        z_ = 1 / torch.clamp_min(
            torch.einsum("nld,nd->nl", q_, torch.sum(k_, axis=1)), eps
        )
        # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
        attn_output = torch.einsum("nld,ndm,nl->nlm", q_, kv_, z_)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)

        # attn_output = self.layer_norm(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = rearrange(attn_output, "l n e -> n l e")

        return attn_output, None
