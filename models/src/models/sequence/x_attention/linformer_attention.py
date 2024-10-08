import math
from email.charset import QP
from tkinter import N

import torch
import torch.nn as nn
from torch.nn import functional as F


class LinformerAttention(nn.Module):
    projection_matrix = None

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0):
        super().__init__()
        self.d_output = d_model
        self.num_head = n_heads
        self.head_dim = int(d_model / n_heads)
        self.linformer_k = 128
        self.seq_len = max_seq_len

        # add
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)

        # TODO modified into the upper two lines for run_norm
        self.E = nn.Parameter(
            torch.Tensor(self.num_head, self.linformer_k, self.seq_len)
        )
        torch.nn.init.normal_(self.E, std=0.02)

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X

    def forward(self, x, mask=None, state=None):
        Q = self.split_heads(self.q_proj(x))
        K = self.split_heads(self.k_proj(x))
        V = self.split_heads(self.v_proj(x))
        b, h, l, d = Q.shape

        if l < self.seq_len:
            Q = F.pad(Q, (0, 0, 0, self.seq_len - l, 0, 0, 0, 0))
            K = F.pad(K, (0, 0, 0, self.seq_len - l, 0, 0, 0, 0))
            V = F.pad(V, (0, 0, 0, self.seq_len - l, 0, 0, 0, 0))
        print(self.E.shape,K.shape)
        mask = torch.ones(b, l).to(Q)
        K = torch.matmul(self.E, K)
        V = torch.matmul(self.E, V)

        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)

        attn = nn.functional.softmax(dot, dim=-1)

        X = torch.matmul(attn, V)
        attn_out = self.combine_heads(X)
        attn_out = attn_out[:, :l, ...]
        return attn_out, None

    def extra_repr(self):
        return f"linformer_k={self.linformer_k}"
