import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class NystromAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        max_seq_len,
        dropout=0,
    ):
        super().__init__()
        self.d_output = d_model
        self.head_dim = int(d_model / n_heads)
        self.num_head = n_heads

        self.num_landmarks = 128
        self.seq_len = max_seq_len

        self.init_option = "original"

        # add
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)

        self.use_conv = False

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

        mask = torch.ones(b, self.seq_len).to(Q)

        Q = Q * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
        K = K * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))

        if self.num_landmarks == self.seq_len:
            attn = torch.nn.functional.softmax(
                torch.matmul(Q, K.transpose(-1, -2))
                - 1e9 * (1 - mask[:, None, None, :]),
                dim=-1,
            )
            X = torch.matmul(attn, V)
        else:
            Q_landmarks = Q.reshape(
                -1,
                self.num_head,
                self.num_landmarks,
                self.seq_len // self.num_landmarks,
                self.head_dim,
            ).mean(dim=-2)
            K_landmarks = K.reshape(
                -1,
                self.num_head,
                self.num_landmarks,
                self.seq_len // self.num_landmarks,
                self.head_dim,
            ).mean(dim=-2)

            kernel_1 = torch.nn.functional.softmax(
                torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim=-1
            )
            kernel_2 = torch.nn.functional.softmax(
                torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim=-1
            )
            kernel_3 = torch.nn.functional.softmax(
                torch.matmul(Q_landmarks, K.transpose(-1, -2))
                - 1e9 * (1 - mask[:, None, None, :]),
                dim=-1,
            )
            X = torch.matmul(
                torch.matmul(kernel_1, self.iterative_inv(kernel_2)),
                torch.matmul(kernel_3, V),
            )

        if self.use_conv:
            X += self.conv(V * mask[:, None, :, None])
        attn_out = self.combine_heads(X)
        attn_out = attn_out[:, :l, ...]

        return attn_out, None

    def iterative_inv(self, mat, n_iter=6):
        I = torch.eye(mat.size(-1), device=mat.device)
        K = mat

        if self.init_option == "original":
            V = 1 / torch.max(torch.sum(K, dim=-2)) * K.transpose(-1, -2)
        else:
            V = (
                1
                / torch.max(torch.sum(K, dim=-2), dim=-1).values[:, :, None, None]
                * K.transpose(-1, -2)
            )

        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(
                0.25 * V,
                13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)),
            )
        return V

    def extra_repr(self):
        return f"num_landmarks={self.num_landmarks}, seq_len={self.seq_len}"
