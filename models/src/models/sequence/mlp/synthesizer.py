import torch
import torch.nn as nn
import torch.nn.functional as F


class SynthesizerDense(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout=0.0, causal=False):
        super().__init__()
        self.d_output = d_model
        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, 300)
        self.max_seq_len = 300
        self.act = nn.ReLU()
        self.causal = causal
        print(f"self.causal {self.causal}")

    def forward(self, x, state=None):
        # x: b, n, d
        b, n, d = x.shape
        mask = None
        m = min(n, self.max_seq_len)
        energy = self.w2(self.act(self.w1(x)))[:, :m, :m]
        if self.causal:
            if (mask == None) or (m < n):
                mask = (torch.triu(torch.ones(m, m)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float("-inf")).to(x)
            energy = energy.masked_fill(mask == float("-inf"), float("-inf"))
        prob = F.softmax(energy, dim=-1)  # b, n, m
        output = torch.matmul(prob, x[:, :m, :])
        output = F.pad(output, (0, 0, 0, n - m, 0, 0))

        return output, state


class SynthesizerRandom(nn.Module):
    def __init__(self, max_seq_len, causal=False):
        super().__init__()
        self.w = nn.Parameter(torch.randn(max_seq_len, max_seq_len), requires_grad=True)
        self.causal = causal
        self.max_seq_len = max_seq_len
        print(f"self.causal {self.causal}")

    def forward(self, x, state=None):
        # x: b, n, d
        b, n, d = x.shape
        mask = None
        m = min(n, self.max_seq_len)
        energy = self.w[:m, :m]
        if self.causal:
            if (mask == None) or (m < n):
                mask = (torch.triu(torch.ones(m, m)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float("-inf")).to(x)
            energy = energy.masked_fill(mask == float("-inf"), float("-inf"))
        prob = F.softmax(energy, dim=-1)
        output = torch.matmul(prob, x[:, :m, :])
        output = F.pad(output, (0, 0, 0, n - m, 0, 0))

        return output, state
