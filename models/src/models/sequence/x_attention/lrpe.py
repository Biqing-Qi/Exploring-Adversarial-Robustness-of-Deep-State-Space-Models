# - https://github.com/zh217/torch-dct
# - https://github.com/zh217/torch-dct/issues/15
# 单头版本, 适配多维

import numpy as np
import torch
import torch.functional as F
import torch.nn as nn
from ..utils import logging_info


class Lrpe(nn.Module):
    def __init__(
        self,
        core_matrix,
        p_matrix,
        max_positions=512,
        embedding_dim=768,
        theta_type="a",
        theta_learned=False,
        householder_learned=False,
    ):
        super().__init__()
        self.core_matrix = core_matrix
        self.p_matrix = p_matrix
        self.theta_type = theta_type
        self.theta_learned = theta_learned
        self.householder_learned = householder_learned

        if self.core_matrix == 1:
            if self.theta_learned:
                logging_info("learn theta!")
                self.theta = nn.Parameter(
                    10000
                    ** (-2 / embedding_dim * torch.arange(embedding_dim // 2)).reshape(
                        1, 1, -1
                    )
                )
            else:
                logging_info(f"theta_type {self.theta_type}")
            logging_info("rope")
        elif self.core_matrix == 2:
            logging_info("mixed")
        elif self.core_matrix == 3:
            logging_info("permutation")
            permutation = self.get_permutation(max_positions, embedding_dim)
            logging_info(permutation.shape)
            self.register_buffer("permutation", permutation)
        elif self.core_matrix == 4:
            if self.theta_learned:
                logging_info("learn theta!")
                self.theta = nn.Parameter(
                    10000
                    ** (-2 / embedding_dim * torch.arange(embedding_dim)).reshape(
                        1, 1, -1
                    )
                )
            else:
                logging_info(f"theta_type {self.theta_type}")
            logging_info("complex exp")

        if self.p_matrix == 1:
            logging_info("Identity")
        elif self.p_matrix == 2:
            logging_info("DCT")
        elif self.p_matrix == 3:
            logging_info("Householder")
            if self.householder_learned:
                logging_info("learn householder!")
                self.v = nn.Parameter(torch.randn(embedding_dim))
            else:
                v = torch.randn(embedding_dim)
                v = v / torch.norm(v)
                logging_info(f"house holder norm is {torch.norm(v)}")
                self.v = nn.Parameter(v, requires_grad=False)
        elif self.p_matrix == 4:
            logging_info("Fourier")
        elif self.p_matrix == 5:
            logging_info("odd_even")

        self.p = self.get_p()
        self.core_transform = self.get_core_transform()
        self.p_transpose = self.get_p_transpose()

    def forward1(self, x):
        # b, l, e
        x = self.p(x)
        x = self.core_transform(x)
        x = self.p_transpose(x)

        return x

    def forward(self, x):
        # b, l, e
        x = self.p(x)
        x = self.core_transform(x)
        return x

    def get_p(self):
        if self.p_matrix == 1:

            def f(x):
                return x

            return f
        elif self.p_matrix == 2:
            return self.dct
        elif self.p_matrix == 3:
            return self.householder
        elif self.p_matrix == 4:

            def f(x):
                return torch.fft.fft(x, norm="ortho")

            return f
        elif self.p_matrix == 5:
            return self.odd_even_permutation

    def get_p_transpose(self):
        if self.p_matrix == 1:

            def f(x):
                return x

            return f
        elif self.p_matrix == 2:
            return self.idct
        elif self.p_matrix == 3:
            return self.householder
        elif self.p_matrix == 4:

            def f(x):
                return torch.fft.ifft(x, norm="ortho")

            return f
        elif self.p_matrix == 5:
            return self.odd_even_permutation_transpose

    def get_core_transform(self):
        if self.core_matrix == 1:
            return self.rope
        elif self.core_matrix == 2:
            return self.mix_rope
        elif self.core_matrix == 3:
            # todo
            return self.do_permutation
        elif self.core_matrix == 4:
            return self.complex_exp

    def get_permutation(self, max_positions, embedding_dim):
        permutation = torch.randperm(embedding_dim).reshape(1, -1)
        # 1 * d
        expanded = [torch.arange(embedding_dim).unsqueeze(0)]
        for _ in range(max_positions - 1):
            previous = expanded[-1]
            current = previous.gather(-1, permutation)
            expanded.append(current)
        expanded = torch.stack(expanded, dim=1)
        return expanded

    def odd_even_permutation_transpose(self, x):
        # d = e - e // 2
        # k->2k, k <= d; k-> 2(k - d) + 1, k > d
        e = x.shape[-1]
        d = e - e // 2
        permutation = torch.cat(
            [2 * torch.arange(d), 2 * (torch.arange(d, e) - d) + 1]
        ).to(x.device)
        x = x.gather(-1, permutation.expand_as(x))

        return x

    def odd_even_permutation(self, x):
        # d = e - e // 2
        # 2k->k, 2k+1->d + k,
        e = x.shape[-1]
        d = e - e // 2
        permutation = torch.arange(e)
        index = torch.arange(e)
        permutation[::2] = index[::2] // 2
        permutation[1::2] = (index[1::2] - 1) // 2 + d
        permutation = permutation.to(x.device)
        x = x.gather(-1, permutation.expand_as(x))

        return x

    def do_permutation(self, x):
        b, l, e = x.shape
        x = x.gather(-1, self.permutation[:, :l, :].expand_as(x))

        return x

    def rope(self, x):
        # b, l, d = x.shape
        d = x.shape[-1]
        e = d - 1 if d % 2 == 1 else d
        return self.mix_transform(x, e)

    def mix_rope(self, x):
        # b, l, d = x.shape
        d = x.shape[-1]
        assert d >= 3
        # split
        e = d // 2
        # 转换为偶数
        if e % 2:
            e += 1
        return self.mix_transform(x, e)

    def mix_transform(self, x, e):
        assert e % 2 == 0
        # b, l, d = x.shape
        l, d = x.shape[-2], x.shape[-1]
        m = len(x.shape)
        # 后e项
        x1 = x[..., e:]
        # 前e项做rope
        x = x[..., :e]
        if self.theta_learned:
            theta = self.theta
        else:
            if self.theta_type == "a":
                theta = 10000 ** (-2 / e * torch.arange(e // 2))
            elif self.theta_type == "b":
                theta = np.pi / 2 / l / (e // 2) * torch.arange(1, e // 2 + 1)
            elif self.theta_type == "c":
                theta = np.pi / 2 / l / torch.arange(1, e // 2 + 1)
            # 调整为相同形状
            ### ...e
            for i in range(len(x.shape) - 1):
                theta = theta.unsqueeze(0)
            theta = theta.to(x)
        # theta = torch.cat([theta, theta], dim=-1)
        theta = torch.stack([theta, theta], dim=-1).reshape(1, 1, e)
        ### l, 1
        index = torch.arange(l).reshape(-1, 1).to(x)
        ### ...., l, 1
        for _ in range(len(theta.shape) - 2):
            index = index.unsqueeze(0)
        # theta = torch.einsum('...e,...ln->...le', theta, index)
        theta = theta * index
        # (-q1, -q3), (q0, q2) -> (-q1, q0, -q3, q2)
        x_half = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        x_transform = x * torch.cos(theta) + x_half * torch.sin(theta)

        if e != d:
            x_transform = torch.cat([x_transform, x1], dim=-1)

        return x_transform

    def complex_exp(self, x):
        b, l, e = x.shape
        if self.theta_learned:
            theta = self.theta
        else:
            if self.theta_type == "a":
                theta = 10000 ** (-2 / e * torch.arange(e))
            theta = theta.reshape(1, 1, -1).to(x.device)
        matrix = theta * torch.arange(l).reshape(1, -1, 1).to(x.device)
        sin_cos = torch.complex(torch.cos(matrix), torch.sin(matrix)).to(x.device)
        x = self.element_wise_complex(x, sin_cos)

        return x

    # https://stackoverflow.com/questions/63855692/matrix-multiplication-for-complex-numbers-in-pytorch
    def element_wise_complex(self, t1, t2):
        return torch.complex(
            t1.real * t2.real - t1.imag * t2.imag, t1.real * t2.imag + t1.imag * t2.real
        )

    def dct(self, x):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        # Vc = torch.rfft(v, 1, onesided=False)
        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

        k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V

    def idct(self, X):
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct(dct(x)) == x
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        # norm
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = (
            torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
            * np.pi
            / (2 * N)
        )
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        # v = torch.fft.irfft(V, 1, onesided=False)
        v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, : N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, : N // 2]

        return x.view(*x_shape)

    def householder(self, x, eps=1e-6):
        if self.householder_learned:
            v = self.v / (torch.norm(self.v) + eps)
        else:
            v = self.v
        # 调整为相同形状
        for i in range(len(x.shape) - 1):
            v = v.unsqueeze(0)
        # b, n, e; 1, 1, e -> 1, n, 1
        y = torch.einsum("...ne,...le->...nl", x, v)
        # 1, n, 1; 1, 1, e -> 1, n, e
        y = torch.einsum("...nl,...le->...ne", y, v)

        return x - 2 * y
