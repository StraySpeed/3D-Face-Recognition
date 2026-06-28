import math
import torch
import torch.nn as nn


class HerPNActivation(nn.Module):
    """
    AESPA HerPN: Hermite Polynomial with Basis-wise Normalization
    (Park et al., 2022 — https://arxiv.org/abs/2201.06699)

    ReLU의 Hermite 전개 계수 (고정):
        h_1(x) = x,             f̂_1 = 1/2
        h_2(x) = (x²-1)/√2,    f̂_2 = 1/(2√π)

    f(x) = γ · [f̂_1·BN_1(h_1(x)) + f̂_2·BN_2(h_2(x))] + β

    CKKS 호환: 결과는 a·x² + b·x + c 형태의 degree-2 다항식으로 전개 가능.
    """
    def __init__(self, num_channels, ndim=1, eps=1e-3):
        super().__init__()
        self.register_buffer('f1', torch.tensor(0.5, dtype=torch.float32))
        self.register_buffer('f2', torch.tensor(1.0 / (2.0 * math.sqrt(math.pi)), dtype=torch.float32))

        BNClass = nn.BatchNorm1d if ndim == 1 else nn.BatchNorm2d
        self.bn1 = BNClass(num_channels, eps=eps, affine=False)
        self.bn2 = BNClass(num_channels, eps=eps, affine=False)

        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta  = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        h1 = x
        h2 = (x.pow(2) - 1.0) * (1.0 / math.sqrt(2.0))
        out = self.f1 * self.bn1(h1) + self.f2 * self.bn2(h2)
        shape = [1, -1] + [1] * (x.dim() - 2)
        return out * self.gamma.view(shape) + self.beta.view(shape)


def channel_shuffle1d(x, groups):
    """Conv1d/Linear 계층용 채널 셔플. groups=1이면 no-op."""
    if groups == 1:
        return x
    B, C, L = x.shape
    x = x.view(B, groups, C // groups, L)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(B, -1, L)
