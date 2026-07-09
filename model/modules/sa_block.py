import torch
import torch.nn as nn

from .activations import HerPNActivation, channel_shuffle1d


class SABlock(nn.Module):
    """
    HE-friendly Set Abstraction block.

    가정: 입력은 Z-order(Morton)로 사전 정렬되어 있음.
    그룹화: k개의 연속된 점을 한 region으로 묶음 (stride=k).

    파이프라인 (모두 HE 가능 연산):
      1. (B, C, N) → reshape → (B, C, N/k, k)
      2. 그룹 centroid = mean over k                     (linear)
      3. relative feature = (x - centroid) / sqrt(k)    (linear, k-normalized)
      4. [x_g | x_rel] concat                           (in_ch*2 채널)
      5. shared MLP: Conv2d(1×1) → HerPN → Dropout2d → Conv2d(1×1)
      6. sum pool over k                                 (linear)
      7. BN1d
      8. channel shuffle (grouped conv 사용 시)

    x_rel을 sqrt(k)로 나누는 이유:
      Morton 그룹이 공간적으로 커질수록 (k가 클수록) x_rel 크기도 비례해 커짐.
      sqrt(k) 나눗셈은 k에 무관한 일정한 상대좌표 스케일을 보장하므로,
      SA0(k=16)와 SA1(k=4)의 x_rel이 서로 비슷한 크기를 가짐.

    Multiplicative depth ≈ 3 / block:
      - Conv1: depth 1
      - HerPN: depth 1
      - Conv2: depth 1
    """
    def __init__(self, in_ch, hid_ch, out_ch, k=4, dropout=0.0, target_groups=4):
        super().__init__()
        self.k = k
        self.inv_sqrt_k = k ** -0.5

        input_ch = in_ch * 2

        if hid_ch % target_groups == 0 and out_ch % target_groups == 0:
            self.groups = target_groups
        else:
            self.groups = 1

        self.conv1 = nn.Conv2d(input_ch, hid_ch, 1, bias=False)
        self.act1  = HerPNActivation(hid_ch, ndim=2)
        self.drop  = nn.Dropout2d(p=dropout) if dropout > 0.0 else None
        self.conv2 = nn.Conv2d(hid_ch, out_ch, 1, groups=self.groups, bias=False)
        self.bn    = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        """
        :param x: (B, in_ch, N)  Z-order 정렬된 시퀀스
        :return:  (B, out_ch, N/k)
        """
        B, C, N = x.shape
        if N % self.k != 0:
            raise ValueError(f"SABlock: N({N}) % k({self.k}) != 0")
        N_out = N // self.k

        x_g   = x.view(B, C, N_out, self.k)
        c     = x_g.mean(dim=-1, keepdim=True)
        x_rel = (x_g - c) * self.inv_sqrt_k   # k-normalized relative coords

        feat  = torch.cat([x_g, x_rel], dim=1)  # (B, in_ch*2, N_out, k)
        feat  = self.conv1(feat)                 # (B, hid_ch, N_out, k)
        feat  = self.act1(feat)
        if self.drop is not None:
            feat = self.drop(feat)
        feat  = self.conv2(feat)                 # (B, out_ch, N_out, k)
        feat  = feat.sum(dim=-1)                 # (B, out_ch, N_out)
        feat  = self.bn(feat)
        feat  = channel_shuffle1d(feat, self.groups)
        return feat
