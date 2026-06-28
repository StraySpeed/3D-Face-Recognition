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
      2. 그룹 centroid = mean over k         (linear)
      3. relative feature = x - centroid     (linear)
      4. (옵션) ||x - centroid||² 채널 추가   (squared dist, sqrt 없음)
      5. shared MLP: Conv2d(1×1) → HerPN → Conv2d(1×1)
      6. sum pool over k                     (linear)
      7. BN1d (추론 시 affine으로 고정)
      8. (옵션) channel shuffle              (rotation만으로 가능)

    Multiplicative depth ≈ 3 / block:
      - Conv1: depth 1
      - HerPN: depth 1 (BN affine 고정 후 x²)
      - Conv2: depth 1
    """
    def __init__(self, in_ch, hid_ch, out_ch, k=4,
                 use_squared_dist=False, target_groups=4):
        super().__init__()
        self.k = k
        self.use_squared_dist = use_squared_dist

        # 입력 채널: [x_g (C), x_g - c (C), 옵션 ||x_g - c||² (1)]
        input_ch = in_ch * 2 + (1 if use_squared_dist else 0)

        # 두 번째 conv는 grouped로 만들어 HE에서 채널 통신을 줄일 수 있음
        # (입력 채널 수가 target_groups로 나누어떨어질 때만 사용)
        if hid_ch % target_groups == 0 and out_ch % target_groups == 0:
            self.groups = target_groups
        else:
            self.groups = 1

        self.conv1 = nn.Conv2d(input_ch, hid_ch, 1, bias=False)
        self.act1  = HerPNActivation(hid_ch, ndim=2)
        self.conv2 = nn.Conv2d(hid_ch, out_ch, 1, groups=self.groups, bias=False)
        self.bn    = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        """
        :param x: (B, C_in, N)  Z-order 정렬된 시퀀스
        :return:  (B, C_out, N/k)
        """
        B, C, N = x.shape
        if N % self.k != 0:
            raise ValueError(f"SABlock: N({N}) % k({self.k}) != 0")
        N_out = N // self.k

        # 1. 그룹화 (단순 reshape — k개 연속 점이 한 region)
        x_g = x.view(B, C, N_out, self.k)

        # 2. centroid
        c = x_g.mean(dim=-1, keepdim=True)             # (B, C, N_out, 1)

        # 3. relative
        x_rel = x_g - c                                # (B, C, N_out, k)

        # 4. feature concat
        feats = [x_g, x_rel]
        if self.use_squared_dist:
            sq_dist = (x_rel * x_rel).sum(dim=1, keepdim=True)  # (B, 1, N_out, k)
            feats.append(sq_dist)
        feat = torch.cat(feats, dim=1)                 # (B, input_ch, N_out, k)

        # 5. shared MLP
        feat = self.conv1(feat)                        # (B, hid, N_out, k)
        feat = self.act1(feat)
        feat = self.conv2(feat)                        # (B, out, N_out, k)

        # 6. sum pool over k (HE-friendly, max 회피)
        feat = feat.sum(dim=-1)                        # (B, out, N_out)

        # 7. BN1d (추론 시 고정 affine)
        feat = self.bn(feat)

        # 8. 채널 셔플 (grouped conv 사용 시 채널 간 통신 복원)
        feat = channel_shuffle1d(feat, self.groups)
        return feat
