"""
PointFaceNet

설계:
  - 4 SABlock + Global Sum Pool (채널만 조정)
  - 채널 수:
      SA1:  64ch (model2: 128) — 64 × 256pt = 16384  ✓ 1 ct 정확히 채움
      SA2: 128ch (model2: 256) — 128 × 64pt =  8192
      SA3: 256ch (model2: 512) — 256 × 16pt =  4096
      SA4: 512ch (model2: 1024) — 512 × 4pt =  2048
      Global sum pool → 512
      FC: 512 → 256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules.sa_block import SABlock


# (in_ch, hid_ch, out_ch, k, use_squared_dist)
SA_BLOCKS = [
    (3,    32,  64,   4, True),    # SA1: 1024 → 256, 채널 3 → 64
    (64,   64,  128,  4, False),   # SA2:  256 →  64, 채널 64 → 128
    (128,  128, 256,  4, False),   # SA3:   64 →  16, 채널 128 → 256
    (256,  256, 512,  4, False),   # SA4:   16 →   4, 채널 256 → 512
]

DEFAULT_FEATURE_DIM = 256
DEFAULT_NUM_POINTS  = 1024


class PointFaceEncoder(nn.Module):
    """
    Z-order 정렬된 (B, 3, 1024) 입력 → feature_dim 임베딩.

    forward:
      SA1 → SA2 → SA3 → SA4 → global_sum_pool → FC → BN
    """
    def __init__(self, feature_dim=DEFAULT_FEATURE_DIM):
        super().__init__()
        self.sa_blocks = nn.ModuleList([
            SABlock(in_ch, hid_ch, out_ch, k=k, use_squared_dist=use_sq)
            for (in_ch, hid_ch, out_ch, k, use_sq) in SA_BLOCKS
        ])
        last_out_ch = SA_BLOCKS[-1][2]
        self.fc = nn.Sequential(
            nn.Linear(last_out_ch, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, points):
        x = points
        for block in self.sa_blocks:
            x = block(x)
        # SA4 출력 (B, C, 4) → global sum pool → (B, C)
        x = x.sum(dim=2)
        x = self.fc(x)
        return x


class PointFaceNet(nn.Module):
    """학습용 래퍼: forward는 L2 정규화 임베딩, encode는 정규화 전."""
    def __init__(self, feature_dim=DEFAULT_FEATURE_DIM):
        super().__init__()
        self.encoder = PointFaceEncoder(feature_dim=feature_dim)

    def encode(self, points):
        return self.encoder(points)

    def forward(self, points):
        emb = self.encoder(points)
        return F.normalize(emb, p=2, dim=1)

    @staticmethod
    def morton_sort(points_np):
        """ 
        3D 공간의 점(N, 3)들을 Z-Order 커브를 따라 정렬 (Numpy 기반)
        """
        coords = points_np[:, :3]
        p_min = np.min(coords, axis=0)
        p_max = np.max(coords, axis=0)
        
        # 0~1로 정규화 후 10bit(0~1023) 양자화
        norm_points = (coords - p_min) / (p_max - p_min + 1e-8)
        quantized = np.clip(np.floor(norm_points * 1024), 0, 1023).astype(np.uint32)
        
        def part1by2(n):
            n &= 0x000003ff
            n = (n ^ (n << 16)) & 0xff0000ff
            n = (n ^ (n <<  8)) & 0x0300f00f
            n = (n ^ (n <<  4)) & 0x030c30c3
            n = (n ^ (n <<  2)) & 0x09249249
            return n
        
        x = np.vectorize(part1by2)(quantized[:, 0])
        y = np.vectorize(part1by2)(quantized[:, 1])
        z = np.vectorize(part1by2)(quantized[:, 2])
        
        codes = (z << 2) | (y << 1) | x
        return points_np[np.argsort(codes)]

    @staticmethod
    def preprocess(points, num_points=1024, device='cpu'):
        """
        추론(Inference) 및 클라이언트 전용 전처리 로직.
        (N, 3) numpy 배열을 입력받아 모델이 기대하는 (1, 3, num_points) 텐서로 변환합니다.
        """
        # 1. Deterministic Sampling (항상 일정한 간격으로 점을 샘플링하여 Double Sort 버그 방지)
        total = len(points)
        if total > num_points:
            indices = np.linspace(0, total - 1, num_points, dtype=int)
            sampled_points = points[indices, :]
        else:
            rng = np.random.RandomState(1) # 모자란 경우 고정된 시드로 복원 추출
            choice = rng.choice(total, num_points, replace=True)
            sampled_points = points[choice, :]

        # 2. Normalization (Unit Sphere: 중심점 빼고 최대 거리로 나누기)
        centroid = np.mean(sampled_points, axis=0)
        sampled_points = sampled_points - centroid
        m = np.max(np.sqrt(np.sum(sampled_points ** 2, axis=1)))
        normalized_points = sampled_points / (m + 1e-8)

        # 3. Morton Sort (Z-order 커브 정렬)
        sorted_points = PointFaceNet.morton_sort(normalized_points)

        # 4. Convert to PyTorch Tensor: (N, 3) -> (3, N) -> 배치 차원 추가 (1, 3, N)
        tensor = torch.from_numpy(sorted_points.astype(np.float32)).t().contiguous()
        tensor = tensor.unsqueeze(0).to(device)
        
        return tensor