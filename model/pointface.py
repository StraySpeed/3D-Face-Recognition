"""
PointFaceNet (model2)

설계:
  - 4 SABlock + Multi-scale Aggregation
  - squared distance 입력 제거: 모든 블록이 [x_g, x_rel] 만 사용
  - SABlock 내부 구조:
      Conv2d(in_ch*2 → hid_ch) → HerPN → Conv2d(hid_ch → out_ch) → sum_pool → BN
  - 채널 수 (Config C: SA1 2×, SA4 병목 유지):
      SA1: 128ch — 256pts  → global sum pool → (B, 128)
      SA2: 256ch —  64pts  → global sum pool → (B, 256)
      SA3: 512ch —  16pts  → global sum pool → (B, 512)
      SA4: 512ch —   4pts  → global sum pool → (B, 512)
  - Multi-scale concat: (B, 128+256+512+512) = (B, 1408)
  - FC: 1408 → feature_dim (512)
  - FHE 최대 채널: 512 (SA4 병목으로 채널 폭발 방지)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules.sa_block import SABlock


# (in_ch, hid_ch, out_ch, k)
# Config C: SA1 채널 2배, SA4 병목(hid=256)으로 FHE 최대 채널 512 고정
SA_BLOCKS = [
    (3,    64,  128,  4),   # SA1: concat  6→ 64→128  (1024→256pts)
    (128, 128,  256,  4),   # SA2: concat 256→128→256  (256→ 64pts)
    (256, 256,  512,  4),   # SA3: concat 512→256→512  ( 64→ 16pts)
    (512, 256,  512,  4),   # SA4: concat1024→256→512  (  16→  4pts, 병목)
]

DEFAULT_FEATURE_DIM = 512
DEFAULT_NUM_POINTS  = 1024


class PointFaceEncoder(nn.Module):
    """
    Z-order 정렬된 (B, 3, 1024) 입력 → feature_dim 임베딩.

    forward:
      SA1 → SA2 → SA3 → SA4
       ↓     ↓     ↓     ↓    (각 블록 출력을 global sum pool)
      pool  pool  pool  pool
       └─────┴─────┴─────┘ concat(1408ch) → FC → BN
    """
    def __init__(self, feature_dim=DEFAULT_FEATURE_DIM):
        super().__init__()
        self.sa_blocks = nn.ModuleList([
            SABlock(in_ch, hid_ch, out_ch, k=k)
            for (in_ch, hid_ch, out_ch, k) in SA_BLOCKS
        ])
        # multi-scale concat 차원: 각 SA block의 out_ch 합산
        ms_dim = sum(cfg[2] for cfg in SA_BLOCKS)  # 128+256+512+512 = 1408
        self.fc = nn.Sequential(
            nn.Linear(ms_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, points):
        x = points
        scale_feats = []
        for block in self.sa_blocks:
            x = block(x)
            scale_feats.append(x.sum(dim=2))  # (B, out_ch) — global sum pool
        # multi-scale concat: (B, 64+128+256) = (B, 448)
        x = torch.cat(scale_feats, dim=1)
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