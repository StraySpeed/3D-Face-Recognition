"""
PointFace v8.2

설계 변경점 (v8.1 대비):
  - SA0 hid 64→128, out 128→256: fine-scale 표현 용량 확대 (0.2% → 0.8%)
  - SA1 in_ch 128→256: SA0 출력과 정합
  - Global pooling: sum → max (scale 균형 8:1→2.6:1, discriminability 향상)

포인트 수 흐름 (k=16, k=4, k=4, k=4):
    SA0:  4096 →  256pts,  256ch   k=16 (fine-scale, 공간 수용장 ~0.078)
    SA1:   256 →   64pts,  256ch   k=4
    SA2:    64 →   16pts,  512ch   k=4  + Dropout2d(0.1)
    SA3:    16 →    4pts,  512ch   k=4  + Dropout2d(0.1)

Multi-scale concat: (B, 256+256+512+512) = (B, 1536)
FC: 1536 → 512   (Dropout(0.1) → Linear → BN)
파라미터: ~1.34M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules.sa_block import SABlock


# (in_ch, hid_ch, out_ch, k, dropout)
SA_BLOCKS = [
    (3,   128,  256, 16, 0.0),   # SA0: 4096→ 256pts  k=16  hid/out 확대
    (256, 128,  256,  4, 0.0),   # SA1:  256→  64pts  k=4   in_ch SA0 맞춤
    (256, 256,  512,  4, 0.1),   # SA2:   64→  16pts  k=4   dropout
    (512, 256,  512,  4, 0.1),   # SA3:   16→   4pts  k=4   dropout
]

DEFAULT_FEATURE_DIM = 512
DEFAULT_NUM_POINTS  = 4096


class PointFaceEncoder(nn.Module):
    """
    Z-order 정렬된 (B, 3, 4096) 입력 → feature_dim 임베딩.

    forward:
      SA0 → SA1 → SA2 → SA3
       ↓     ↓     ↓     ↓    (각 블록 출력을 global sum pool)
      pool  pool  pool  pool
       └─────┴─────┴─────┘ concat(1408ch) → Dropout → FC → BN
    """
    def __init__(self, feature_dim=DEFAULT_FEATURE_DIM):
        super().__init__()
        self.sa_blocks = nn.ModuleList([
            SABlock(in_ch, hid_ch, out_ch, k=k, dropout=drop)
            for (in_ch, hid_ch, out_ch, k, drop) in SA_BLOCKS
        ])
        ms_dim = sum(cfg[2] for cfg in SA_BLOCKS)  # 256+256+512+512 = 1536
        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(ms_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, points):
        x = points
        scale_feats = []
        for block in self.sa_blocks:
            x = block(x)
            scale_feats.append(x.max(dim=2)[0])  # (B, out_ch) — global max pool
        x = torch.cat(scale_feats, dim=1)  # (B, 1408)
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
    def preprocess(points, num_points=4096, device='cpu'):
        """
        추론(Inference) 및 클라이언트 전용 전처리 로직.
        (N, 3) numpy 배열을 입력받아 모델이 기대하는 (1, 3, num_points) 텐서로 변환합니다.
        """
        # 1. Deterministic Sampling
        total = len(points)
        if total > num_points:
            indices = np.linspace(0, total - 1, num_points, dtype=int)
            sampled_points = points[indices, :]
        else:
            rng = np.random.RandomState(1)
            choice = rng.choice(total, num_points, replace=True)
            sampled_points = points[choice, :]

        # 2. Normalization (Unit Sphere)
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
