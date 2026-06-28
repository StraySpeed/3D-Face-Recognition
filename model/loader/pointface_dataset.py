"""
PointFace dataset loader.

기존 대비 단순화:
  - FPS/Ball Query 사전 계산 제거 → pre_data dict 없음
  - 단일 출력: (3, N) Z-order 정렬된 좌표 텐서
  - 모든 SA 단계가 동일한 stride(k=4)이므로 데이터 흐름 일관

배치 단위 반환: (anchor_points, pos_points, label)
  anchor_points / pos_points: (3, N) float32 텐서 (Morton 정렬 완료)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .pointcloud_augmentation import PointCloudAugmentation
from ..modules.serialization import morton_sort_tensor


class PointFaceDataset(Dataset):
    def __init__(self, data_root, train=True, num_points=1024):
        """
        :param data_root: 데이터 폴더 (구조: root/ID/sample.npy)
        :param train: 학습 모드 (True면 augmentation 적용)
        :param num_points: 모델 입력 점 개수
        """
        self.data_root = data_root
        self.train     = train
        self.num_points = num_points
        self.transform = PointCloudAugmentation(num_points=num_points)
        self.pairs     = []

        self.subjects = self._load_data(data_root)
        self._generate_pairs()
        print(f"[model] Dataset Loaded: {len(self.subjects)} subjects, {len(self.pairs)} pairs.")

    def _load_data(self, root):
        subjects = {}
        if not os.path.exists(root):
            return subjects
        label_idx = 0
        for folder_name in sorted(os.listdir(root)):
            folder_path = os.path.join(root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
            if files:
                subjects[label_idx] = sorted(files)
                label_idx += 1
        return subjects

    def _generate_pairs(self):
        """neutral 표정을 anchor로 고정, 나머지 표정과 양방향 쌍 생성."""
        self.pairs = []
        for label_idx, file_list in self.subjects.items():
            n = len(file_list)
            if n < 1:
                continue
            if n == 1:
                self.pairs.append((file_list[0], file_list[0], label_idx))
                continue
            neutral = next(
                (f for f in file_list if "neutral" in os.path.basename(f).lower()),
                file_list[0]
            )
            others = [f for f in file_list if f != neutral]
            for pos in others:
                self.pairs.append((neutral, pos, label_idx))
                self.pairs.append((pos, neutral, label_idx))

    def _deterministic_sample_and_normalize(self, points):
        """추론(Test) 시 결정론적 샘플링 + unit sphere 정규화."""
        total = len(points)
        if total > self.num_points:
            indices = np.linspace(0, total - 1, self.num_points, dtype=int)
            points = points[indices, :]
        else:
            rng = np.random.RandomState(1)  # 결정론적 시드
            choice = rng.choice(total, self.num_points, replace=True)
            points = points[choice, :]
        centroid = np.mean(points, axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        return points / m

    def _preprocess(self, points_np):
        """
        Raw (N, 3) numpy → augment/normalize → Morton 정렬 → (3, num_points) 텐서.
        """
        if self.train:
            # augmentation은 (3, N) 텐서를 반환
            tensor = self.transform(points_np)
        else:
            arr = self._deterministic_sample_and_normalize(points_np)
            tensor = torch.from_numpy(arr.astype(np.float32)).t().contiguous()

        # Z-order 정렬 (클라이언트 측 전처리에 해당)
        tensor = morton_sort_tensor(tensor)
        return tensor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor_path, pos_path, label = self.pairs[idx]
        anchor_np = np.load(anchor_path)
        pos_np    = np.load(pos_path)
        anchor_t  = self._preprocess(anchor_np)
        pos_t     = self._preprocess(pos_np)
        return anchor_t, pos_t, label


def get_dataloader(data_root, batch_size=32, num_workers=4, train=True, num_points=1024):
    dataset = PointFaceDataset(data_root=data_root, train=train, num_points=num_points)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )
