from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
from .pointcloud_augmentation import PointCloudAugmentation
try:
    from ..modules.utils.utils import farthest_point_sample, index_points, query_ball_point
except ImportError:
    from model.modules.utils.utils import farthest_point_sample, index_points, query_ball_point
from config import CONFIG

class PointFaceDataset(Dataset):
    def __init__(self, data_root, train=True):
        """
        # 데이터셋 로더

        :param data_root: 데이터 폴더 (구조: root/ID/sample.npy)
        """
        self.data_root = data_root
        self.train = train
        self.transform = PointCloudAugmentation(num_points=CONFIG["MODEL"]["num_points"])
        self.pairs = [] # (path_anchor, path_positive, label_idx)
        
        # 1. 데이터 로드 및 정리
        # subjects: {label_idx: [path1, path2, ...]}
        self.subjects = self._load_data(data_root)
        
        # 2. Pair Generation
        self._generate_pairs()
        
        print(f"Dataset Loaded: {len(self.subjects)} subjects, {len(self.pairs)} pairs generated.")

    def _load_data(self, root):
        # 폴더 구조를 읽어서 ID별로 파일 경로를 저장
        subjects = {}
        # 폴더명이 ID (0, 1, 2...)
        if not os.path.exists(root):
            return {} # 빈 딕셔너리 반환

        for label_idx, folder_name in enumerate(sorted(os.listdir(root))):
            folder_path = os.path.join(root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
            if len(files) > 0:
                subjects[label_idx] = sorted(files)
        return subjects

    def _generate_pairs(self):
        """
        Pair Selection Strategy

        각 Subject에 대해 리스트를 순회하며 (i, i+1) 형태로 쌍을 생성
        """
        self.pairs = []
        for label_idx, file_list in self.subjects.items():
            n = len(file_list)
            if n < 1: continue
            
            # 데이터가 1개뿐인 경우 자기 자신을 Positive로 (예외 처리)
            if n == 1:
                self.pairs.append((file_list[0], file_list[0], label_idx))
                continue

            # 순환 연결 방식: (P_i, P_{(i) mod len + 1})
            for i in range(n):
                anchor = file_list[i]
                positive = file_list[(i + 1) % n] # 다음 인덱스 (마지막이면 처음으로)
                self.pairs.append((anchor, positive, label_idx))

    def _sort_by_y_axis(self, tensor_3xn):
        """
        텐서를 Y축(인덱스 1)을 기준으로 오름차순 정렬

        :param tensor_3xn: (3, N) 형태의 텐서
        """
        # (3, 5000) -> (5000, 3)으로 변환하여 정렬하기 쉽게 만듦
        xyz = tensor_3xn.transpose(0, 1) 
        
        # Y축(dim=1)을 기준으로 정렬된 인덱스 추출
        sorted_indices = torch.argsort(xyz[:, 1]) 
        
        # 추출한 인덱스를 이용해 원래 텐서의 순서를 재배치
        sorted_xyz = xyz[sorted_indices]
        
        # 다시 원래의 (3, 5000) 형태로 복구
        return sorted_xyz.transpose(0, 1)

    def _deterministic_sample_and_normalize(self, points, num_points=CONFIG["MODEL"]["num_points"]):
        """추론(Test) 시 점의 개수를 고정하고 크기를 정규화하는 함수"""
        total_points = len(points)
        if total_points > num_points:
            # 일정한 간격으로 균등하게 5000개 추출 (랜덤성 배제)
            indices = np.linspace(0, total_points - 1, num_points, dtype=int)
            points = points[indices, :]
        else:
            np.random.seed(1)
            choice = np.random.choice(total_points, num_points, replace=True)
            np.random.seed(None)
            points = points[choice, :]
            
        # 정규화 (Zero-centering & Unit Sphere Scaling)
        centroid = np.mean(points, axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / m
        
        return points

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor_path, pos_path, label = self.pairs[idx]
        
        # 1. 원본 데이터 로드 [N, 3]
        anchor_points = np.load(anchor_path)
        pos_points = np.load(pos_path)
        
        # 2. 전처리 및 증강 (Transform)
        if self.train and self.transform is not None:
            anchor_points = self.transform(anchor_points)
            pos_points = self.transform(pos_points)
        else:
            anchor_points = self._deterministic_sample_and_normalize(anchor_points, CONFIG["MODEL"]["num_points"])
            pos_points = self._deterministic_sample_and_normalize(pos_points, CONFIG["MODEL"]["num_points"])
        # 3. 사전 연산 데이터(pre_data) 생성
        anchor_pre_data = self._generate_pre_data(anchor_points)
        pos_pre_data = self._generate_pre_data(pos_points)
        
        # 반환: (Anchor의 사전 데이터, Positive의 사전 데이터, 라벨)
        return anchor_pre_data, pos_pre_data, label

    def _generate_pre_data(self, points):
        """
        Numpy 포인트 클라우드를 입력받아 FPS, Ball Query를 수행하고
        최종적으로 4단계(Stage 1~4)의 relation_vector와 indices를 딕셔너리로 묶어 반환
        """
        # Numpy (N, 3) -> Tensor (3, N) 로 변환
        if isinstance(points, torch.Tensor):
            curr_xyz = points.detach().clone().float()
        else:
            curr_xyz = torch.tensor(points, dtype=torch.float32)

        # 두 번째 차원이 3이면 (N, 3) 형태이므로 전치(transpose)하여 (3, N)으로 만듦
        if curr_xyz.shape[1] == 3:
            curr_xyz = curr_xyz.t()

        # Y축 기준 오름차순 정렬 (모델의 결정론적 특성을 위해 적용)
        curr_xyz = self._sort_by_y_axis(curr_xyz)
        
        pre_data = {'rel': {}, 'idx': {}}
        sa_params = CONFIG['SA_PARAMS'] # config.py에서 파라미터 로드
        
        for i, params in enumerate(sa_params):
            npoint = params['npoint']
            radius = params['radius']
            nsample = params['nsample']
            
            # 유틸 함수들은 배치(Batch) 차원을 요구하므로 (1, 3, N) 형태로 차원 추가
            curr_xyz_b = curr_xyz.unsqueeze(0) 
            
            # BxNx3 형태로 변환 (FPS 및 Ball Query용)
            xyz_trans = curr_xyz_b.transpose(1, 2).contiguous()
            
            # a. Farthest Point Sample -> (1, npoint)
            fps_idx = farthest_point_sample(xyz_trans, npoint)
            
            # b. 중심점 추출 -> (1, 3, npoint)
            centroids_b = index_points(xyz_trans, fps_idx).transpose(1, 2).contiguous()
            
            # c. Ball Query (이웃점 인덱스 찾기) -> (1, npoint, nsample)
            idx_b = query_ball_point(radius, nsample, xyz_trans, centroids_b.transpose(1, 2).contiguous())
            
            # 배치 차원(1) 제거 -> (3, npoint) 및 (npoint, nsample)
            centroids = centroids_b.squeeze(0)
            idx = idx_b.squeeze(0).long()
            
            # d. 10차원 Relation Vector 조립 (이전에 구현하신 _compute_relation_vector 활용)
            rel_vec = self._compute_relation_vector(curr_xyz, centroids, idx)
            
            # e. 딕셔너리에 저장
            stage_key = f's{i+1}'
            pre_data['rel'][stage_key] = rel_vec
            pre_data['idx'][stage_key] = idx
            
            # f. 다음 단계를 위해 현재의 중심점을 새로운 입력 점들로 갱신
            curr_xyz = centroids
            
        return pre_data

    def _compute_relation_vector(self, xyz, centroids, idx):
        """
        주어진 점(xyz)과 중심점(centroids), 인덱스(idx)를 이용해 
        10차원 relation_vector를 조립
        :param xyz: (3, N) - 현재 단계의 점들
        :param centroids: (3, npoint) - 샘플링된 중심점들
        :param idx: (npoint, nsample) - 이웃점 인덱스
        """
        C, N = xyz.shape
        _, npoint = centroids.shape
        nsample = idx.shape[1]
        
        idx_safe = idx.clone()
        idx_safe[idx_safe >= N] = 0

        # 1. Grouped XYZ (이웃점들 가져오기) -> (3, npoint, nsample)
        grouped_xyz = xyz[:, idx.flatten()].view(C, npoint, nsample)
        
        # 2. Expanded Centroids (중심점 복사) -> (3, npoint, nsample)
        center_xyz = centroids.unsqueeze(-1).expand(C, npoint, nsample)
        
        # 3. Difference (좌표 차이) -> (3, npoint, nsample)
        diff = center_xyz - grouped_xyz
        
        # 4. Squared Euclidean (유클리드 거리 제곱) -> (1, npoint, nsample)
        sq_dist = torch.sum(diff ** 2, dim=0, keepdim=True)
        
        # 5. Concat (10차원 벡터 조립) -> (10, npoint, nsample)
        relation_vector = torch.cat([center_xyz, grouped_xyz, diff, sq_dist], dim=0)
        
        return relation_vector

    
def get_dataloader(data_root, batch_size=32, num_workers=4, train=True):
    # Dataset 인스턴스 생성
    dataset = PointFaceDataset(data_root=data_root, train=train)
    
    # DataLoader 생성
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, # GPU 전송 속도 향상
        drop_last=True   # 배치 크기가 일정해야 Loss 계산이 용이
    )
    
    return loader