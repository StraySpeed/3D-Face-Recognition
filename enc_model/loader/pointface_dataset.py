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
        self.transform = PointCloudAugmentation(num_points=5000)
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

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_anchor, path_positive, label = self.pairs[idx]
        
        # 1. 파일 로드 (여기서는 .npy)
        pc_anchor = np.load(path_anchor)[:, :3] # (N, 3) XYZ 좌표만 사용
        pc_positive = np.load(path_positive)[:, :3]
        
        # 2. Augmentation & Preprocessing
        # Anchor와 Positive에 대해 서로 다른 랜덤 증강 적용
        tensor_anchor = self.transform(pc_anchor, train=self.train)
        tensor_positive = self.transform(pc_positive, train=self.train)

        # y축 기준 정렬
        tensor_anchor = self._sort_by_y_axis(tensor_anchor)
        tensor_positive = self._sort_by_y_axis(tensor_positive)

        # 3. 각 텐서별 중심점 미리 계산
        pre_data_anchor = self.get_precomputed_data(tensor_anchor)
        pre_data_positive = self.get_precomputed_data(tensor_positive)

        return tensor_anchor, pre_data_anchor, tensor_positive, pre_data_positive, label
    
    def get_precomputed_data(self, tensor_3xn):
        """
        단일 샘플(3, 5000)에 대해 4 -> 2 -> 1 병합 단계별 중심점, idx를 미리 계산
        """
        # (3, 5000) -> (1, 5000, 3) 형태로 변환 (FPS 함수 호환을 위함)
        xyz = tensor_3xn.unsqueeze(0).permute(0, 2, 1) 
        # SA 파라미터 가져오기
        params = CONFIG['SA_PARAMS']

        sa1_n = params[0]['npoint']
        sa2_n = params[1]['npoint']
        sa3_n = params[2]['npoint']
        sa4_n = params[3]['npoint']

        pre_data = {'centroids': {}, 'indices': {}}
        
        with torch.no_grad():
            # Y축 기준 정렬
            sorted_idx = torch.argsort(xyz[:, :, 1], dim=1)
            xyz = torch.gather(xyz, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, 3))

            # [Stage 1] 4등분 (각 1250개)
            xyz_chunks = torch.chunk(xyz, 4, dim=1)
            xyz_b1 = torch.cat(xyz_chunks, dim=0) # (4, 1250, 3)
            idx1 = farthest_point_sample(xyz_b1, sa1_n)
            c1 = index_points(xyz_b1, idx1)
            ball_idx1 = query_ball_point(params[0]['radius'], params[0]['nsample'], xyz_b1, c1)

            pre_data['centroids']['s1'] = c1.permute(0, 2, 1) # (4, 3, 512)
            pre_data['indices']['s1'] = ball_idx1
            
            # [Stage 2] 2개씩 병합
            c1_chunks = torch.chunk(c1, 4, dim=0)
            xyz_12 = torch.cat([c1_chunks[0], c1_chunks[1]], dim=1) # (1, 1024, 3)
            xyz_34 = torch.cat([c1_chunks[2], c1_chunks[3]], dim=1) # (1, 1024, 3)
            xyz_b2 = torch.cat([xyz_12, xyz_34], dim=0) # (2, 1024, 3)
            
            idx2 = farthest_point_sample(xyz_b2, sa2_n)
            c2 = index_points(xyz_b2, idx2)
            ball_idx2 = query_ball_point(params[1]['radius'], params[1]['nsample'], xyz_b2, c2)

            pre_data['centroids']['s2'] = c2.permute(0, 2, 1) # (2, 3, 512)
            pre_data['indices']['s2'] = ball_idx2
            
            # [Stage 3] 1개로 병합
            c2_chunks = torch.chunk(c2, 2, dim=0)
            xyz_b3 = torch.cat([c2_chunks[0], c2_chunks[1]], dim=1) # (1, 1024, 3)
            
            idx3 = farthest_point_sample(xyz_b3, sa3_n)
            c3 = index_points(xyz_b3, idx3)
            ball_idx3 = query_ball_point(params[2]['radius'], params[2]['nsample'], xyz_b3, c3)

            pre_data['centroids']['s3'] = c3.permute(0, 2, 1) # (1, 3, 512)
            pre_data['indices']['s3'] = ball_idx3
            
            # [Stage 4] 최종 
            idx4 = farthest_point_sample(c3, sa4_n)
            c4 = index_points(c3, idx4)
            ball_idx4 = query_ball_point(params[3]['radius'], params[3]['nsample'], c3, c4)

            pre_data['centroids']['s4'] = c4.permute(0, 2, 1) # (1, 3, 256)
            pre_data['indices']['s4'] = ball_idx4

        return pre_data
    
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