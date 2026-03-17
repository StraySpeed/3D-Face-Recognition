try:
    from modules.utils.utils import farthest_point_sample, query_ball_point, index_points
    from pointface_encoder import PointFaceEncoder
except ImportError:
    from .modules.utils.utils import farthest_point_sample, query_ball_point, index_points
    from .pointface_encoder import PointFaceEncoder

import torch.nn as nn
import torch.nn.functional as F
from config import CONFIG
import torch
import numpy as np

class PointFaceNet(nn.Module):
    def __init__(self, num_classes):
        super(PointFaceNet, self).__init__()
        self.encoder = PointFaceEncoder()
        
        # 학습 시 Identity Classification을 위한 Softmax Layer
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, pre_data=None):
        # 1. 인코더를 통해 임베딩 추출
        embedding = self.encoder(pre_data)
        
        # 2. L2 정규화
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        
        # 3. 학습용 Logits 계산
        if self.training:
            logits = self.classifier(norm_embedding)
            return norm_embedding, logits
        else:
            return norm_embedding
        
    @staticmethod
    def preprocess(points, num_points=CONFIG["MODEL"]["num_points"], device='cpu'):
        """
        점들을 전처리하는 함수
        """
        total_points = len(points)
        if total_points > num_points:
            # 균등 샘플링 (랜덤성 배제)
            indices = np.linspace(0, total_points - 1, num_points, dtype=int)
            points = points[indices, :]
        else:
            # 부족할 경우 고정 시드로 복원 추출
            np.random.seed(1)
            choice = np.random.choice(total_points, num_points, replace=True)
            np.random.seed(None)
            points = points[choice, :]
            
        # Zero-centering & Unit Sphere 정규화
        centroid = np.mean(points, axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / m

        # 텐서 변환 및 Y축 정렬
        curr_xyz = torch.tensor(points, dtype=torch.float32).to(device)
        
        # 항상 (3, N) 형태로 보장
        if curr_xyz.shape[1] == 3:
            curr_xyz = curr_xyz.t()
            
        # Y축(인덱스 1) 기준 오름차순 정렬
        sorted_indices = torch.argsort(curr_xyz[1, :])
        curr_xyz = curr_xyz[:, sorted_indices]

        # 모델 입력용 pre_data 조립 (Stage 1~4)
        pre_data = {'rel': {}, 'idx': {}}
        sa_params = CONFIG['SA_PARAMS']
        
        for i, params in enumerate(sa_params):
            npoint = params['npoint']
            radius = params['radius']
            nsample = params['nsample']
            
            # 유틸 함수용 배치 차원 추가 -> (1, 3, N)
            curr_xyz_b = curr_xyz.unsqueeze(0) 
            xyz_trans = curr_xyz_b.transpose(1, 2).contiguous() # (1, N, 3)
            
            # FPS & Ball Query
            fps_idx = farthest_point_sample(xyz_trans, npoint)
            centroids_b = index_points(xyz_trans, fps_idx).transpose(1, 2).contiguous()
            idx_b = query_ball_point(radius, nsample, xyz_trans, centroids_b.transpose(1, 2).contiguous())
            
            # 차원 축소
            centroids = centroids_b.squeeze(0) # (3, npoint)
            idx = idx_b.squeeze(0).long()      # (npoint, nsample)
            
            # --- 10차원 Relation Vector 계산 ---
            C, N = curr_xyz.shape
            idx_safe = idx.clone()
            idx_safe[idx_safe >= N] = 0 # Dummy index 방어
            
            grouped_xyz = curr_xyz[:, idx_safe.flatten()].view(C, npoint, nsample)
            center_xyz = centroids.unsqueeze(-1).expand(C, npoint, nsample)
            diff = center_xyz - grouped_xyz
            sq_dist = torch.sum(diff ** 2, dim=0, keepdim=True)
            
            rel_vec = torch.cat([center_xyz, grouped_xyz, diff, sq_dist], dim=0) # (10, npoint, nsample)
            # ---------------------------------------------------
            
            stage_key = f's{i+1}'
            
            # DataLoader가 해주던 배치 묶기(Batching) 작업을 수동으로 처리
            # 모델은 항상 맨 앞에 Batch 차원(B)을 요구하므로 .unsqueeze(0)
            pre_data['rel'][stage_key] = rel_vec.unsqueeze(0) # -> (1, 10, npoint, nsample)
            pre_data['idx'][stage_key] = idx.unsqueeze(0)     # -> (1, npoint, nsample)
            
            # 다음 단계를 위해 갱신
            curr_xyz = centroids
            
        return pre_data