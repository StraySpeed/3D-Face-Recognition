try:
    from modules.utils.utils import farthest_point_sample, query_ball_point, index_points
    from pointface_encoder import PointFaceEncoder
except ImportError:
    from .modules.utils.utils import farthest_point_sample, query_ball_point, index_points
    from .pointface_encoder import PointFaceEncoder

import torch.nn as nn
import torch.nn.functional as F
from config import CONFIG
import numpy as np
import torch

class PointFaceNet(nn.Module):
    def __init__(self, num_classes):
        super(PointFaceNet, self).__init__()
        self.encoder = PointFaceEncoder()
        
        # 학습 시 Identity Classification을 위한 Softmax Layer
        self.classifier = nn.Linear(CONFIG["MODEL"]["feature_dim"], num_classes)

    def forward(self, x, pre_data=None):
        # 1. 인코더를 통해 임베딩 추출
        embedding = self.encoder(x, pre_data)
        
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
        # 1. 리샘플링 (5000개)
        if len(points) > num_points:
            choice = np.linspace(0, len(points) - 1, num_points, dtype=int)
            points = points[choice, :]
        elif len(points) < num_points:
            np.random.seed(1)
            choice = np.random.choice(len(points), num_points, replace=True)
            np.random.seed(None)
            points = points[choice, :]
            
        # 2. 정규화 (Center & Scale)
        points = points - np.mean(points, axis=0)
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / dist

        # 3. Y축 기준 정렬
        sorted_idx = np.argsort(points[:, 1])
        points = points[sorted_idx]
        
        # 4. 텐서 변환 (1, 3, 5000) - 배치 차원 추가
        tensor = torch.from_numpy(points.astype(np.float32)).transpose(0, 1).unsqueeze(0).to(device)
        
        xyz = tensor.permute(0, 2, 1) # (1, 5000, 3)
        pre_data = {'centroids': {}, 'indices': {}}

        # SA 파라미터 가져오기
        params = CONFIG['SA_PARAMS']

        sa1_n = params[0]['npoint']
        sa2_n = params[1]['npoint']
        sa3_n = params[2]['npoint']
        sa4_n = params[3]['npoint']

        with torch.no_grad():
            # Stage 1
            xyz_chunks = torch.chunk(xyz, 4, dim=1)
            xyz_b1 = torch.cat(xyz_chunks, dim=0) # (4, 1250, 3)
            idx1 = farthest_point_sample(xyz_b1, sa1_n)
            c1 = index_points(xyz_b1, idx1)
            ball_idx1 = query_ball_point(params[0]['radius'], params[0]['nsample'], xyz_b1, c1)
            pre_data['centroids']['s1'] = c1.permute(0, 2, 1) # (4, 3, 512)
            pre_data['indices']['s1'] = ball_idx1
            
            # Stage 2
            c1_chunks = torch.chunk(c1, 4, dim=0)
            xyz_12 = torch.cat([c1_chunks[0], c1_chunks[1]], dim=1) 
            xyz_34 = torch.cat([c1_chunks[2], c1_chunks[3]], dim=1) 
            xyz_b2 = torch.cat([xyz_12, xyz_34], dim=0) # (2, 1024, 3)
            
            idx2 = farthest_point_sample(xyz_b2, sa2_n)
            c2 = index_points(xyz_b2, idx2)
            ball_idx2 = query_ball_point(params[1]['radius'], params[1]['nsample'], xyz_b2, c2)
            pre_data['centroids']['s2'] = c2.permute(0, 2, 1) # (2, 3, 512)
            pre_data['indices']['s2'] = ball_idx2
            
            # Stage 3
            c2_chunks = torch.chunk(c2, 2, dim=0)
            xyz_b3 = torch.cat([c2_chunks[0], c2_chunks[1]], dim=1) # (1, 1024, 3)
            
            idx3 = farthest_point_sample(xyz_b3, sa3_n)
            c3 = index_points(xyz_b3, idx3)
            ball_idx3 = query_ball_point(params[2]['radius'], params[2]['nsample'], xyz_b3, c3)
            pre_data['centroids']['s3'] = c3.permute(0, 2, 1) # (1, 3, 512)
            pre_data['indices']['s3'] = ball_idx3

            # Stage 4
            idx4 = farthest_point_sample(c3, sa4_n)
            c4 = index_points(c3, idx4)
            ball_idx4 = query_ball_point(params[3]['radius'], params[3]['nsample'], c3, c4)
            pre_data['centroids']['s4'] = c4.permute(0, 2, 1) # (1, 3, 256)
            pre_data['indices']['s4'] = ball_idx4

            for k in pre_data['centroids']:
                pre_data['centroids'][k] = pre_data['centroids'][k].to(device)
            for k in pre_data['indices']:
                pre_data['indices'][k] = pre_data['indices'][k].to(device)
        
        return tensor, pre_data