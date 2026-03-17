import torch
import torch.nn as nn

class RSConv(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel):
        super(RSConv, self).__init__()
        # Relation Prior는 10차원 벡터: (p_cen, p_k, p_cen - p_k, euclidean_dist)
        self.relation_dim = 10 
        
        # M = 가중치를 학습하는 MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(self.relation_dim, hidden_channel, 1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(True),
            nn.Conv2d(hidden_channel, in_channel, 1, bias=False), # 채널을 입력 특징과 맞춤
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True)
        )
        
        # 특징 변환을 위한 선형 레이어
        self.linear = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, grouped_xyz, grouped_features, new_xyz):
        """
        :param grouped_xyz: (B, 3, npoint, nsample) - 그룹핑된 좌표
        :param grouped_features: (B, C, npoint, nsample) - 그룹핑된 특징
        :param new_xyz: (B, 3, npoint) - 중심점 좌표
        """
        B, C, npoint, nsample = grouped_features.shape
        
        # 1. Relation Prior (h_ij) 생성
        # 중심점 확장: (B, 3, npoint, 1) -> (B, 3, npoint, nsample)
        center_xyz = new_xyz.unsqueeze(-1).repeat(1, 1, 1, nsample)
        
        # 유클리드 거리 계산
        euclidean = torch.sqrt(torch.sum((grouped_xyz - center_xyz)**2, dim=1, keepdim=True))
        
        # 10차원 벡터 연결: (p_cen, p_k, p_cen-p_k, dist)
        # p_cen: center_xyz
        # p_k: grouped_xyz
        relation_vector = torch.cat([
            center_xyz, 
            grouped_xyz, 
            center_xyz - grouped_xyz, 
            euclidean
        ], dim=1) # (B, 10, npoint, nsample)
        
        # 2. MLP를 통해 가중치 학습
        weights = self.mlp(relation_vector) # (B, C, npoint, nsample)
        
        # 3. Element-wise Product (Hadamard product)
        if grouped_features is not None:
            if grouped_features.shape[1] > weights.shape[1]:
                grouped_features = grouped_features[:, 3:, :, :]
            weighted_features = grouped_features * weights
        else:
            weighted_features = weights
        
        # 4. Aggregation (Max Pooling)
        aggregated_features = torch.max(weighted_features, dim=-1)[0] # (B, C, npoint)
        
        # 5. 최종 차원 변환
        new_features = self.linear(aggregated_features)
        
        return new_features