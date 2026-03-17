import torch
import torch.nn as nn
import math

class SquareActivation(nn.Module):
    def forward(self, x):
        return torch.pow(x, 2) # x^2 반환


class HermiteActivation(nn.Module):
    def __init__(self):
        super(HermiteActivation, self).__init__()
        # ReLU의 에르미트 2차 전개(Hermite Expansion)에 기반한 분석적 초기 계수값
        # H_0(x) = 1, H_1(x) = x, H_2(x) = x^2 - 1
        c0 = 1.0 / math.sqrt(2 * math.pi)
        c1 = 0.5
        c2 = 1.0 / (2 * math.sqrt(2 * math.pi))
        
        # 다항식 ax^2 + bx + c 로 묶었을 때의 초기값
        init_a = c2
        init_b = c1
        init_c = c0 - c2
        
        # 모델이 Loss를 줄이는 방향으로 이 계수들을 스스로 최적화하도록 nn.Parameter 등록
        self.a = nn.Parameter(torch.tensor(init_a))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.c = nn.Parameter(torch.tensor(init_c))

    def forward(self, x):
        return x * (self.a * x + self.b) + self.c
        #return self.a * torch.pow(x, 2) + self.b * x + self.c
    

class RSConv(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel):
        super(RSConv, self).__init__()
        # Relation Prior는 10차원 벡터: (p_cen, p_k, p_cen - p_k, euclidean_dist)
        self.relation_dim = 10 
        
        # M = 가중치를 학습하는 MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(self.relation_dim, hidden_channel, 1, bias=False),
            nn.BatchNorm2d(hidden_channel, affine=False),
            #nn.ReLU(True),
            #SquareActivation(),
            HermiteActivation(),
            nn.Conv2d(hidden_channel, in_channel, 1, bias=False), # 채널을 입력 특징과 맞춤
            nn.BatchNorm2d(in_channel),
            #nn.ReLU(True)
            # SquareActivation()
        )
        
        # 특징 변환을 위한 선형 레이어
        self.linear = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm1d(out_channel),
            #nn.ReLU(True)
            #SquareActivation()
            #HermiteActivation()
        )

    def forward(self, grouped_features, relation_vector):
        """
        :param grouped_features: (B, C, npoint, nsample) - 그룹핑된 특징
        :param relation_vector: 10차원 벡터
        """
        # 1. MLP를 통해 가중치 학습
        weights = self.mlp(relation_vector) # (B, C, npoint, nsample)
        
        # 2. Element-wise Product (Hadamard product)
        if grouped_features is not None:
            if grouped_features.shape[1] > weights.shape[1]:
                grouped_features = grouped_features[:, 3:, :, :]
            weighted_features = grouped_features * weights
        else:
            weighted_features = weights
        
        # 3. Aggregation (Max Pooling)
        aggregated_features = torch.sum(weighted_features, dim=-1)
        
        # 4. 최종 차원 변환
        new_features = self.linear(aggregated_features)
        
        return new_features