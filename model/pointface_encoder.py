try:      
    from modules.setabstraction import SetAbstraction
except ImportError:
    from .modules.setabstraction import SetAbstraction
import torch.nn as nn
import torch

class PointFaceEncoder(nn.Module):
    def __init__(self, input_channel=3): # 보통 좌표(3) 또는 좌표+법선(6)
        super(PointFaceEncoder, self).__init__()
        
        # features with dimension of 64, 128, 256, 512, 1024 
        # (npoint, radius, nsample, in_channel, out_channel, hidden_mlp)
        # radius는 PointNet++ 일반 설정(점진적 증가)을 따름
        
        self.sa1 = SetAbstraction(2048, 0.10, 32, input_channel, 64, 16)
        self.sa2 = SetAbstraction(1024, 0.20, 32, 64, 128, 32)
        self.sa3 = SetAbstraction(512, 0.40, 32, 128, 256, 64)
        self.sa4 = SetAbstraction(256, 0.60, 32, 256, 512, 128)
        # 마지막 레이어는 보통 Global Feature를 위해 점 개수를 줄이거나 radius를 크게 함
        self.sa5 = SetAbstraction(64, 0.80, 32, 512, 1024, 256) 
        
        # Final Fully Connected Layer 
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
        )

    def forward(self, x):
        """
        :param x: (B, C, N)
        """
        xyz = x[:, :3, :] # 좌표
        features = x      # 특징 (좌표를 포함할 수 있음)
        
        # 5단계 계층적 특징 추출
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        
        # Global Max Pooling: (B, 1024, 64) -> (B, 1024)
        global_feature = torch.max(l5_points, 2)[0]
        
        # Embedding 생성 (B, 512)
        embedding = self.fc(global_feature)
        
        return embedding
