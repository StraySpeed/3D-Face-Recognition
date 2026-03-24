try:      
    from modules.setabstraction import SetAbstraction
except ImportError:
    from .modules.setabstraction import SetAbstraction
import torch.nn as nn
import torch
from config import CONFIG

class PointFaceEncoder(nn.Module):
    def __init__(self): # 보통 좌표(3) 또는 좌표+법선(6)
        super(PointFaceEncoder, self).__init__()
        
        # features with dimension of 64, 128, 256, 512, 1024 
        # (npoint, radius, nsample, in_channel, out_channel, hidden_mlp)
        # radius는 PointNet++처럼 점진적 증가
        sa_params = CONFIG['SA_PARAMS']
        
        self.sa1 = SetAbstraction(sa_params[0]['npoint'], sa_params[0]['radius'], sa_params[0]['nsample'], sa_params[0]['in_ch'], sa_params[0]['out_ch'], sa_params[0]['hid_ch'])
        self.sa2 = SetAbstraction(sa_params[1]['npoint'], sa_params[1]['radius'], sa_params[1]['nsample'], sa_params[1]['in_ch'], sa_params[1]['out_ch'], sa_params[1]['hid_ch'])
        self.sa3 = SetAbstraction(sa_params[2]['npoint'], sa_params[2]['radius'], sa_params[2]['nsample'], sa_params[2]['in_ch'], sa_params[2]['out_ch'], sa_params[2]['hid_ch'])
        self.sa4 = SetAbstraction(sa_params[3]['npoint'], sa_params[3]['radius'], sa_params[3]['nsample'], sa_params[3]['in_ch'], sa_params[3]['out_ch'], sa_params[3]['hid_ch'])
        
        # Final Fully Connected Layer 
        self.fc = nn.Sequential(
            nn.Linear(sa_params[-1]['out_ch'], CONFIG["MODEL"]["feature_dim"]),
            nn.BatchNorm1d(CONFIG["MODEL"]["feature_dim"]),
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
        
        # Global Max Pooling: (B, 512, 256) -> (B, 512)
        global_feature = torch.max(l4_points, 2)[0]
        
        # Embedding 생성 (B, 512)
        embedding = self.fc(global_feature)
        
        return embedding
