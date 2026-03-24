try:      
    from modules.setabstraction import SetAbstraction
except ImportError:
    from .modules.setabstraction import SetAbstraction
import torch.nn as nn
import torch
from config import CONFIG

class PointFaceEncoder(nn.Module):
    def __init__(self): # 보통 좌표(3) 또는 좌표+법선(6)인데 3을 이용
        super(PointFaceEncoder, self).__init__()
        
        # (in_channel, out_channel, hidden_mlp)
        sa_params = CONFIG['SA_PARAMS']
        
        # 설정 파일의 값을 그대로 가져오기
        self.sa1 = SetAbstraction(sa_params[0]['in_ch'], sa_params[0]['out_ch'], sa_params[0]['hid_ch'])
        self.sa2 = SetAbstraction(sa_params[1]['in_ch'], sa_params[1]['out_ch'], sa_params[1]['hid_ch'])
        self.sa3 = SetAbstraction(sa_params[2]['in_ch'], sa_params[2]['out_ch'], sa_params[2]['hid_ch'])
        self.sa4 = SetAbstraction(sa_params[3]['in_ch'], sa_params[3]['out_ch'], sa_params[3]['hid_ch'])
        
        # Final Fully Connected Layer 
        self.fc = nn.Sequential(
            nn.Linear(sa_params[-1]['out_ch'], CONFIG["MODEL"]["feature_dim"]),
            nn.BatchNorm1d(CONFIG["MODEL"]["feature_dim"]),
        )

    def forward(self, pre_data):
        precomputed_rel_vec = pre_data['rel']
        precomputed_indices = pre_data['idx']
        
        # Stage 1 (입력 feature 없음)
        l1_features = self.sa1(None, precomputed_rel_vec['s1'], precomputed_indices['s1'])
        
        # Stage 2 ~ 4
        l2_features = self.sa2(l1_features, precomputed_rel_vec['s2'], precomputed_indices['s2'])
        l3_features = self.sa3(l2_features, precomputed_rel_vec['s3'], precomputed_indices['s3'])
        l4_features = self.sa4(l3_features, precomputed_rel_vec['s4'], precomputed_indices['s4'])
        
        # Global Sum Pooling
        global_feature = torch.sum(l4_features, 2)
        
        # Embedding 생성 (B, feature_dim)
        embedding = self.fc(global_feature)
        
        return embedding
