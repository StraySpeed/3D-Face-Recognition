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
        
        # (npoint, radius, nsample, in_channel, out_channel, hidden_mlp)
        sa_params = CONFIG['SA_PARAMS']
        
        # 설정 파일의 값을 그대로 가져오기
        self.sa1 = SetAbstraction(sa_params[0]['npoint'], sa_params[0]['radius'], sa_params[0]['nsample'], sa_params[0]['in_ch'], sa_params[0]['out_ch'], sa_params[0]['hid_ch'])
        self.sa2 = SetAbstraction(sa_params[1]['npoint'], sa_params[1]['radius'], sa_params[1]['nsample'], sa_params[1]['in_ch'], sa_params[1]['out_ch'], sa_params[1]['hid_ch'])
        self.sa3 = SetAbstraction(sa_params[2]['npoint'], sa_params[2]['radius'], sa_params[2]['nsample'], sa_params[2]['in_ch'], sa_params[2]['out_ch'], sa_params[2]['hid_ch'])
        self.sa4 = SetAbstraction(sa_params[3]['npoint'], sa_params[3]['radius'], sa_params[3]['nsample'], sa_params[3]['in_ch'], sa_params[3]['out_ch'], sa_params[3]['hid_ch'])
        
        # Final Fully Connected Layer 
        self.fc = nn.Sequential(
            nn.Linear(sa_params[-1]["out_ch"], CONFIG["MODEL"]["feature_dim"]),
            nn.BatchNorm1d(CONFIG["MODEL"]["feature_dim"]),
        )

    def forward(self, x, pre_data=None):
        """
        :param x: (B, C, N)
        """
        B = x.shape[0]
        xyz = x[:, :3, :] # 좌표
        features = x[:, 3:, :] if x.shape[1] > 3 else None      # 특징
        precomputed_centroids = pre_data['centroids']
        precomputed_indices = pre_data['indices']
        
        # 입력 데이터를 4등분 후 1단계 처리
        num_points = CONFIG["MODEL"]["num_points"] // 4
        xyz_batch1 = xyz.view(B, 3, 4, num_points).transpose(1, 2).reshape(-1, 3, num_points)
        if features is not None:
            feat_batch1 = features.view(B, features.shape[1], 4, num_points).transpose(1, 2).reshape(-1, features.shape[1], num_points)
        else:
            feat_batch1 = None

        s1_points = CONFIG["SA_PARAMS"][0]["npoint"]
        s1_sample = CONFIG["SA_PARAMS"][0]["nsample"]
        c1 = precomputed_centroids['s1'].view(-1, 3, s1_points) if precomputed_centroids else None
        i1 = precomputed_indices['s1'].view(-1, s1_points, s1_sample)  if precomputed_indices else None
        l1_xyz, l1_points = self.sa1(xyz_batch1, feat_batch1, c1, i1)

        # 4 -> 2개씩 붙여서 2단계 처리
        l1_x_unflat = l1_xyz.view(B, 4, 3, s1_points)
        xyz_12 = torch.cat([l1_x_unflat[:, 0], l1_x_unflat[:, 1]], dim=2) # (B, 3, 1024)
        xyz_34 = torch.cat([l1_x_unflat[:, 2], l1_x_unflat[:, 3]], dim=2)
        xyz_batch2 = torch.stack([xyz_12, xyz_34], dim=1).view(-1, 3, s1_points * 2) # (2B, 3, 1024)
        
        l1_f_unflat = l1_points.view(B, 4, l1_points.shape[1], s1_points)
        feat_12 = torch.cat([l1_f_unflat[:, 0], l1_f_unflat[:, 1]], dim=2)
        feat_34 = torch.cat([l1_f_unflat[:, 2], l1_f_unflat[:, 3]], dim=2)
        feat_batch2 = torch.stack([feat_12, feat_34], dim=1).view(-1, l1_points.shape[1], s1_points * 2)
        
        s2_points = CONFIG["SA_PARAMS"][1]["npoint"]
        s2_sample = CONFIG["SA_PARAMS"][1]["nsample"]
        c2 = precomputed_centroids['s2'].view(-1, 3, s2_points) if precomputed_centroids else None
        i2 = precomputed_indices['s2'].view(-1, s2_points, s2_sample)  if precomputed_indices else None
        l2_xyz, l2_points = self.sa2(xyz_batch2, feat_batch2, c2, i2)
        
        # 2 -> 1로 최종 처리
        l2_x_unflat = l2_xyz.view(B, 2, 3, s2_points)
        l2_xyz_all = torch.cat([l2_x_unflat[:, 0], l2_x_unflat[:, 1]], dim=2)
        
        l2_f_unflat = l2_points.view(B, 2, l2_points.shape[1], s2_points)
        l2_points_all = torch.cat([l2_f_unflat[:, 0], l2_f_unflat[:, 1]], dim=2)
        
        s3_points = CONFIG["SA_PARAMS"][2]["npoint"]
        s3_sample = CONFIG["SA_PARAMS"][2]["nsample"]
        c3 = precomputed_centroids['s3'].view(-1, 3, s3_points) if precomputed_centroids else None
        i3 = precomputed_indices['s3'].view(-1, s3_points, s3_sample)  if precomputed_indices else None
        l3_xyz, l3_points = self.sa3(l2_xyz_all, l2_points_all, c3, i3)

        s4_points = CONFIG["SA_PARAMS"][3]["npoint"]
        s4_sample = CONFIG["SA_PARAMS"][3]["nsample"]
        # 이게 필요한가?는 고민해볼 것
        c4 = precomputed_centroids['s4'].view(-1, 3, s4_points) if precomputed_centroids else None
        i4 = precomputed_indices['s4'].view(-1, s4_points, s4_sample)  if precomputed_indices else None
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points, c4, i4)
        
        # Global Max Pooling: (B, 512, 256) -> (B, 512)
        global_feature = torch.max(l4_points, 2)[0]
        
        # Embedding 생성 (B, 512)
        embedding = self.fc(global_feature)
        
        return embedding
