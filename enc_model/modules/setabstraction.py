try:
    from rsconv import RSConv
    from utils.utils import sample_and_group
except ImportError:
    from .rsconv import RSConv
    from .utils.utils import sample_and_group
import torch.nn as nn


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, out_channel, hidden_channel):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        self.rsconv = RSConv(in_channel, out_channel, hidden_channel)

    def forward(self, xyz, features, precomputed_centroids=None, precomputed_indices=None):
        """
        :param xyz: (B, 3, N)
        :param features: (B, C, N)
        :param precomputed_centroids: (B, 3, npoint)    // 미리 중심점을 계산
        """
        # 1. Sampling (FPS) & Grouping (Ball Query)        
        # new_xyz: 샘플링된 중심점 (B, 3, npoint)
        # grouped_xyz: 그룹핑된 점들의 좌표 (B, 3, npoint, nsample)
        # grouped_features: 그룹핑된 점들의 특징 (B, C, npoint, nsample)
        new_xyz, grouped_xyz, grouped_features = sample_and_group(
            self.npoint, self.radius, self.nsample, xyz, features, precomputed_centroids, precomputed_indices
        )
        
        # 2. Feature Learning (RSConv)
        new_features = self.rsconv(grouped_xyz, grouped_features, new_xyz)
        
        return new_xyz, new_features