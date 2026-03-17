try:
    from rsconv import RSConv
    from utils.utils import index_points
except ImportError:
    from .rsconv import RSConv
    from .utils.utils import index_points
import torch.nn as nn


class SetAbstraction(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel):
        super(SetAbstraction, self).__init__()
        self.rsconv = RSConv(in_channel, out_channel, hidden_channel)

    def forward(self, features, relation_vector, precomputed_indices=None):
        """
        :param xyz: (B, 3, N)
        :param relation_vector: 10차원 벡터
        :param precomputed_indices: Grouping Indices
        """
        if features is not None:
            features_t = features.permute(0, 2, 1)
            grouped_features = index_points(features_t, precomputed_indices).permute(0, 3, 1, 2)
        else:
            grouped_features = None

        # 2. Feature Learning (RSConv)
        new_features = self.rsconv(grouped_features, relation_vector)
        
        return new_features