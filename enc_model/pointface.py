try:
    from pointface_encoder import PointFaceEncoder
except ImportError:
    from .pointface_encoder import PointFaceEncoder

import torch.nn as nn
import torch.nn.functional as F


class PointFaceNet(nn.Module):
    def __init__(self, num_classes):
        super(PointFaceNet, self).__init__()
        self.encoder = PointFaceEncoder()
        
        # 학습 시 Identity Classification을 위한 Softmax Layer
        self.classifier = nn.Linear(512, num_classes)

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