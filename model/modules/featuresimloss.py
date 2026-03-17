import torch
import torch.nn as nn

class FeatureSimilarityLoss(nn.Module):
    """
    ## Formula
    L_sim = max(0, Dist(Anchor, Positive) - Dist(Anchor, Negative) + margin)

    ## Distance
    Cosine Distance
    """
    def __init__(self, margin=0.35):
        super(FeatureSimilarityLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        :param anchor: (Batch, Dim) - L2 Normalized Embedding
        :param positive: (Batch, Dim) - L2 Normalized Embedding
        :param negative: (Batch, Dim) - L2 Normalized Embedding
        """
        
        # 1. Cosine Similarity 계산
        # 입력이 이미 L2 Normalize 되어 있으므로 내적이 코사인 유사도
        # (Batch, Dim) * (Batch, Dim) -> (Batch,)
        sim_pos = torch.sum(anchor * positive, dim=1)
        sim_neg = torch.sum(anchor * negative, dim=1)
        
        # 2. Cosine Distance 변환 (Distance = 1 - Similarity)
        # L2 거리 대신 코사인 거리를 사용
        dist_pos = 1.0 - sim_pos
        dist_neg = 1.0 - sim_neg
        
        # 3. Triplet Loss 계산
        # Loss = D(a, p) - D(a, n) + margin
        # D(a, p)는 작아져야 하고, D(a, n)은 커져야 함
        loss = dist_pos - dist_neg + self.margin
        
        # 4. ReLU (0보다 작은 값은 0으로 처리)
        loss = torch.clamp(loss, min=0.0)
        
        # 5. Loss 반환
        return loss