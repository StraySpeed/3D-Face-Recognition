import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss  (margin warmup 지원)

    forward(x, labels, margin)
      - margin 전달 시: 해당 값으로 동적 마진 적용 (warmup 용)
      - margin=None   : self.m 고정값 사용
      - labels=None   : 마진 없이 cos_theta * s 반환 (embedding-only 경로)
    """
    def __init__(self, in_features, num_classes, s=32.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None, margin=None):
        m = margin if margin is not None else self.m

        W = F.normalize(self.weight, dim=1)
        cos_theta = F.linear(x, W).clamp(-1.0, 1.0)  # (B, num_classes)

        if labels is None or m == 0.0:
            return cos_theta * self.s

        cos_m = math.cos(m)
        sin_m = math.sin(m)
        th    = math.cos(math.pi - m)
        mm    = math.sin(math.pi - m) * m

        sin_theta   = torch.sqrt((1.0 - cos_theta ** 2).clamp(min=1e-8))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        cos_theta_m = torch.where(cos_theta > th, cos_theta_m, cos_theta - mm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        return logits * self.s
