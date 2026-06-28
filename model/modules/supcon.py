import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020)

    배치 내 동일 레이블 쌍을 모두 Positive로, 나머지를 Negative로 처리.
    Triplet과 달리 pair mining이 필요 없고, 배치가 클수록 신호가 풍부해짐.

    L = Σ_i  -1/|P_i|  Σ_{p∈P_i}  log( exp(sim(i,p)/τ) / Σ_{a≠i} exp(sim(i,a)/τ) )
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        :param features: (N, dim) — L2 정규화된 임베딩
        :param labels:   (N,)    — 클래스 레이블
        """
        device = features.device
        N = features.shape[0]

        # 코사인 유사도 행렬 / 온도
        sim = torch.matmul(features, features.T) / self.temperature  # (N, N)

        # 수치 안정성: 최댓값 빼기
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Positive 마스크: 같은 레이블이면 1, 자기 자신은 제외
        labels_col = labels.view(-1, 1)
        pos_mask = torch.eq(labels_col, labels_col.T).float()      # (N, N)
        self_mask = torch.eye(N, device=device)
        pos_mask  = pos_mask - self_mask                            # 대각선 제거

        # 분모: 자기 자신 제외한 모든 쌍
        exp_sim   = torch.exp(sim) * (1.0 - self_mask)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # 샘플별 손실: positive 쌍들의 평균 log-probability
        log_prob  = sim - log_denom                                 # (N, N)
        pos_count = pos_mask.sum(dim=1)                             # (N,)
        loss_per  = -(pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

        # positive 쌍이 하나도 없는 샘플(배치 내 유일 클래스)은 제외
        valid = pos_count > 0
        return loss_per[valid].mean()
