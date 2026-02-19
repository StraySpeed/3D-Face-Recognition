import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.pointface import PointFaceNet
from model.modules.featuresimloss import FeatureSimilarityLoss
from model.loader import get_dataloader
import time, datetime
from logger import get_logger

# 1. 모델 및 손실 함수 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

# num_classes: 데이터셋의 총 ID 개수
model = PointFaceNet(num_classes=143).to(device)

# 람다(lambda) 값 (두 loss 간의 비율)
lambda_factor = 1.0 

# 손실 함수 정의
criterion_softmax = nn.CrossEntropyLoss()
criterion_similarity = FeatureSimilarityLoss(margin=0.35).to(device)

# 옵티마이저 (Adam, lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. 학습 루프 (Training Loop)
def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    total_loss = 0.0
    
    # dataloader는 (anchor_img, positive_img, labels) 형태의 배치를 반환
    # anchor와 positive는 같은 사람(label)의 서로 다른 데이터
    for batch_idx, (data_anchor, data_positive, labels) in enumerate(dataloader):
        data_anchor = data_anchor.to(device)   # (B, 3, N)
        data_positive = data_positive.to(device) # (B, 3, N)
        labels = labels.to(device)             # (B, )

        optimizer.zero_grad()

        # --- Forward Pass (Siamese Network) ---
        # 가중치를 공유하는 인코더에 각각 통과시킴
        # emb: 정규화된 임베딩 (L2 Normalized)
        # logits: 신원 분류 결과
        emb_anchor, logits_anchor = model(data_anchor)
        emb_positive, logits_positive = model(data_positive)

        # --- Hardest Negative Mining (배치 내에서 찾기) ---
        # 현재 배치 내의 다른 샘플들을 Negative로 간주
        # Anchor와 가장 가까운(코사인 유사도가 높은) Negative를 찾음
        
        # 1. Anchor와 배치 내 모든 Positive 간의 유사도 행렬 계산
        # emb_positive.T -> (512, B)
        # similarity_matrix: (B, B)
        similarity_matrix = torch.matmul(emb_anchor, emb_positive.T)
        
        # 2. 같은 사람(자기 자신 포함)은 마스킹하여 제외
        # labels: (B, ) -> labels.unsqueeze(1) == labels.unsqueeze(0): (B, B)
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        
        # 같은 사람인 곳은 유사도를 매우 낮게(-100) 설정하여 선택되지 않게 함
        # 이렇게 하면 다른 사람 중에서 가장 유사도가 높은 것을 찾을 수 있음
        similarity_matrix[label_matrix] = -100.0
        
        # 3. 각 Anchor에 대해 가장 유사도가 높은(Hardest) Negative 인덱스 추출
        hardest_negative_indices = torch.max(similarity_matrix, dim=1)[1]
        
        # 4. Hardest Negative 임베딩 가져오기
        emb_negative = emb_positive[hardest_negative_indices] # (B, 512)

        # --- Loss Calculation ---
        # 1. Softmax Loss (Classification) - Anchor와 Positive 모두 잘 분류해야 함
        loss_cls_anchor = criterion_softmax(logits_anchor, labels)
        loss_cls_positive = criterion_softmax(logits_positive, labels)
        loss_softmax = loss_cls_anchor + loss_cls_positive
        
        # 2. Feature Similarity Loss (Contrastive)
        loss_sim = criterion_similarity(emb_anchor, emb_positive, emb_negative)
        
        # 3. Total Loss
        loss = loss_softmax + lambda_factor * loss_sim
        
        # --- Backward Pass ---
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # DEBUG 출력
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}] Loss: {loss.item():.4f} "
                  f"(Softmax: {loss_softmax.item():.4f}, Sim: {loss_sim.item():.4f})")

# 3. Checkpoint
def save_checkpoint(model, optimizer, epoch, save_dir="./checkpoints"):
    """
    모델 가중치와 학습 상태를 저장하는 함수

    :param model: PointFace Model
    :param optimizer: Optimizer (Adam)
    :param epoch: epoch
    :param save_dir: ./checkpoints
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 저장할 파일 경로
    save_path = os.path.join(save_dir, f"pointface_epoch_{epoch:03d}.pth")
    
    # 저장할 데이터 딕셔너리 구성
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),      # 모델 가중치
        'optimizer_state_dict': optimizer.state_dict(), # 옵티마이저 상태 (이어하기 용)
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    # 로거 생성
    logger = get_logger(name='train')
    print = logger.info

    root_folder = 'dataset/umbdb'
    train_loader = get_dataloader(root_folder, batch_size=32, num_workers=0)
    max_epoch = 200
    for epoch in range(max_epoch):

        start_time = time.time()
        train_one_epoch(train_loader, model, optimizer, epoch)
        end_time = time.time()
        epoch_duration = end_time - start_time
        time_str = str(datetime.timedelta(seconds=int(epoch_duration)))

        print(f"Epoch [{epoch}/{max_epoch}] Time: {time_str} ({epoch_duration:.2f}s)")
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + 1)