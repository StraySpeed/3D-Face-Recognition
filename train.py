import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.pointface import PointFaceNet
from model.modules.featuresimloss import FeatureSimilarityLoss
from model.loader import get_dataloader
import time, datetime
from logger import get_logger
from config import CONFIG

# 1. 모델 및 손실 함수 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

# num_classes: 데이터셋의 총 ID 개수
model = PointFaceNet(num_classes=CONFIG["MODEL"]["num_classes"]).to(device)

# 람다(lambda) 값 (두 loss 간의 비율)
lambda_factor = CONFIG["TRAIN"]["lambda_factor"]

# 손실 함수 정의
criterion_softmax = nn.CrossEntropyLoss()
criterion_similarity = FeatureSimilarityLoss(margin=CONFIG["TRAIN"]["margin"]).to(device)

# 옵티마이저 (Adam, lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=CONFIG["TRAIN"]["learning_rate"])


def create_negative_pre_data(pre_data, shifts=1):
    """
    기존 pre_data의 배치(Batch) 차원을 지정한 만큼 이동(shift)시켜 
    부정 쌍(Negative pre_data)을 생성
    """
    neg_pre_data = {'rel': {}, 'idx': {}}
    
    # 1. relation_vector 텐서들을 모두 shift
    for stage in ['s1', 's2', 's3', 's4']:
        neg_pre_data['rel'][stage] = torch.roll(pre_data['rel'][stage], shifts=shifts, dims=0)
        
    # 2. indices 텐서들을 모두 shift
    for stage in ['s1', 's2', 's3', 's4']:
        neg_pre_data['idx'][stage] = torch.roll(pre_data['idx'][stage], shifts=shifts, dims=0)
        
    return neg_pre_data

# 2. 학습 루프 (Training Loop)
def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    total_loss = 0.0
    
    # dataloader는 (anchor_pre_data, pos_pre_data, labels) 형태의 배치를 반환
    # anchor와 positive는 같은 사람(label)의 서로 다른 데이터
    for batch_idx, (anchor_pre_data, pos_pre_data, labels) in enumerate(dataloader):
        # Anchor와 Positive 각각을 디바이스(GPU)로 이동
        for key in ['rel', 'idx']:
            for stage in ['s1', 's2', 's3', 's4']:
                anchor_pre_data[key][stage] = anchor_pre_data[key][stage].to(device)
                pos_pre_data[key][stage] = pos_pre_data[key][stage].to(device)
        labels = labels.to(device)
        
        # 부정 쌍을 생성
        # 부정 쌍은 roll을 이용해서 한 칸씩 밀어버림
        neg_pre_data = create_negative_pre_data(anchor_pre_data)
        # 라벨도 1칸 Shift하여 Negative 쌍의 원래 라벨을 추적
        neg_labels = torch.roll(labels, shifts=1, dims=0)
        optimizer.zero_grad()

        # --- Forward Pass (Siamese Network) ---
        # 가중치를 공유하는 인코더에 각각 통과시킴
        # emb: 정규화된 임베딩 (L2 Normalized)
        # logits: 신원 분류 결과
        emb_anchor, logits_anchor = model(anchor_pre_data)
        emb_positive, logits_positive = model(pos_pre_data)
        emb_negative, _ = model(neg_pre_data)

        # --- Loss Calculation ---
        # 1. Softmax Loss (Classification) - Anchor와 Positive 모두 잘 분류해야 함
        
        loss_cls_anchor = criterion_softmax(logits_anchor, labels)
        loss_cls_positive = criterion_softmax(logits_positive, labels)
        loss_softmax = loss_cls_anchor + loss_cls_positive
        
        # anchor와 negative의 라벨이 다를 때만 1.0, 같으면 0.0이 되는 마스크 생성
        valid_mask = (labels != neg_labels).float()

        # 개별 샘플의 Loss를 계산
        raw_sim_loss = criterion_similarity(emb_anchor, emb_positive, emb_negative)

        # 유효한 쌍에 대해서만 Loss를 남기고 평균 계산
        loss_sim = (raw_sim_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
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
def save_checkpoint(model, optimizer, epoch, save_dir="./checkpoints_enc2"):
    """
    모델 가중치와 학습 상태를 저장하는 함수

    :param model: PointFace Model
    :param optimizer: Optimizer (Adam)
    :param epoch: epoch
    :param save_dir: ./checkpoints_enc
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

def resume_from_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 1. 모델 가중치 복구
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 2. 옵티마이저 상태 복구 (Adam의 모멘텀 등 내부 상태 유지)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 3. 시작 에포크 업데이트 (저장된 에포크 다음부터 시작)
        start_epoch = checkpoint['epoch']
        print(f"Start at {start_epoch} epoch.")
        return start_epoch
    return 0


if __name__ == '__main__':
    # 로거 생성
    logger = get_logger(name='train_v4')
    print = logger.info

    root_folder = CONFIG["PATH"]["data_root"]
    batch_size = CONFIG["TRAIN"]["batch_size"]
    num_workers = CONFIG["TRAIN"]["num_workers"]
    savepath = CONFIG["PATH"]["checkpoint_dir"]
    train_loader = get_dataloader(root_folder, batch_size=batch_size, num_workers=num_workers)

    max_epoch = CONFIG["TRAIN"]["epochs"]
    start_epoch = 0
    # 이어서 학습할 파일 경로 지정
    #CHECKPOINT_PATH = "./checkpoints_enc2/pointface_epoch_070.pth" 
    #start_epoch = resume_from_checkpoint(CHECKPOINT_PATH)
    
    for epoch in range(start_epoch, max_epoch):

        start_time = time.time()
        train_one_epoch(train_loader, model, optimizer, epoch)
        end_time = time.time()
        epoch_duration = end_time - start_time
        time_str = str(datetime.timedelta(seconds=int(epoch_duration)))

        print(f"Epoch [{epoch}/{max_epoch}] Time: {time_str} ({epoch_duration:.2f}s)")
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + 1, savepath)
