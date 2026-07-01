import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
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
scaler = GradScaler()

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

        with autocast(device_type='cuda'):
            # --- Forward Pass (Siamese Network) ---
            emb_anchor, logits_anchor = model(data_anchor)
            emb_positive, logits_positive = model(data_positive)

            # --- Hardest Negative Mining ---
            similarity_matrix = torch.matmul(emb_anchor, emb_positive.T)
            label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
            similarity_matrix[label_matrix] = -100.0
            hardest_negative_indices = torch.max(similarity_matrix, dim=1)[1]
            emb_negative = emb_positive[hardest_negative_indices]

            # --- Loss Calculation ---
            loss_cls_anchor  = criterion_softmax(logits_anchor,  labels)
            loss_cls_positive = criterion_softmax(logits_positive, labels)
            loss_softmax = loss_cls_anchor + loss_cls_positive
            loss_sim = criterion_similarity(emb_anchor, emb_positive, emb_negative)
            loss = loss_softmax + lambda_factor * loss_sim

        # --- Backward Pass ---
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
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
    logger = get_logger(name='train_v1')
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