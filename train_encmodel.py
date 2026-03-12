import os
import torch
import torch.nn as nn
import torch.optim as optim
from enc_model.pointface import PointFaceNet
from enc_model.modules.featuresimloss import FeatureSimilarityLoss
from enc_model.loader import get_dataloader
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


def shift_centroids_for_negative(pre_data, shifts=1):
    """ Negative 샘플 생성을 위해 pre data를 한 칸씩 밀어버림 """
    if pre_data is None:
        return None
    return {k: {l: torch.roll(n, shifts=shifts, dims=0) for l, n in v.items()} for k, v in pre_data.items()}

# 2. 학습 루프 (Training Loop)
def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    total_loss = 0.0
    
    # dataloader는 (anchor_img, positive_img, labels) 형태의 배치를 반환
    # anchor와 positive는 같은 사람(label)의 서로 다른 데이터
    for batch_idx, (data_anchor, pre_data_anchor, data_positive, pre_data_positive, labels) in enumerate(dataloader):
        # 데이터를 device로 넘기기
        data_anchor = data_anchor.to(device)   # (B, 3, N)
        data_positive = data_positive.to(device) # (B, 3, N)
        labels = labels.to(device)             # (B, )
        for k in pre_data_anchor['centroids']:
            pre_data_anchor['centroids'][k] = pre_data_anchor['centroids'][k].to(device)
        for k in pre_data_anchor['indices']:
            pre_data_anchor['indices'][k] = pre_data_anchor['indices'][k].to(device)

        for k in pre_data_positive['centroids']:
            pre_data_positive['centroids'][k] = pre_data_positive['centroids'][k].to(device)
        for k in pre_data_positive['indices']:
            pre_data_positive['indices'][k] = pre_data_positive['indices'][k].to(device)
        
        # 부정 쌍을 생성
        # 부정 쌍은 roll을 이용해서 한 칸씩 밀어버림
        data_negative = torch.roll(data_anchor, shifts=1, dims=0)
        pre_data_negative = shift_centroids_for_negative(pre_data_anchor, shifts=1)

        optimizer.zero_grad()

        # --- Forward Pass (Siamese Network) ---
        # 가중치를 공유하는 인코더에 각각 통과시킴
        # emb: 정규화된 임베딩 (L2 Normalized)
        # logits: 신원 분류 결과
        emb_anchor, logits_anchor = model(data_anchor, pre_data_anchor)
        emb_positive, logits_positive = model(data_positive, pre_data_positive)
        emb_negative, _ = model(data_negative, pre_data_negative)

        # --- Loss Calculation ---
        # 1. Softmax Loss (Classification) - Anchor와 Positive 모두 잘 분류해야 함
        
        loss_cls_anchor = criterion_softmax(logits_anchor, labels)
        loss_cls_positive = criterion_softmax(logits_positive, labels)
        loss_softmax = loss_cls_anchor + loss_cls_positive
        
        # 2. Feature Similarity Loss (Contrastive)
        loss_sim = criterion_similarity(emb_anchor, emb_positive, emb_negative).mean()
        
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
def save_checkpoint(model, optimizer, epoch, save_dir="./checkpoints_enc"):
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
    logger = get_logger(name='train_encmodel')
    print = logger.info

    root_folder = CONFIG["PATH"]["data_root"]
    batch_size = CONFIG["TRAIN"]["batch_size"]
    num_workers = CONFIG["TRAIN"]["num_workers"]
    train_loader = get_dataloader(root_folder, batch_size=batch_size, num_workers=num_workers)

    max_epoch = CONFIG["TRAIN"]["epochs"]
    start_epoch = 0
    # 이어서 학습할 파일 경로 지정
    # CHECKPOINT_PATH = "./checkpoints_enc/pointface_epoch_080.pth" 
    # start_epoch = resume_from_checkpoint(CHECKPOINT_PATH)
    
    for epoch in range(start_epoch, max_epoch):

        start_time = time.time()
        train_one_epoch(train_loader, model, optimizer, epoch)
        end_time = time.time()
        epoch_duration = end_time - start_time
        time_str = str(datetime.timedelta(seconds=int(epoch_duration)))

        print(f"Epoch [{epoch}/{max_epoch}] Time: {time_str} ({epoch_duration:.2f}s)")
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + 1)
