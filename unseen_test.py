import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model.loader import get_dataloader
from model.pointface import PointFaceNet
from config import CONFIG

def get_test_identities(data_root):
    all_classes = sorted([
        d for d in os.listdir(data_root) 
        if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')
    ])
    
    return all_classes

def extract_embeddings(model, loader, device):
    """모델을 통해 모든 데이터의 임베딩과 라벨 추출"""
    model.eval()
    embeddings = []
    labels = []
    
    print("Extracting features...")
    with torch.no_grad():
        for anchor_pre_data, pos_pre_data, batch_labels in loader:
            # Anchor와 Positive 각각을 디바이스(GPU)로 이동
            for key in ['rel', 'idx']:
                for stage in ['s1', 's2', 's3', 's4']:
                    anchor_pre_data[key][stage] = anchor_pre_data[key][stage].to(device)
                    pos_pre_data[key][stage] = pos_pre_data[key][stage].to(device)
            batch_labels = batch_labels.to(device)
            # (B, 3, N) -> (B, 512)
            emb = model(anchor_pre_data)
            
            # CPU로 이동 및 저장
            embeddings.append(emb.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
            
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    
    # L2 정규화 (Cosine Similarity를 위해 필수)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    return embeddings, labels

def evaluate_verification(embeddings, labels):
    """
    모든 쌍(Pair)에 대해 유사도를 계산하고 성능 측정
    """
    print("Calculating Similarity Matrix...")
    # (N, 512) @ (512, N) = (N, N) Similarity Matrix
    sim_mat = np.dot(embeddings, embeddings.T)
    
    # 라벨 매트릭스 (같은 ID면 True)
    label_mat = (labels[:, None] == labels[None, :])
    
    # 상삼각행렬(Upper Triangle)만 추출 (중복 및 자기자신 비교 제외)
    # k=1: 대각선 제외
    triu_indices = np.triu_indices(len(labels), k=1)
    
    scores = sim_mat[triu_indices]
    is_same = label_mat[triu_indices]
    
    # --- 성능 지표 계산 ---
    pos_scores = scores[is_same]       # 같은 사람끼리의 점수 (높아야 함)
    neg_scores = scores[~is_same]      # 다른 사람끼리의 점수 (낮아야 함)
    
    print(f"\n[Statistics]")
    print(f" - Positive Pairs: {len(pos_scores)} (Avg Score: {np.mean(pos_scores):.4f})")
    print(f" - Negative Pairs: {len(neg_scores)} (Avg Score: {np.mean(neg_scores):.4f})")
    
    # 최적 Threshold 찾기
    best_acc = 0
    best_thresh = 0
    thresholds = np.arange(-1.0, 1.0, 0.01)
    
    for th in thresholds:
        # th보다 높으면 True(Same), 낮으면 False(Diff)로 예측
        pred = (scores > th)
        acc = np.mean(pred == is_same)
        if acc > best_acc:
            best_acc = acc
            best_thresh = th
            
    print(f"\n[Result on Unseen Data]")
    print(f" >> Best Threshold : {best_thresh:.2f}")
    print(f" >> Best Accuracy  : {best_acc*100:.2f}%")
    
    return pos_scores, neg_scores, best_thresh

def plot_distributions(pos_scores, neg_scores, threshold, savename):
    """점수 분포 시각화"""
    plt.figure(figsize=(10, 6))
    
    # Negative 분포 (Red)
    plt.hist(neg_scores, bins=100, alpha=0.6, color='red', density=True, label='Negative (Diff)')
    
    # Positive 분포 (Blue)
    plt.hist(pos_scores, bins=100, alpha=0.6, color='blue', density=True, label='Positive (Same)')
    
    # Threshold 선
    plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
    
    plt.title('Cosine Similarity Distribution (Unseen Test)')
    plt.xlabel('Similarity Score (-1 to 1)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 저장 및 출력
    plt.savefig(savename + '.png')
    print(f"\nGraph saved to '{savename}.png'")
    plt.show()

if __name__ == "__main__":
    # 설정
    DATA_ROOT = CONFIG["PATH"]["gallery_dir"] # 데이터셋 경로
    MODEL_PATH = CONFIG["PATH"]["checkpoint_dir"] # 데이터셋 경로
    DEVICE = CONFIG["DEVICE"]
    
    # 1. Unseen Identity 리스트 확보
    test_identities = get_test_identities(DATA_ROOT)
    print(f"Unseen Identities ({len(test_identities)}): {test_identities}")
    
    if len(test_identities) == 0:
        print("Error: No Unseen Identity Data.")
        exit()

    # 2. Dataset & Loader 생성
    # 테스트 시에는 Augmentation OFF (회전 등 금지), 오직 정규화만 수행
    test_loader = get_dataloader(DATA_ROOT, CONFIG["TRAIN"]["batch_size"], CONFIG["TRAIN"]["num_workers"], False)
    
    # 3. 모델 로드
    # 학습 때 num_classes=143으로 했으므로 로드할 때도 맞춰야 에러가 안 남
    # (임베딩 추출에는 마지막 레이어를 안 쓰므로 개수는 상관 없지만 weight shape 맞추기 위함)
    model_path = os.path.join(MODEL_PATH, 'pointface_epoch_200.pth')
    model = PointFaceNet(num_classes=CONFIG["MODEL"]["num_classes"]).to(DEVICE)
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        # state_dict 키 처리
        state_dict = checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
        model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # 4. 임베딩 추출
    embeddings, labels = extract_embeddings(model, test_loader, DEVICE)
    
    # 5. 검증 및 시각화
    pos_scores, neg_scores, best_th = evaluate_verification(embeddings, labels)
    plot_distributions(pos_scores, neg_scores, best_th, os.path.join(MODEL_PATH, 'unseen_test_result'))