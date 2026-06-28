"""
PointFace(model, 4블럭) unseen test 스크립트.

  - test 데이터셋으로 임베딩 추출 → 모든 쌍에 대한 cosine similarity
  - threshold sweep으로 best accuracy / EER 측정
  - positive/negative 분포 시각화
  - 모델: PointFaceNet (model, 4블럭)
  - 체크포인트 디렉토리: CONFIG["PATH"]["checkpoint_dir"]
  - 체크포인트 파일명: pointface_epoch_*.pth
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model.pointface import PointFaceNet, DEFAULT_FEATURE_DIM
from model.loader.pointface_dataset import PointFaceDataset
from config import CONFIG


def get_test_identities(data_root):
    return sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')
    ])


def extract_embeddings(model, loader, device):
    """모델을 통해 모든 데이터의 임베딩과 라벨 추출."""
    model.eval()
    embeddings = []
    labels = []

    print("Extracting features...")
    with torch.no_grad():
        for anchor, _pos, batch_labels in loader:
            anchor = anchor.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            emb = model(anchor)

            embeddings.append(emb.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    # 수치 안정성을 위해 한 번 더 L2 정규화
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    return embeddings, labels


def evaluate_verification(embeddings, labels):
    """모든 쌍에 대해 유사도 계산 및 성능 측정."""
    print("Calculating Similarity Matrix...")
    sim_mat = np.dot(embeddings, embeddings.T)
    label_mat = (labels[:, None] == labels[None, :])

    # 상삼각만 추출 (중복 및 자기자신 비교 제외)
    triu_indices = np.triu_indices(len(labels), k=1)
    scores = sim_mat[triu_indices]
    is_same = label_mat[triu_indices]

    pos_scores = scores[is_same]
    neg_scores = scores[~is_same]

    print(f"\n[Statistics]")
    print(f" - Positive Pairs: {len(pos_scores)} (Avg Score: {np.mean(pos_scores):.4f})")
    print(f" - Negative Pairs: {len(neg_scores)} (Avg Score: {np.mean(neg_scores):.4f})")

    best_acc = 0
    best_thresh = 0
    eer_thresh = 0
    eer_value = 1.0
    eer_far = eer_frr = 0
    thresholds = np.arange(0.0, 1.0, 0.001)

    for th in thresholds:
        pred = (scores > th)
        acc = np.mean(pred == is_same)
        if acc > best_acc:
            best_acc = acc
            best_thresh = th

        far = np.mean(pred[~is_same])      # 다른 사람인데 통과
        frr = np.mean(~pred[is_same])      # 같은 사람인데 차단
        eer_dist = abs(far - frr)
        if eer_dist < eer_value:
            eer_value = eer_dist
            eer_thresh = th
            eer_far = far
            eer_frr = frr

    print(f"\n[Result on Unseen Data]")
    print(f" >> Best Threshold (max accuracy) : {best_thresh:.3f}  Acc: {best_acc*100:.2f}%")
    print(f" >> EER Threshold                 : {eer_thresh:.3f}  EER: {(eer_far+eer_frr)/2*100:.2f}%  (FAR={eer_far*100:.2f}%, FRR={eer_frr*100:.2f}%)")

    print(f"\n[FRR / FAR at key thresholds]")
    for th in [0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97]:
        pred_th = (scores > th)
        far_th = np.mean(pred_th[~is_same]) * 100
        frr_th = np.mean(~pred_th[is_same]) * 100
        print(f"  th={th:.2f}  FAR={far_th:5.2f}%  FRR={frr_th:5.2f}%")

    return pos_scores, neg_scores, best_thresh


def plot_distributions(pos_scores, neg_scores, threshold, savename):
    plt.figure(figsize=(10, 6))
    plt.hist(neg_scores, bins=100, alpha=0.6, color='red',  density=True, label='Negative (Diff)')
    plt.hist(pos_scores, bins=100, alpha=0.6, color='blue', density=True, label='Positive (Same)')
    plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.2f})')
    plt.title('Cosine Similarity Distribution (Unseen Test)')
    plt.xlabel('Similarity Score (-1 to 1)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(savename + '.png')
    print(f"\nGraph saved to '{savename}.png'")
    plt.show()


if __name__ == "__main__":
    DATA_ROOT  = CONFIG["PATH"]["gallery_dir"]
    MODEL_PATH = CONFIG["PATH"]["checkpoint_dir"]
    DEVICE     = CONFIG["DEVICE"]

    test_identities = get_test_identities(DATA_ROOT)
    print(f"Unseen Identities ({len(test_identities)}): {test_identities}")

    if len(test_identities) == 0:
        print("Error: No Unseen Identity Data.")
        exit()

    # Dataset (Augmentation OFF, drop_last=False로 전체 평가)
    test_dataset = PointFaceDataset(
        data_root=DATA_ROOT,
        train=False,
        num_points=CONFIG["MODEL"]["num_points"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["TRAIN"]["batch_size"],
        shuffle=False,
        num_workers=CONFIG["TRAIN"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    # 최신 체크포인트 자동 선택
    if not os.path.isdir(MODEL_PATH):
        print(f"Error: Checkpoint directory not found: {MODEL_PATH}")
        exit()
    pth_files = sorted([
        f for f in os.listdir(MODEL_PATH)
        if f.startswith('pointface_epoch_') and f.endswith('.pth')
    ])
    # pth_files = ["pointface_epoch_400.pth"]
    if not pth_files:
        print(f"Error: No checkpoint found in {MODEL_PATH}")
        exit()
    model_path = os.path.join(MODEL_PATH, pth_files[-1])
    print(f"Using checkpoint: {model_path}")

    model = PointFaceNet(feature_dim=DEFAULT_FEATURE_DIM).to(DEVICE)
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully.")
    except RuntimeError as e:
        if "size mismatch" in str(e) or "unexpected key" in str(e) or "Missing key" in str(e):
            print(f"\n[Architecture Mismatch]")
            print(f"체크포인트가 현재 모델 구조와 다릅니다.")
            print(f"원인: 모델 구조 변경 후 아직 재학습이 완료되지 않았습니다.")
            print(f"해결: python train.py 로 재학습 후 다시 실행하세요.\n")
            print(f"상세 오류: {e}")
        else:
            print(f"Error loading model: {e}")
        exit()

    embeddings, labels = extract_embeddings(model, test_loader, DEVICE)
    pos_scores, neg_scores, best_th = evaluate_verification(embeddings, labels)
    plot_distributions(
        pos_scores, neg_scores, best_th,
        os.path.join(MODEL_PATH, 'unseen_test_result'),
    )
