# config.py
import os
import torch

# 현재 프로젝트의 절대 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # 1. 데이터 및 경로 설정
    "PATH": {
        "data_root": os.path.join(BASE_DIR, "dataset/facescape/facescape_npy/train"),   # 학습용 데이터 경로
        "checkpoint_dir": os.path.join(BASE_DIR, "checkpoints_facescape"),    # 체크포인트 경로 (Morton Stride)
        "gallery_dir": os.path.join(BASE_DIR, "dataset/facescape/facescape_npy/test"), # 매칭할 얼굴 데이터 경로
        "gallery_storage": os.path.join(BASE_DIR, "gallery_storage/facescape"), # 미리 저장된 얼굴 데이터 경로
        "gallery_storage_enc": os.path.join(BASE_DIR, "gallery_storage_enc/facescape"), # 미리 저장된 얼굴 데이터 경로
        "secret.context": os.path.join(BASE_DIR, "gallery_storage_enc/secret.context"),
        "key_dir": os.path.join(BASE_DIR, "gallery_storage_enc/keys"),    # key directory
    },
    
    # 2. 학습(Training) 하이퍼파라미터
    "TRAIN": {
        "epochs": 500,  # 에포크 수
        "batch_size": 64,   # 배치 크기
        "learning_rate": 0.001, # 초기 학습률
        "num_workers": 4,   # 워커 개수
        "lambda_factor": 1.0, # Triplet Loss 가중치
        "margin": 0.35,       # Triplet Loss 마진
        "arcface_margin_warmup_epochs": 500,  
    },
    
    # 3. 모델(Model) 기본 설정
    "MODEL": {
        "num_classes": 700, # 분류 개수 (모델 학습에만 필요한 파라미터. feature 추출에는 상관 X)
        "input_channel": 3,  # 기본 좌표(x,y,z)
        "num_points": 1024,  # 원본 입력 점 개수
        "feature_dim": 256
    },
    
    # 4. 매칭(Matching) 설정
    "MATCHING": {
        "threshold": 0.88
    }, 
    "DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu'
}