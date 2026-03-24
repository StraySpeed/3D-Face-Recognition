# config.py
import os
import torch

# 현재 프로젝트의 절대 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # 1. 데이터 및 경로 설정
    "PATH": {
        "data_root": os.path.join(BASE_DIR, "dataset/umbdb"),   # 학습용 데이터 경로
        "checkpoint_dir": os.path.join(BASE_DIR, "checkpoints2"),    # 체크포인트 경로
        "gallery_dir": os.path.join(BASE_DIR, "dataset_matching/umbdb_unpreprocessed"), # 매칭할 얼굴 데이터 경로
        "gallery_storage": os.path.join(BASE_DIR, "gallery_storage2/umbdb"), # 미리 저장된 얼굴 데이터 경로
        "gallery_storage_enc": os.path.join(BASE_DIR, "gallery_storage2_enc/umbdb"), # 미리 저장된 얼굴 데이터 경로
        "secret.context": os.path.join(BASE_DIR, "gallery_storage2_enc/secret.context")
    },
    
    # 2. 학습(Training) 하이퍼파라미터
    "TRAIN": {
        "epochs": 200,  # 에포크 수
        "batch_size": 32,   # 배치 크기
        "learning_rate": 0.001, # 초기 학습률
        "num_workers": 4,   # 워커 개수
        "lambda_factor": 1.0, # Loss 가중치
        "margin": 0.35, # Feature Similarity Loss 마진
    },
    
    # 3. 모델(Model) 기본 설정
    "MODEL": {
        "num_classes": 143, # 분류 개수 (모델 학습에만 필요한 파라미터. feature 추출에는 상관 X)
        "input_channel": 3,  # 기본 좌표(x,y,z)
        "num_points": 5000,  # 원본 입력 점 개수
        "feature_dim": 512    # 최종 특징 차원
    },
    
    # 4. Set Abstraction (RSConv) 계층별 파라미터 (4 -> 2 -> 1 분할 구조)
    # [npoint, radius, nsample, in_channel, out_channel, hidden_channel]

    "SA_PARAMS": [
        {'npoint': 512, 'radius': 0.10, 'nsample': 32, 'in_ch': 3,   'out_ch': 64,  'hid_ch': 16},  # SA1
        {'npoint': 512, 'radius': 0.20, 'nsample': 32, 'in_ch': 64,  'out_ch': 128, 'hid_ch': 32},  # SA2
        {'npoint': 512, 'radius': 0.40, 'nsample': 32, 'in_ch': 128, 'out_ch': 256, 'hid_ch': 64},  # SA3
        {'npoint': 256, 'radius': 0.60, 'nsample': 32, 'in_ch': 256, 'out_ch': 512, 'hid_ch': 128}  # SA4
    ],
    
    # 5. 매칭(Matching) 설정
    "MATCHING": {
        "threshold": 0.7
    }, 
    "DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu'
}