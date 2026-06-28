# config2.py  (model2 전용)
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # 1. 데이터 및 경로 설정
    "PATH": {
        "data_root": os.path.join(BASE_DIR, "dataset/facescape/facescape_npy/train"),
        "checkpoint_dir": os.path.join(BASE_DIR, "checkpoints_facescape_v2"),   # model2 전용 체크포인트
        "gallery_dir": os.path.join(BASE_DIR, "dataset/facescape/facescape_npy/test"),
        "gallery_storage": os.path.join(BASE_DIR, "gallery_storage/facescape"),
        "gallery_storage_enc": os.path.join(BASE_DIR, "gallery_storage_enc/facescape"),
        "secret.context": os.path.join(BASE_DIR, "gallery_storage_enc/secret.context"),
        "key_dir": os.path.join(BASE_DIR, "gallery_storage_enc/keys"),
    },

    # 2. 학습(Training) 하이퍼파라미터
    "TRAIN": {
        "epochs": 500,
        "batch_size": 64,
        "learning_rate": 0.001,
        "num_workers": 4,
        "lambda_factor": 1.0,
        "margin": 0.35,
        "arcface_margin_warmup_epochs": 500,
    },

    # 3. 모델(Model) 기본 설정
    "MODEL": {
        "num_classes": 700,
        "input_channel": 3,
        "num_points": 1024,
        "feature_dim": 512,
    },

    # 4. 매칭(Matching) 설정
    "MATCHING": {
        "threshold": 0.88
    },
    "DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu'
}
