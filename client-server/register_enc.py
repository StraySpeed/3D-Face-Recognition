import os, torch
import tenseal as ts
import numpy as np
from client import PointFaceClient
from server import PointFaceServer
from logger import get_logger

if __name__ =="__main__":
    # 1. 설정
    print = get_logger(name='register_enc').info
    MODEL_PATH = "./checkpoints/pointface_epoch_200.pth"
    RAW_DATA_DIR = "./dataset_matching/umbdb_unpreprocessed"
    SAVE_DIR = "./gallery_storage/umbdb_enc_30" # 저장될 폴더

    # 저장 폴더 생성
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    # 2. 클라이언트 초기화 (2^30 스케일)
    client = PointFaceClient(MODEL_PATH, device='cpu', logger=print)
    
    # 3. Context 저장 (Server가 쓸 공개 키)
    # 편의상 비밀 키도 같이 넣어버림
    with open(os.path.join(SAVE_DIR, "secret.context"), "wb") as f:
        f.write(client.get_public_context(True))
    print(f"Saved: secret.context")

    # 4. 서버 초기화 (등록용)
    server = PointFaceServer(MODEL_PATH, client.get_public_context(), logger=print)
    
    # 5. 재등록 루프
    identities = sorted([d for d in os.listdir(RAW_DATA_DIR) 
                         if os.path.isdir(os.path.join(RAW_DATA_DIR, d))])
    
    print(f"Registering {len(identities)} identities (Averaging mode)...")
    
    count = 0
    for identity in identities:
        person_dir = os.path.join(RAW_DATA_DIR, identity)
        embeddings = []

        # 1. 해당 ID 폴더 내의 모든 파일 순회
        files = [f for f in os.listdir(person_dir) if f.endswith('.npy')]
        
        for filename in files:
            file_path = os.path.join(person_dir, filename)
            # 1. 파일 로드
            points = np.load(file_path)[:, :3]
            
            # 2. 전처리 & 텐서 변환 (Client 내부 메서드 활용)
            tensor = client.preprocess(points)
            
            # 3. 특징 추출 (암호화 전, 평문 상태)
            with torch.no_grad():
                # (1, 512) -> (512,)
                feature = client.extractor(tensor).cpu().detach().numpy().flatten()
                embeddings.append(feature)

        if len(embeddings) > 0:
            # (N, 512) -> (512,) 평균 계산
            mean_embedding = np.mean(embeddings, axis=0)

        else:
            print(f"Warning: No valid files found for {identity}")
            continue

        enc_vec = ts.ckks_vector(client.context, mean_embedding)
        enc_data = enc_vec.serialize()

        # FC Layer 통과 (서버 연산)
        # 평균된 특징에 가중치를 곱하므로, 평균 임베딩이 됨
        enc_emb = server.process_feature(enc_data)
        
        # Norm Squared 계산
        enc_norm_sq = enc_emb.dot(enc_emb)
        
        # 개별 파일로 저장
        vec_path = os.path.join(SAVE_DIR, f"{identity}_vec.ts")
        norm_path = os.path.join(SAVE_DIR, f"{identity}_norm.ts")
        
        with open(vec_path, "wb") as f:
            f.write(enc_emb.serialize())
        with open(norm_path, "wb") as f:
            f.write(enc_norm_sq.serialize())
            
        count += 1
    print(f"Total: {count} IDs(Encrypted).")