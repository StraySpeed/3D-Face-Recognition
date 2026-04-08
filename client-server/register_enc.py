import os, torch, sys, pickle
import tenseal as ts
import numpy as np
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from model.pointface import PointFaceNet
from logger import get_logger
from config import CONFIG

if __name__ =="__main__":
    # 1. 설정
    print = get_logger(name='register_enc').info
    MODEL_PATH = os.path.join(CONFIG["PATH"]["checkpoint_dir"], "pointface_epoch_200.pth")
    RAW_DATA_DIR = CONFIG["PATH"]["gallery_dir"]
    SAVE_DIR = os.path.dirname(CONFIG["PATH"]["gallery_storage_enc"])
    DEVICE = CONFIG["DEVICE"]

    # 2. 동형 암호 컨텍스트 설정 (CKKS 스킴)
    # poly_modulus_degree: 32768 (128 bit 보안성)
    # coeff_mod_bit_sizes: 곱셈 깊이에 따라 설정
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=32768,
        #coeff_mod_bit_sizes=[60] + [40] * 18 + [60]
        coeff_mod_bit_sizes=[60] + [33] * 22 + [60]
    )
    ctx.global_scale = 2**33
    ctx.generate_relin_keys()
    ctx.generate_galois_keys()

    # 저장 폴더 생성
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # 모델 설정 및 초기화
    model = PointFaceNet(num_classes=CONFIG["MODEL"]["num_classes"]).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    # 2. 평가 모드 전환 (Dropout, BatchNorm 고정) 
    model.eval()


    # 3. Context 저장 (Server가 쓸 공개 키)
    # 편의상 비밀 키도 같이 넣어버림
    context_bytes = ctx.serialize(save_secret_key=True)
    with open(os.path.join(SAVE_DIR, "secret.context"), "wb") as f:
        f.write(context_bytes)
    print(f"Saved: secret.context")
    
    # 4. 등록 루프
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
            pre_data = PointFaceNet.preprocess(points, device=DEVICE)
            
            # 3. 특징 추출 (암호화 전, 평문 상태)
            with torch.no_grad():
                feature = model(pre_data).cpu().numpy().flatten()
                embeddings.append(feature)

        if len(embeddings) > 0:
            # 평균 계산
            mean_embedding = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(mean_embedding)
            if norm > 0:
                mean_embedding = mean_embedding / norm
        else:
            print(f"Warning: No valid files found for {identity}")
            continue

        # 1개로 저장 (plainText)
        # np.save(os.path.join(SAVE_DIR,"umbdb", f"{identity}.npy"), mean_embedding.astype(np.float32))
        
        # 128개로 저장
        enc_gallery_list = []
        for val in mean_embedding:
            # 단일 스칼라 값을 리스트 형태로 감싸서 암호화
            enc_vec = ts.ckks_vector(ctx, [val.item()])
            enc_gallery_list.append(enc_vec.serialize())
        
        # pickle을 이용하여 128개의 직렬화된 암호문 바이트 리스트를 한 번에 저장
        save_path = os.path.join(SAVE_DIR, "umbdb", f"{identity}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(enc_gallery_list, f)
            
        count += 1
        exit(0)
    print(f"Total: {count} IDs(Encrypted).")