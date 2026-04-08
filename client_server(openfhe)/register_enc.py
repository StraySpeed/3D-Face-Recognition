import os, torch, sys, pickle, tempfile
import openfhe
import numpy as np
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from model.pointface import PointFaceNet
from logger import get_logger
from config import CONFIG

def serialize_openfhe_to_bytes(obj):
    """ OpenFHE 객체를 Byte 단위로 직렬화 (안전한 임시 파일 활용) """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        temp_path = tmp.name
    openfhe.SerializeToFile(temp_path, obj, openfhe.BINARY)
    with open(temp_path, "rb") as f:
        data = f.read()
    os.remove(temp_path)
    return data

if __name__ =="__main__":
    # 1. 설정
    print = get_logger(name='register_enc').info
    MODEL_PATH = os.path.join(CONFIG["PATH"]["checkpoint_dir"], "pointface_epoch_200.pth")
    RAW_DATA_DIR = CONFIG["PATH"]["gallery_dir"]
    SAVE_DIR = CONFIG["PATH"]["gallery_storage_enc"]
    KEY_DIR = CONFIG["PATH"]["key_dir"]
    DEVICE = CONFIG["DEVICE"]

    # 2. 동형 암호 컨텍스트 설정 (CKKS 스킴)
    # modulus_degree: 32768 (128 bit 보안성)
    parameters = openfhe.CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(22) # 곱셈 깊이
    parameters.SetScalingModSize(33)      # global_scale = 2**33 과 동일
    parameters.SetFirstModSize(60)
    parameters.SetRingDim(32768)          # N = 32768
    parameters.SetBatchSize(16384)         # 패킹할 최대 슬롯 개수

    # 보안 검사 해제
    parameters.SetSecurityLevel(openfhe.HEStd_NotSet)

    ctx = openfhe.GenCryptoContext(parameters)
    ctx.Enable(openfhe.PKE)
    ctx.Enable(openfhe.KEYSWITCH)
    ctx.Enable(openfhe.LEVELEDSHE)
    ctx.Enable(openfhe.ADVANCEDSHE)

    # 3. 키 생성
    keys = ctx.KeyGen()
    ctx.EvalMultKeyGen(keys.secretKey)
    # 서버의 matmul 연산을 위한 회전 키 생성 (예: -256 ~ 256)
    rotations = []
    for i in range(15): # 2^0(1) 부터 2^14(16384) 까지 생성
        rotations.append(2 ** i)
        rotations.append(-(2 ** i))
        
    ctx.EvalRotateKeyGen(keys.secretKey, rotations)

    # 4. 키 및 컨텍스트 파일 저장 (OpenFHE는 개별 저장)
    if not os.path.exists(KEY_DIR):
        os.makedirs(KEY_DIR)
        
    openfhe.SerializeToFile(os.path.join(KEY_DIR, "cryptocontext.txt"), ctx, openfhe.BINARY)
    openfhe.SerializeToFile(os.path.join(KEY_DIR, "key-public.txt"), keys.publicKey, openfhe.BINARY)
    openfhe.SerializeToFile(os.path.join(KEY_DIR, "key-secret.txt"), keys.secretKey, openfhe.BINARY)
    ctx.SerializeEvalMultKey(os.path.join(KEY_DIR, "key-eval-mult.txt"), openfhe.BINARY)
    ctx.SerializeEvalAutomorphismKey(os.path.join(KEY_DIR, "key-eval-rot.txt"), openfhe.BINARY)
    print(f"[DEBUG] Saved: secret.context")

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
        
    # 평가 모드 전환 (Dropout, BatchNorm 고정) 
    model.eval()

    
    # 등록 루프
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
            points = PointFaceNet.morton_sort(points)
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
        ptxt = ctx.MakeCKKSPackedPlaintext(mean_embedding.tolist())
        ctxt = ctx.Encrypt(keys.publicKey, ptxt)
        enc_gallery_list.append(serialize_openfhe_to_bytes(ctxt))
        
        # pickle을 이용하여 256개 직렬화된 암호문 바이트 리스트를 한 번에 저장
        save_path = os.path.join(SAVE_DIR, f"{identity}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(enc_gallery_list, f)
            
        count += 1
        exit(0)
    print(f"Total: {count} IDs(Encrypted).")
    