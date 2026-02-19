import torch, time
import numpy as np
import tenseal as ts
from he_utils import ClientBody, load_server_weights
import os, sys
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from logger import get_logger

class PointFaceClient:
    def __init__(self, model_path, device='cpu', logger=None):
        self.device = torch.device(device)

        if logger:
            self.print = logger
        else:
            self.print = get_logger("matching_client").info

        # 1. 모델 준비
        # 전체 모델을 로드한 뒤 앞부분만 추출
        _, _, full_model = load_server_weights(model_path, device)
        self.extractor = ClientBody(full_model).to(self.device)
        self.extractor.eval()
        
        # 2. TenSEAL Context 생성 (비밀키 보유)
        # 지금 이 Context로는 복호화가 안됨
        # 사이즈를 더 늘려야 함
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )

        self.context.global_scale = 2**40
        self.context.generate_galois_keys()
        
    def get_public_context(self, save_secret_key=False):
        """서버에게 줄 공개 키만 포함된 컨텍스트"""
        return self.context.serialize(save_secret_key=save_secret_key)
    
    def set_public_context(self, context_path):
        """ 외부에서 공개 키를 가져오기 """
        try:
            with open(context_path, "rb") as f:
                context_bytes = f.read()
                self.context = ts.context_from(context_bytes, n_threads=4)
            self.print("Context loaded successfully.")
        except Exception as e:
            self.print(f"Error loading context: {e}")
            return None

    def preprocess(self, points):
        """입력 전처리 (Sampling + Norm)"""
        # 1. 리샘플링 (5000개)
        if len(points) > 5000:
            choice = np.random.choice(len(points), 5000, replace=False)
            points = points[choice, :]
        elif len(points) < 5000:
            choice = np.random.choice(len(points), 5000, replace=True)
            points = points[choice, :]
            
        # 2. 정규화 (Center & Scale)
        points = points - np.mean(points, axis=0)
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / dist
        
        # 3. 텐서 변환 (1, 3, 5000) - 배치 차원 추가
        tensor = torch.from_numpy(points.astype(np.float32)).transpose(0, 1)
        return tensor.unsqueeze(0).to(self.device)

    def extract_and_encrypt(self, npy_path):
        """
        Feature 추출 -> 암호화

        :Return: CKKSVector (serialized)
        """

        points = np.load(npy_path)[:, :3]
        self.print("[Client Side #1 Start]")
        start_time = time.time()
        input_tensor = self.preprocess(points)
        pre_time = time.time()
        self.print(f"0. Preprocess Time: {pre_time - start_time:.4f}")
        
        with torch.no_grad():
            global_feature = self.extractor(input_tensor).cpu().flatten().numpy()
        global_feature = global_feature * 1000.0
        feat_time = time.time()
        self.print(f"1. Global Feature Time: {feat_time - pre_time:.4f}")

        # 2. Encryption
        enc_vec = ts.ckks_vector(self.context, global_feature)
        end_time = time.time()
        self.print(f"2. Encryption Time: {end_time - feat_time:.4f}")
        self.print(f"3. Client #1 Total Time: {end_time - start_time:.4f}")
        self.print("[Client Side #1 Finished]")
        return enc_vec.serialize()

    def decrypt_result(self, enc_dot_bytes, enc_norm_a_bytes, enc_norm_b_bytes):
        """
        서버에서 온 암호화된 결과값들을 복호화하여 코사인 유사도 계산
        """
        self.print("[Client Side #2 Start]")
        start_time = time.time()
        # 복호화
        dot_prod = ts.ckks_vector_from(self.context, enc_dot_bytes).decrypt()[0]
        norm_sq_a = ts.ckks_vector_from(self.context, enc_norm_a_bytes).decrypt()[0]
        norm_sq_b = ts.ckks_vector_from(self.context, enc_norm_b_bytes).decrypt()[0]
        pre_time = time.time()
        self.print(f"0. Decryption Time: {pre_time - start_time:.4f}")
        self.print(f"DEBUG RAW: Dot={dot_prod:.6f}, NormA={norm_sq_a:.6f}, NormB={norm_sq_b:.6f}")
        # norm 보정
        if norm_sq_a < 0: norm_sq_a = 1e-10
        if norm_sq_b < 0: norm_sq_b = 1e-10

        # 코사인 유사도 계산
        # L2 Norm = sqrt(sum(x^2))
        norm_a = np.sqrt(norm_sq_a)
        norm_b = np.sqrt(norm_sq_b)

        similarity = dot_prod / (norm_a * norm_b)
        end_time = time.time()
        self.print(f"1. Similarity Time: {end_time - pre_time:.4f}")
        self.print(f"2. Client #2 Total Time: {end_time - start_time:.4f}")
        self.print("[Client Side #2 Finished]")
        return similarity