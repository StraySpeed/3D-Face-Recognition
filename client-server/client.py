import torch, time
import numpy as np
import tenseal as ts
import os, sys
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from logger import get_logger
from model.pointface import PointFaceNet

class PointFaceClient:
    def __init__(self, model_path, device='cpu', logger=None):
        self.device = torch.device(device)

        if logger:
            self.print = logger
        else:
            self.print = get_logger("matching_client").info

        # 1. 모델 준비
        # 전체 모델을 로드
        self.model = PointFaceNet(num_classes=143).to(self.device)

        # 가중치 파일 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        # 2. 평가 모드 전환 (Dropout, BatchNorm 고정) 
        self.model.eval()
        
        # 3. 동형 암호 컨텍스트 설정 (CKKS 스킴)
        # poly_modulus_degree: 8192
        # coeff_mod_bit_sizes: 곱셈 깊이에 따라 설정
        self.ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.ctx.global_scale = 2**40
        self.ctx.generate_galois_keys() # 벡터 연산(회전 등)을 위해 필요
        
    def get_public_context(self, save_secret_key=False):
        """서버에게 줄 공개 키만 포함된 컨텍스트"""
        return self.ctx.serialize(save_secret_key=save_secret_key)
    
    def set_public_context(self, context_path):
        """ 외부에서 공개 키를 가져오기 """
        try:
            with open(context_path, "rb") as f:
                context_bytes = f.read()
                self.ctx = ts.context_from(context_bytes, n_threads=4)
            self.print("Context loaded successfully.")
        except Exception as e:
            self.print(f"Error loading context: {e}")
            return None

    def preprocess(self, points):
        """
        입력 데이터 전처리 (학습 때와 동일해야 함)
        :param points: (N, 3) numpy array
        """
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

    def get_embedding_and_encrypt(self, npy_path):
        """
        임베딩을 추출한 후 암호화해서 제공

        :param npy_path: 파일
        :return: CKKSVector (serialized)
        """
        if not os.path.exists(npy_path): return None
        try:
            points = np.load(npy_path)[:, :3] # XYZ만 사용
            self.print("[Client Side #1 Start]")
            start_time = time.time()
            tensor = self.preprocess(points)
            pre_time = time.time()
            self.print(f"0. Preprocess Time: {pre_time - start_time:.4f}")

            with torch.no_grad():
                embedding = self.model(tensor).cpu().numpy().flatten()
            feature = embedding / np.linalg.norm(embedding)
            feat_time = time.time()
            self.print(f"1. Feature Extract Time: {feat_time - pre_time:.4f}")

            # Encryption
            enc_vec = ts.ckks_vector(self.ctx, feature)
            end_time = time.time()
            self.print(f"2. Encryption Time: {end_time - feat_time:.4f}")
            self.print(f"3. Client #1 Total Time: {end_time - start_time:.4f}")
            self.print("[Client Side #1 Finished]")
            
            return enc_vec.serialize()
        except Exception as e:
            print(f"Error processing {npy_path}: {e}")
            return None


    def decrypt_result(self, enc_score):
        """
        서버에서 온 암호화된 점수를 복호화해서 인증 여부만을 제공
        """
        self.print("[Client Side #2 Start]")
        start_time = time.time()
        # 복호화
        score = ts.ckks_vector_from(self.ctx, enc_score).decrypt()[0]
        end_time = time.time()
        self.print(f"1. Decryption Time: {end_time - start_time:.4f}")
        self.print(f"Client #2 Total Time: {end_time - start_time:.4f}")
        self.print("[Client Side #2 Finished]")
        return True if score > 0 else False