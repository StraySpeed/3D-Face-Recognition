import torch
import numpy as np
import tenseal as ts
import os
import time
from model.pointface import PointFaceNet 

class HEFaceRecognizer:
    def __init__(self, model_path, device='cpu'):
        # 1. 일반 모델 로드 (임베딩 추출)
        self.device = torch.device(device)
        self.model = PointFaceNet(num_classes=100).to(self.device)
        self.load_weights(model_path)
        self.model.eval()

        # 2. 동형 암호 컨텍스트 설정 (CKKS 스킴)
        # poly_modulus_degree: 8192 (보안 수준과 속도의 타협점)
        # coeff_mod_bit_sizes: 곱셈 깊이에 따라 설정
        self.ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.ctx.global_scale = 2**40
        self.ctx.generate_galois_keys() # 벡터 연산(회전 등)을 위해 필요
        
        self.encrypted_gallery = {} # {id: EncryptedVector}

    def load_weights(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)

    def preprocess(self, points):
        # (기존과 동일한 전처리 로직: 리샘플링 -> 정규화 -> 텐서)
        if len(points) > 5000:
            choice = np.random.choice(len(points), 5000, replace=False)
            points = points[choice, :]
        elif len(points) < 5000:
            choice = np.random.choice(len(points), 5000, replace=True)
            points = points[choice, :]
            
        points = points - np.mean(points, axis=0)
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / dist
        return torch.from_numpy(points.astype(np.float32)).transpose(0, 1).unsqueeze(0).to(self.device)

    def get_embedding(self, npy_path):
        """임베딩 추출 및 L2 정규화 (Plaintext)"""
        if not os.path.exists(npy_path): return None
        points = np.load(npy_path)[:, :3]
        tensor = self.preprocess(points)
        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy().flatten()
        
        # L2 Normalization (동형암호 내에서 나눗셈을 피하기 위해 미리 수행)
        norm = np.linalg.norm(embedding)
        return embedding / norm

    def register_gallery_encrypted(self, data_root):
        """갤러리 데이터를 암호화하여 저장"""
        self.encrypted_gallery = {}
        
        for person_id in os.listdir(data_root):
            person_dir = os.path.join(data_root, person_id)
            if not os.path.isdir(person_dir): continue
            
            files = [f for f in os.listdir(person_dir) if f.endswith('.npy')]
            if not files: continue
            
            # 1. 일반 임베딩 추출 (Plaintext)
            embedding = self.get_embedding(os.path.join(person_dir, files[0]))
            
            # 2. 동형 암호화 (CKKS Vector)
            # 비밀키(Secret Key)를 가진 클라이언트가 암호화해서 서버에 보낸다고 가정
            enc_vector = ts.ckks_vector(self.ctx, embedding)
            
            # 3. 저장 (암호화된 상태로 메모리에 유지)
            self.encrypted_gallery[person_id] = enc_vector
            print(f"[DEBUG] Registered (Encrypted): {person_id}")

    def recognize_encrypted(self, probe_path, threshold=0.5):
        """암호화된 상태에서 매칭 수행"""
        # 1. 프로브 임베딩 추출 및 암호화
        probe_vec = self.get_embedding(probe_path)
        if probe_vec is None: return "Error", 0.0
        
        enc_probe = ts.ckks_vector(self.ctx, probe_vec)
        
        best_score = -100.0
        best_id = "Unknown"
        
        print("\n[Secure Matching Start]")
        start_time = time.time()
        
        # 2. 갤러리와 동형 연산 (내적) 수행
        for person_id, enc_gallery_vec in self.encrypted_gallery.items():
            
            # Homomorphic Dot Product
            # (A . B) = Cosine Similarity (||A||=1, ||B||=1)
            enc_score = enc_probe.dot(enc_gallery_vec)
            
            # 3. 결과 복호화 (Decrypt)
            # 실제 서비스에서는 이 '암호화된 점수'를 클라이언트로 보내서 클라이언트가 복호화
            score = enc_score.decrypt()[0] 
            
            # 점수 보정 (CKKS 오차로 인해 1.0000001 등이 나올 수 있음)
            if score > 1.0: score = 1.0
            
            if score > best_score:
                best_score = score
                best_id = person_id
                
        elapsed = time.time() - start_time
        print(f"[DEBUG] Secure Matching Time: {elapsed:.4f} sec")
        
        if best_score < threshold:
            return "Unknown", best_score
        else:
            return best_id, best_score

# --- 실행 예시 ---
if __name__ == "__main__":
    # 설정
    MODEL_PATH = "./checkpoints/pointface_epoch_050.pth"
    GALLERY_DIR = "./dummy_dataset"
    TEST_FILE = "./test_scan.npy" # 테스트용 파일 생성 필요
    
    # 1. 보안 인식기 초기화
    secure_recognizer = HEFaceRecognizer(MODEL_PATH, device='mps')
    
    # 2. 갤러리 암호화 등록
    secure_recognizer.register_gallery_encrypted(GALLERY_DIR)
    
    # 3. 테스트 파일 생성 (임시)
    import numpy as np
    np.save(TEST_FILE, np.random.rand(5000, 3).astype(np.float32))

    # 4. 암호화 인식 수행
    identity, score = secure_recognizer.recognize_encrypted(TEST_FILE, threshold=0.4)
    print(f"Result: {identity} (Score: {score:.4f})")