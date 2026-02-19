import torch
import numpy as np
import tenseal as ts
import os, glob, time
from model.pointface import PointFaceNet 
from logger import get_logger

class HEFaceRecognizer:
    def __init__(self, model_path, num_classes=143, device='cpu'):
        self.device = torch.device(device)
        
        # 1. 모델 초기화 및 가중치 로드
        # Inference 시에는 num_classes가 중요하지 않지만 구조를 맞추기 위해 넣음
        self.model = PointFaceNet(num_classes=num_classes).to(self.device)
        
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

        self.encrypted_gallery = {} 
        print(f"Model loaded from {model_path}")

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

    def get_embedding(self, npy_path):
        if not os.path.exists(npy_path): return None
        try:
            points = np.load(npy_path)[:, :3] # XYZ만 사용
            tensor = self.preprocess(points)
            with torch.no_grad():
                embedding = self.model(tensor).cpu().numpy().flatten()
            
            # L2 Normalize (개별 임베딩도 정규화)
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            print(f"Error processing {npy_path}: {e}")
            return None

    def register_gallery_encrypted(self, data_root):
        """
        갤러리 등록 (Average Embedding 방식)
        폴더 내의 모든 파일을 읽어서 임베딩의 평균을 구함
        """
        self.encrypted_gallery = {}
        
        # 폴더 목록 가져오기
        identities = sorted([d for d in os.listdir(data_root) 
                             if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')])
        
        print(f"Registering {len(identities)} identities (Averaging mode)...")

        for identity in identities:
            person_dir = os.path.join(data_root, identity)
            embeddings = []
            
            # 1. 해당 ID 폴더 내의 모든 파일 순회
            files = [f for f in os.listdir(person_dir) if f.endswith('.npy')]
            
            for filename in files:
                file_path = os.path.join(person_dir, filename)
                
                # 임베딩 추출
                emb = self.get_embedding(file_path)
                
                if emb is not None:
                    embeddings.append(emb)
            
            # 2. 유효한 임베딩이 하나라도 있으면 평균 계산
            if len(embeddings) > 0:
                # (N, 512) -> (512,) 평균 계산
                mean_embedding = np.mean(embeddings, axis=0)
                
                # 평균을 내면 벡터 길이가 1보다 작아지므로 다시 정규화해야 함
                norm = np.linalg.norm(mean_embedding)
                if norm > 0:
                    mean_embedding = mean_embedding / norm
            else:
                print(f"Warning: No valid files found for {identity}")

            # 2. 동형 암호화 (CKKS Vector)
            # 비밀키(Secret Key)를 가진 클라이언트가 암호화해서 서버에 보낸다고 가정
            enc_vector = ts.ckks_vector(self.ctx, mean_embedding)
            
            # 3. 저장 (암호화된 상태로 메모리에 유지)
            self.encrypted_gallery[identity] = enc_vector

        print(f"Total: {len(self.encrypted_gallery)} IDs(Encrypted).")

    def recognize_encrypted(self, npy_path, threshold=0.5):
        """암호화된 상태에서 매칭 수행"""
        if not os.path.exists(npy_path):
            return "File Error", 0.0
        
        # 1. 입력 데이터 로드 및 임베딩 추출
        points = np.load(npy_path)[:, :3]
        print("[1:N Secure Recognization Start]")
        start_time = time.time()
        input_tensor = self.preprocess(points)
        pre_time = time.time()
        print(f"0. Preprocess Time: {pre_time - start_time:.4f}")

        with torch.no_grad():
            probe_emb = self.model(input_tensor).cpu().numpy().flatten()
        probe_emb = probe_emb / np.linalg.norm(probe_emb)

        emb_time = time.time()
        print(f"1. Embedding Time: {emb_time - pre_time:.4f}")
        
        # 암호화 로직 추가
        enc_probe = ts.ckks_vector(self.ctx, probe_emb)
        enc_time = time.time()
        print(f"2. Encryption Time: {enc_time - emb_time:.4f}")

        # 2. 매칭 (Gallery 전체와 비교)
        best_score = -1.0
        best_id = "Unknown"
        
        sim_start_time = time.time()
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

        end_time = time.time()
        print(f"3. 1:N Matching Time: {end_time - sim_start_time:.4f}")
        print(f"4. Total Matching Time: {end_time - start_time:.4f}")
        print(f"[1:N Secure Recognization Finished]")
        
         # 3. 임계값(Threshold) 비교
        if best_score < threshold:
            return "Unknown", best_score
        else:
            return best_id, best_score

    def recognize_encrypted_id(self, npy_path, id, threshold=0.5):
        """암호화된 상태에서 매칭 수행"""
        if not os.path.exists(npy_path):
            return "File Error", 0.0
        
        # 1. 입력 데이터 로드 및 임베딩 추출
        points = np.load(npy_path)[:, :3]

        print("[1:1 Secure Recognization Start]")
        start_time = time.time()
        input_tensor = self.preprocess(points)
        pre_time = time.time()
        print(f"0. Preprocess Time: {pre_time - start_time:.4f}")

        with torch.no_grad():
            probe_emb = self.model(input_tensor).cpu().numpy().flatten()
        probe_emb = probe_emb / np.linalg.norm(probe_emb)

        emb_time = time.time()
        print(f"1. Embedding Time: {emb_time - pre_time:.4f}")
        
        # 암호화 로직 추가
        enc_probe = ts.ckks_vector(self.ctx, probe_emb)
        enc_time = time.time()
        print(f"2. Encryption Time: {enc_time - emb_time:.4f}")

        # 2. 매칭 (Gallery 전체와 비교)
        best_score = -1.0
        best_id = "Unknown"
        
        sim_start_time = time.time()
        # 2. 갤러리와 동형 연산 (내적) 수행
        person_id = id
        enc_gallery_vec = self.encrypted_gallery.get(id)
            
        # Homomorphic Dot Product
        # (A . B) = Cosine Similarity (||A||=1, ||B||=1)
        enc_score = enc_probe.dot(enc_gallery_vec)
        end_time = time.time()
        print(f"3. 1:1 Matching Time: {end_time - sim_start_time:.4f}")
        print(f"4. Total Matching Time: {end_time - start_time:.4f}")
        print(f"[1:1 Secure Recognization Finished]")

        # 3. 결과 복호화 (Decrypt)
        # 실제 서비스에서는 이 '암호화된 점수'를 클라이언트로 보내서 클라이언트가 복호화
        score = enc_score.decrypt()[0] 
        
        # 점수 보정 (CKKS 오차로 인해 1.0000001 등이 나올 수 있음)
        if score > 1.0: score = 1.0
        
        if score > best_score:
            best_score = score
            best_id = person_id
        
        # 3. 임계값(Threshold) 비교
        if best_score < threshold:
            return "Unknown", best_score
        else:
            return best_id, best_score

    def save_gallery_individual(self, save_dir="./gallery_storage"):
        """
        각 Identity의 임베딩을 개별 .ts 파일로 저장
        """
        if not self.encrypted_gallery:
            print("Warning: Gallery is empty. Nothing to save.")
            return

        # 저장 폴더 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        print(f"Saving individual embeddings to '{save_dir}'...")

        # 여기서는 복호화를 할 수 있도록 sk를 같이 저장
        #context_bytes = self.ctx.serialize(save_secret_key=False)
        context_bytes = self.ctx.serialize(save_secret_key=True)
        with open(os.path.join(save_dir, "secret.context"), "wb") as f:
            f.write(context_bytes)
        print(f"Saved: secret.context")

        count = 0
        for identity, embedding in self.encrypted_gallery.items():
            # 파일명: id_XXX.npy
            serialized_emb = embedding.serialize()
            file_path = os.path.join(save_dir, f"{identity}.ts")
            with open(file_path, 'wb') as f:
                f.write(serialized_emb)
            count += 1
            
        print(f"Saved {count} files successfully.")

    def load_gallery_individual(self, load_dir="./gallery_storage"):
        """
        폴더 내의 모든 .ts 파일을 읽어서 갤러리로 로드
        """
        if not os.path.exists(load_dir):
            print(f"Error: Directory '{load_dir}' not found.")
            return False

        # 1. Context 로드
        context_path = os.path.join(load_dir, "secret.context")
        if not os.path.exists(context_path):
            print("Error: 'secret.context' file not found in directory.")
            return False
        try:
            with open(context_path, "rb") as f:
                context_bytes = f.read()
                self.ctx = ts.context_from(context_bytes, n_threads=4)
            print("Context loaded successfully.")
        except Exception as e:
            print(f"Error loading context: {e}")
            return None
        
        ts_files = glob.glob(os.path.join(load_dir, "*.ts"))
        
        if not ts_files:
            print("Warning: No .ts files found in the directory.")
            return False
            
        print(f"Loading embeddings from '{load_dir}'...")
        
        self.encrypted_gallery = {}
        for file_path in ts_files:
            # 파일명에서 확장자 제거하여 ID로 사용 (예: id_001.ts -> id_001)
            filename = os.path.basename(file_path)
            identity = os.path.splitext(filename)[0]
            
            # 로드
            try:
                with open(file_path, 'rb') as f:
                    vec_bytes = f.read()
                
                # 로드한 Context를 이용해 벡터 복원
                vec = ts.ckks_vector_from(self.ctx, vec_bytes)
                self.encrypted_gallery[identity] = vec
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
            
        print(f"Loaded {len(self.encrypted_gallery)} identities.")
        return True


if __name__ == "__main__":
    # 1. 설정
    print = get_logger(name='matching_he').info
    MODEL_PATH = "./checkpoints/pointface_epoch_200.pth" # 학습된 모델 경로
    GALLERY_DIR = "./dataset_matching/umbdb_unpreprocessed" # 등록할 얼굴들이 있는 폴더

    # 2. 인식기 초기화
    secure_recognizer = HEFaceRecognizer(MODEL_PATH, device='cpu')
    
    # 3. 갤러리 암호화 등록
    # secure_recognizer.register_gallery_encrypted(GALLERY_DIR)
    # secure_recognizer.save_gallery_individual("./gallery_storage/umbdb_enc2")
    # 저장된 데이터가 있으면 로드
    secure_recognizer.load_gallery_individual("./gallery_storage/umbdb_enc")
    
    # 4. 인식 수행
    identities = sorted([d for d in os.listdir(GALLERY_DIR) if os.path.isdir(os.path.join(GALLERY_DIR, d)) and not d.startswith('.')])
    correct = 0; wrong = 0; total = 0
    for id in identities:
        person_dir = os.path.join(GALLERY_DIR, id)
        for f in os.listdir(person_dir):
            total += 1
            identity_file = os.path.join(person_dir, f)
            print(f"[Matching] Identity: {id}")

            # 1:N Matching
            #identity, score = secure_recognizer.recognize_encrypted(identity_file, threshold=0.7)
            
            # 1:1 Matching
            identity, score = secure_recognizer.recognize_encrypted_id(identity_file, id, threshold=0.7)
            print(f"[Result] Identity: {identity} (Score: {score:.4f})")

            if id == identity:
                correct += 1
            else:
                wrong += 1

    print(f"[Result] Total : {total}, Correct : {correct} ({correct} / {total}), Wrong : {wrong} ({wrong} / {total})")
    