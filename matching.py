import torch
import numpy as np
import os, glob, time
from model.pointface import PointFaceNet
from logger import get_logger
from config import CONFIG

class FaceRecognizer:
    def __init__(self, model_path, num_classes=CONFIG["MODEL"]['num_classes'], device='cpu'):
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
        
        self.gallery_dict = {} 
        print(f"Model loaded from {model_path}")

    def preprocess(self, points):
        """
        입력 데이터 전처리 (학습 때와 동일해야 함)
        
        :param points: (N, 3) numpy array
        """
        # 1. 리샘플링 (cnt 개)
        cnt = CONFIG["MODEL"]['num_points']
        if len(points) > cnt:
            choice = np.random.choice(len(points), cnt, replace=False)
            points = points[choice, :]
        elif len(points) < cnt:
            choice = np.random.choice(len(points), cnt, replace=True)
            points = points[choice, :]
            
        # 2. 정규화 (Center & Scale)
        points = points - np.mean(points, axis=0)
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / dist
        
        # 3. 텐서 변환 (1, 3, cnt) - 배치 차원 추가
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
        
    def register_gallery(self, data_root):
        """
        갤러리 등록 (Average Embedding 방식)
        폴더 내의 모든 파일을 읽어서 임베딩의 평균을 구함
        """
        self.gallery_dict = {}
        
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
                    
                self.gallery_dict[identity] = mean_embedding
            else:
                print(f"Warning: No valid files found for {identity}")
                
        print(f"Total: {len(self.gallery_dict)} IDs.")

    def recognize(self, npy_path, threshold=0.5):
        """
        새로운 얼굴(Probe) 인식
        """
        if not os.path.exists(npy_path):
            return "File Error", 0.0

        # 1. 입력 데이터 로드 및 임베딩 추출
        points = np.load(npy_path)[:, :3]
        print("[1:N Recognization Start]")
        start_time = time.time()
        input_tensor = self.preprocess(points)
        pre_time = time.time()
        print(f"0. Preprocess Time: {pre_time - start_time:.4f}")
        
        with torch.no_grad():
            probe_emb = self.model(input_tensor).cpu() # (1, 512)

        emb_time = time.time()
        print(f"1. Embedding Time: {emb_time - pre_time:.4f}")

        # 2. 매칭 (Gallery 전체와 비교)
        best_score = -1.0
        best_id = "Unknown"
        
        sim_start_time = time.time()
        for gallery_id, gallery_emb in self.gallery_dict.items():
            
            # 코사인 유사도 계산 (이미 정규화된 벡터이므로 내적과 동일)
            # Similarity = A . B (Range: -1 ~ 1)
            similarity = torch.sum(probe_emb * gallery_emb).item()
            if similarity > best_score:
                best_score = similarity
                best_id = gallery_id

        end_time = time.time()
        print(f"2. 1:N Matching Time: {end_time - sim_start_time:.4f}")
        print(f"3. Total Matching Time: {end_time - start_time:.4f}")
        print(f"[1:N Matching Finished]")

        # 3. 임계값(Threshold) 비교
        if best_score < threshold:
            return "Unknown", best_score
        else:
            return best_id, best_score

    def recognize_id(self, npy_path, id, threshold=0.5):
        """
        새로운 얼굴(Probe) 인식
        """
        if not os.path.exists(npy_path):
            return "File Error", 0.0

        # 1. 입력 데이터 로드 및 임베딩 추출
        points = np.load(npy_path)[:, :3]

        print("[1:1 Recognization Start]")
        start_time = time.time()
        input_tensor = self.preprocess(points)
        pre_time = time.time()
        print(f"0. Preprocess Time: {pre_time - start_time:.4f}")

        with torch.no_grad():
            probe_emb = self.model(input_tensor).cpu() # (1, 512)

        emb_time = time.time()
        print(f"1. Embedding Time: {emb_time - pre_time:.4f}")

        # 2. 매칭 (Gallery 전체와 비교)
        best_score = -1.0
        best_id = "Unknown"
        
        gallery_id = id
        gallery_emb = self.gallery_dict.get(gallery_id, None)

        # 코사인 유사도 계산 (이미 정규화된 벡터이므로 내적과 동일)
        # Similarity = A . B (Range: -1 ~ 1)
        sim_start_time = time.time()
        similarity = torch.sum(probe_emb * gallery_emb).item()
        if similarity > best_score:
            best_score = similarity
            best_id = gallery_id
        end_time = time.time()
        print(f"2. 1:1 Matching Time: {end_time - sim_start_time:.4f}")
        print(f"3. Total Matching Time: {end_time - start_time:.4f}")
        print(f"[1:1 Matching Finished]")
        
        # 3. 임계값(Threshold) 비교
        if best_score < threshold:
            return "Unknown", best_score
        else:
            return best_id, best_score
        
    def save_gallery_individual(self, save_dir="./gallery_storage"):
        """
        각 Identity의 임베딩을 개별 .npy 파일로 저장
        """
        if not self.gallery_dict:
            print("Warning: Gallery is empty. Nothing to save.")
            return

        # 저장 폴더 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        print(f"Saving individual embeddings to '{save_dir}'...")
        
        count = 0
        for identity, embedding in self.gallery_dict.items():
            # 파일명: id_XXX.npy
            file_path = os.path.join(save_dir, f"{identity}.npy")
            np.save(file_path, embedding)
            count += 1
            
        print(f"Saved {count} files successfully.")

    def load_gallery_individual(self, load_dir="./gallery_storage"):
        """
        폴더 내의 모든 .npy 파일을 읽어서 갤러리로 로드
        """
        if not os.path.exists(load_dir):
            print(f"Error: Directory '{load_dir}' not found.")
            return False
            
        npy_files = glob.glob(os.path.join(load_dir, "*.npy"))
        
        if not npy_files:
            print("Warning: No .npy files found in the directory.")
            return False
            
        print(f"Loading embeddings from '{load_dir}'...")
        
        self.gallery_dict = {}
        for file_path in npy_files:
            # 파일명에서 확장자 제거하여 ID로 사용 (예: id_001.npy -> id_001)
            filename = os.path.basename(file_path)
            identity = os.path.splitext(filename)[0]
            
            # 로드
            embedding = np.load(file_path)
            self.gallery_dict[identity] = embedding
            
        print(f"Loaded {len(self.gallery_dict)} identities.")
        return True

if __name__ == "__main__":
    # 1. 설정
    print = get_logger(name='matching').info
    MODEL_PATH = os.path.join(CONFIG["PATH"]["checkpoint_dir"], "pointface_epoch_200.pth")  # 학습된 모델 경로
    DATABASE_DIR = CONFIG["PATH"]["gallery_storage"] # 저장된 데이터들
    MATCHING_DIR = CONFIG["PATH"]["gallery_dir"] # 인식할 얼굴들이 있는 폴더
    THRESHOLD = CONFIG["MATCHING"]["threshold"]

    # 2. 인식기 초기화
    recognizer = FaceRecognizer(MODEL_PATH, device='cuda')
    
    # 3. 갤러리 등록
    #recognizer.register_gallery(MATCHING_DIR)
    # 저장된 데이터가 있으면 로드
    recognizer.load_gallery_individual(DATABASE_DIR)

    # 4. 인식 수행    
    identities = sorted([d for d in os.listdir(MATCHING_DIR) if os.path.isdir(os.path.join(MATCHING_DIR, d)) and not d.startswith('.')])
    correct = 0; wrong = 0; total = 0; FAR = 0; FRR = 0
    for id in identities:
        person_dir = os.path.join(MATCHING_DIR, id)
        for f in os.listdir(person_dir):
            total += 1
            identity_file = os.path.join(person_dir, f)
            print(f"[Matching] Identity: {id}")

            # 1:N Matching
            #identity, score = recognizer.recognize(identity_file, threshold=THRESHOLD)

            # 1:1 Matching
            identity, score =recognizer.recognize_id(identity_file, id, threshold=THRESHOLD)

            print(f"[Result] Identity: {identity} (Score: {score:.4f})")

            if id == identity:
                correct += 1
            else:
                wrong += 1

    print(f"[Result] Total : {total}, Correct : {correct} ({correct} / {total}), Wrong : {wrong} ({wrong} / {total})")
    #recognizer.save_gallery_individual("./gallery_storage/umbdb")