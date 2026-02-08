import torch
import torch.nn.functional as F
import numpy as np
import os
from model.pointface import PointFaceNet 

class FaceRecognizer:
    def __init__(self, model_path, num_classes=100, device='cpu'):
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
        
        self.gallery_embeddings = [] # (embedding_vector, label_id)
        print(f"[DEBUG] Model loaded from {model_path}")

    def preprocess(self, points):
        """
        입력 데이터 전처리 (학습 때와 동일해야 함)
        points: (N, 3) numpy array
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

    def register_gallery(self, data_root):
        """
        Gallery(등록된 얼굴) 데이터베이스 구축
        폴더 구조: data_root/person_id/scan.npy
        """
        self.gallery_embeddings = []
        
        with torch.no_grad(): # 그래디언트 계산 불필요
            for person_id in os.listdir(data_root):
                person_dir = os.path.join(data_root, person_id)
                if not os.path.isdir(person_dir): continue
                
                # 각 사람의 첫 번째 파일만 등록한다고 가정 (여러 개 등록해서 평균 낼 수도 있음)
                files = [f for f in os.listdir(person_dir) if f.endswith('.npy')]
                if not files: continue
                
                file_path = os.path.join(person_dir, files[0])
                points = np.load(file_path)[:, :3]
                
                # 전처리 및 임베딩 추출
                input_tensor = self.preprocess(points)
                embedding = self.model(input_tensor) # (1, 512)
                
                # L2 정규화는 모델 내부 forward에서 이미 수행됨
                
                self.gallery_embeddings.append({
                    "id": person_id,
                    "embedding": embedding.cpu() # CPU로 내려서 저장
                })
                print(f"Registered: {person_id}")
        
        print(f"Done! {len(self.gallery_embeddings)} identities in gallery.")

    def recognize(self, npy_path, threshold=0.5):
        """
        새로운 얼굴(Probe) 인식
        """
        if not os.path.exists(npy_path):
            return "File Error", 0.0

        # 1. 입력 데이터 로드 및 임베딩 추출
        points = np.load(npy_path)[:, :3]
        input_tensor = self.preprocess(points)
        
        with torch.no_grad():
            probe_emb = self.model(input_tensor).cpu() # (1, 512)

        # 2. 매칭 (Gallery 전체와 비교)
        best_score = -1.0
        best_id = "Unknown"
        
        for record in self.gallery_embeddings:
            gallery_emb = record['embedding']
            gallery_id = record['id']
            
            # 코사인 유사도 계산 (이미 정규화된 벡터이므로 내적과 동일)
            # Similarity = A . B (Range: -1 ~ 1)
            similarity = torch.sum(probe_emb * gallery_emb).item()
            
            if similarity > best_score:
                best_score = similarity
                best_id = gallery_id
        
        # 3. 임계값(Threshold) 비교
        if best_score < threshold:
            return "Unknown", best_score
        else:
            return best_id, best_score

# --- 사용 예시 ---
if __name__ == "__main__":
    # 1. 설정
    MODEL_PATH = "./checkpoints/pointface_epoch_050.pth" # 학습된 모델 경로
    GALLERY_DIR = "./dummy_dataset" # 등록할 얼굴들이 있는 폴더
    TEST_FILE = "./test_scan.npy"   # 테스트할 새로운 파일
    
    # 2. 인식기 초기화 (Mac M1이라면 'mps', 아니면 'cuda' or 'cpu')
    recognizer = FaceRecognizer(MODEL_PATH, device='mps')
    
    # 3. 갤러리 등록 (한 번만 수행하면 됨)
    recognizer.register_gallery(GALLERY_DIR)
    
    # 4. 인식 수행
    # 테스트용 가짜 파일 생성
    np.save(TEST_FILE, np.random.rand(5000, 3).astype(np.float32))
    
    identity, score = recognizer.recognize(TEST_FILE, threshold=0.4)
    print(f"\n[Result] Identity: {identity}, Score: {score:.4f}")