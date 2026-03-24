import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model.pointface import PointFaceNet
from biohashing.biohasher import BioHasher
from config import CONFIG

class HamdisEvaluator:
    def __init__(self, model_path, device=CONFIG["DEVICE"]):
        self.device = torch.device(device)
        
        # 모델 & Hasher 로드
        self.model = PointFaceNet(num_classes=CONFIG["MODEL"]["num_classes"]).to(self.device)
        self.hasher = BioHasher(input_dim=CONFIG["MODEL"]["feature_dim"], code_length=CONFIG["MODEL"]["feature_dim"])
        
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.gallery = {} # {id: biocode}

    def preprocess(self, points):
        """ 전처리: Sampling -> Normalize -> Tensor """
        cnt = CONFIG["MODEL"]["num_points"]
        if len(points) > cnt:
            choice = np.random.choice(len(points), cnt, replace=False)
            points = points[choice, :]
        elif len(points) < cnt:
            choice = np.random.choice(len(points), cnt, replace=True)
            points = points[choice, :]
            
        points = points - np.mean(points, axis=0)
        dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / dist
        
        tensor = torch.from_numpy(points.astype(np.float32)).transpose(0, 1)
        return tensor.unsqueeze(0).to(self.device)

    def get_biocode_from_file(self, npy_path, seed):
        """ 파일 -> 특징 추출 -> BioCode 생성 """
        if not os.path.exists(npy_path): return None
        points = np.load(npy_path)[:, :3]
        tensor = self.preprocess(points)
        
        with torch.no_grad():
            emb = self.model(tensor).cpu().detach().numpy().flatten() # Flatten 필수!
            
        return self.hasher.get_biocode(emb, seed)

    def load_gallery(self, gallery_dir):
        """ 저장된 갤러리(.npy) 로드 """
        print(f"Loading gallery from {gallery_dir}...")
        for f in os.listdir(gallery_dir):
            if f.endswith('.npy'):
                identity = os.path.splitext(f)[0]
                path = os.path.join(gallery_dir, f)
                self.gallery[identity] = np.load(path)
        print(f"Loaded {len(self.gallery)} identities.")

    def calculate_scores(self, test_root):
        """
        Genuine(본인) 및 Imposter(타인) 점수 리스트 생성
        """
        genuine_scores = []
        imposter_scores = []
        
        identities = sorted([d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))])
        
        print("Calculating scores...")
        for probe_id in identities:
            person_dir = os.path.join(test_root, probe_id)
            seed = int(probe_id[-4:])
            files = [f for f in os.listdir(person_dir) if f.endswith('.npy')]
            
            for fname in files:
                fpath = os.path.join(person_dir, fname)
                
                # Probe의 BioCode 생성
                probe_code = self.get_biocode_from_file(fpath, seed)
                if probe_code is None: continue
                
                # 갤러리의 모든 사람과 비교 (N:N 매칭)
                for gallery_id, gallery_code in self.gallery.items():
                    # 해밍 거리 계산
                    dist = self.hasher.hamming_distance(probe_code, gallery_code)
                    
                    if probe_id == gallery_id:
                        genuine_scores.append(dist)  # 본인
                    else:
                        imposter_scores.append(dist) # 타인
                        
        return genuine_scores, imposter_scores

    def evaluate_metrics(self, genuine_scores, imposter_scores):
        """ FAR, FRR, EER 계산 및 시각화 """
        thresholds = np.arange(0.0, 1.0, 0.005) # 0.0 ~ 1.0 까지 0.005 단위
        fars = []
        frrs = []
        
        print(f"\n[Statistics]")
        print(f"Genuine Samples: {len(genuine_scores)}")
        print(f"Imposter Samples: {len(imposter_scores)}")
        print(f"Max Genuine Score: {np.max(genuine_scores):.4f}")
        print(f"Max Imposter Score: {np.max(imposter_scores):.4f}")

        min_diff = 1.0
        eer_val = 0.0
        eer_threshold = 0.0
        
        for th in thresholds:
            # FAR: 타인(Imposter)인데 가깝다(Distance < th)고 판단한 비율
            fa_count = np.sum(np.array(imposter_scores) < th)
            far = fa_count / len(imposter_scores) if len(imposter_scores) > 0 else 0
            
            # FRR: 본인(Genuine)인데 멀다(Distance >= th)고 판단한 비율
            fr_count = np.sum(np.array(genuine_scores) >= th)
            frr = fr_count / len(genuine_scores) if len(genuine_scores) > 0 else 0
            
            fars.append(far)
            frrs.append(frr)
            
            # EER 업데이트
            if abs(far - frr) < min_diff:
                min_diff = abs(far - frr)
                eer_val = (far + frr) / 2
                eer_threshold = th

        print(f"\n>>> EER (Equal Error Rate): {eer_val*100:.2f}%")
        print(f">>> Recommended Threshold at EER: {eer_threshold:.4f}")

        # 그래프 그리기
        plt.figure(figsize=(12, 5))
        
        # 1. FAR vs FRR Curve
        plt.subplot(1, 2, 1)
        plt.plot(thresholds, fars, label='FAR (False Accept)', color='red')
        plt.plot(thresholds, frrs, label='FRR (False Reject)', color='blue')
        plt.axvline(eer_threshold, color='green', linestyle='--', label=f'EER ({eer_threshold:.2f})')
        plt.title('FAR vs FRR Curve')
        plt.xlabel('Threshold (Hamming Distance)')
        plt.ylabel('Error Rate')
        plt.legend()
        plt.grid(True)
        
        # 2. Score Distribution
        plt.subplot(1, 2, 2)
        plt.hist(genuine_scores, bins=50, alpha=0.6, color='blue', label='Genuine', density=True)
        plt.hist(imposter_scores, bins=50, alpha=0.6, color='red', label='Imposter', density=True)
        plt.axvline(eer_threshold, color='green', linestyle='--')
        plt.title('Score Distribution')
        plt.xlabel('Hamming Distance')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('Hamming_distance_result.png')

if __name__ == "__main__":
    MODEL_PATH = os.path.join(CONFIG["PATH"]["checkpoint_dir"], "pointface_epoch_200.pth")  # 학습된 모델 경로
    DATABASE_DIR = CONFIG["PATH"]["gallery_storage_bio"] # 저장된 데이터들
    MATCHING_DIR = CONFIG["PATH"]["gallery_dir"] # 인식할 얼굴들이 있는 폴더
    
    evaluator = HamdisEvaluator(MODEL_PATH, device=CONFIG["DEVICE"])
    
    # 1. 갤러리 로드
    evaluator.load_gallery(DATABASE_DIR)
    
    # 2. 모든 조합에 대해 점수 계산
    gen_scores, imp_scores = evaluator.calculate_scores(MATCHING_DIR)
    
    # 3. 성능 측정 및 그래프
    evaluator.evaluate_metrics(gen_scores, imp_scores)