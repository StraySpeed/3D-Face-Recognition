import torch
import numpy as np
import os
from model.pointface import PointFaceNet
from config import CONFIG

class Rank1Evaluator:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = PointFaceNet(num_classes=143).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)

        self.model.eval()

    def extract_embedding(self, npy_path):
        if not os.path.exists(npy_path): return None
        try:
            points = np.load(npy_path)[:, :3]
            pre_data = PointFaceNet.preprocess(points, device=self.device)
            with torch.no_grad():
                emb = self.model(pre_data).cpu().numpy().flatten()
            # L2 Normalize
            return emb / np.linalg.norm(emb)
        except Exception as e:
            return None

    def evaluate(self, data_root):
        """
        Rank-1 Accuracy 측정
        """
        gallery_feats = []  # [(embedding, label_id), ...]
        probe_feats = []    # [(embedding, label_id), ...]
        
        identities = sorted([d for d in os.listdir(data_root) 
                             if os.path.isdir(os.path.join(data_root, d))])
        
        print(f"Extracting features from {len(identities)} identities...")
        
        for idx, identity in enumerate(identities):
            person_dir = os.path.join(data_root, identity)
            files = sorted([f for f in os.listdir(person_dir) if f.endswith('.npy')])
            
            if len(files) < 2:
                continue # 비교할 대상이 없으면 스킵
            
            # 1. First file -> Gallery
            gal_path = os.path.join(person_dir, files[0])
            gal_emb = self.extract_embedding(gal_path)
            if gal_emb is not None:
                gallery_feats.append((gal_emb, idx)) # idx를 라벨로 사용
            
            # 2. Rest files -> Probes
            for f in files[1:]:
                prob_path = os.path.join(person_dir, f)
                prob_emb = self.extract_embedding(prob_path)
                if prob_emb is not None:
                    probe_feats.append((prob_emb, idx))
                    
        # Matrix 연산을 위해 Stack
        if not gallery_feats or not probe_feats:
            print(f"{len(gallery_feats)} gallery feats, {len(probe_feats)} probe feats.")
            return

        gallery_matrix = np.array([f[0] for f in gallery_feats]) # (G, 512)
        gallery_labels = np.array([f[1] for f in gallery_feats]) # (G,)
        
        probe_matrix = np.array([f[0] for f in probe_feats])     # (P, 512)
        probe_labels = np.array([f[1] for f in probe_feats])     # (P,)
        
        print(f"Evaluating Rank-1 Accuracy (Gallery: {len(gallery_feats)}, Probes: {len(probe_feats)})...")
        
        # Cosine Similarity Matrix: (P, G)
        # Probe i와 Gallery j의 유사도 = sim_matrix[i][j]
        sim_matrix = np.dot(probe_matrix, gallery_matrix.T)
        
        # 각 Probe에 대해 가장 높은 점수를 가진 Gallery 인덱스 찾기
        pred_indices = np.argmax(sim_matrix, axis=1) # (P,)
        
        # 예측된 Gallery의 라벨 가져오기
        pred_labels = gallery_labels[pred_indices]
        
        # 정확도 계산
        correct = np.sum(pred_labels == probe_labels)
        total = len(probe_labels)
        accuracy = correct / total * 100
        
        print(f" Rank-1 Accuracy: {accuracy:.2f}% ({correct}/{total})")


        return accuracy

if __name__ == "__main__":
    MODEL_FILE = os.path.join(CONFIG["PATH"]["checkpoint_dir"], "pointface_epoch_200.pth")
    DATA_DIR = CONFIG["PATH"]["gallery_dir"]
    
    evaluator = Rank1Evaluator(MODEL_FILE, device=CONFIG["DEVICE"])
    evaluator.evaluate(DATA_DIR)