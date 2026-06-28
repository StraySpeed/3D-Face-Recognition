import torch
import numpy as np
import os, glob, time
from model.pointface import PointFaceNet
from logger import get_logger
from config import CONFIG


class FaceRecognizer:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.num_points = CONFIG["MODEL"]["num_points"]

        self.model = PointFaceNet(feature_dim=CONFIG["MODEL"]["feature_dim"]).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        self.model.eval()

        self.gallery_dict = {}
        print(f"Model loaded from {model_path}")

    def get_embedding(self, npy_path):
        if not os.path.exists(npy_path):
            return None
        try:
            points = np.load(npy_path)[:, :3]
            pre_data = PointFaceNet.preprocess(points, num_points=self.num_points,
                                               device=str(self.device))
            with torch.no_grad():
                emb = self.model(pre_data).cpu().numpy().flatten()
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 1e-8 else emb
        except Exception as e:
            print(f"Error processing {npy_path}: {e}")
            return None

    def register_gallery(self, data_root):
        """갤러리 등록: 각 ID의 모든 샘플 임베딩을 평균."""
        self.gallery_dict = {}
        identities = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')
        ])
        print(f"Registering {len(identities)} identities...")

        for identity in identities:
            person_dir = os.path.join(data_root, identity)
            embeddings = []
            for f in os.listdir(person_dir):
                if not f.endswith('.npy'):
                    continue
                emb = self.get_embedding(os.path.join(person_dir, f))
                if emb is not None:
                    embeddings.append(emb)

            if embeddings:
                mean_emb = np.mean(embeddings, axis=0)
                norm = np.linalg.norm(mean_emb)
                self.gallery_dict[identity] = mean_emb / norm if norm > 1e-8 else mean_emb
            else:
                print(f"Warning: no valid embeddings for {identity}")

        print(f"Registered {len(self.gallery_dict)} IDs.")

    def _embed_probe(self, npy_path):
        """probe 포인트클라우드 → L2 정규화 임베딩 (numpy 1-D)."""
        points = np.load(npy_path)[:, :3]
        pre_data = PointFaceNet.preprocess(points, num_points=self.num_points,
                                           device=str(self.device))
        with torch.no_grad():
            emb = self.model(pre_data).cpu().numpy().flatten()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-8 else emb

    def recognize(self, npy_path, threshold=CONFIG["MATCHING"]["threshold"]):
        """1:N 인식."""
        if not os.path.exists(npy_path):
            return "File Error", 0.0

        t0 = time.time()
        probe_emb = self._embed_probe(npy_path)
        t1 = time.time()
        print(f"[1:N] Embed: {t1-t0:.4f}s")

        best_score, best_id = -1.0, "Unknown"
        for gid, gemb in self.gallery_dict.items():
            score = float(np.dot(probe_emb, gemb))
            if score > best_score:
                best_score, best_id = score, gid

        t2 = time.time()
        print(f"[1:N] Match: {t2-t1:.4f}s  Total: {t2-t0:.4f}s")

        if best_score < threshold:
            return "Unknown", best_score
        return best_id, best_score

    def recognize_id(self, npy_path, identity_id,
                     threshold=CONFIG["MATCHING"]["threshold"]):
        """1:1 인증."""
        if not os.path.exists(npy_path):
            return "File Error", 0.0

        gallery_emb = self.gallery_dict.get(identity_id)
        if gallery_emb is None:
            print(f"Warning: {identity_id} not in gallery.")
            return "Unknown", 0.0

        t0 = time.time()
        probe_emb = self._embed_probe(npy_path)
        score = float(np.dot(probe_emb, gallery_emb))
        print(f"[1:1] {identity_id} Score: {score:.4f}  Time: {time.time()-t0:.4f}s")

        if score < threshold:
            return "Unknown", score
        return identity_id, score

    def save_gallery(self, save_dir=None):
        save_dir = save_dir or CONFIG["PATH"]["gallery_storage"]
        os.makedirs(save_dir, exist_ok=True)
        for identity, emb in self.gallery_dict.items():
            np.save(os.path.join(save_dir, f"{identity}.npy"), emb)
        print(f"Saved {len(self.gallery_dict)} embeddings to {save_dir}.")

    def load_gallery(self, load_dir=None):
        load_dir = load_dir or CONFIG["PATH"]["gallery_storage"]
        if not os.path.exists(load_dir):
            print(f"Error: {load_dir} not found.")
            return False
        files = glob.glob(os.path.join(load_dir, "*.npy"))
        if not files:
            print("Warning: no .npy files found.")
            return False
        self.gallery_dict = {
            os.path.splitext(os.path.basename(f))[0]: np.load(f)
            for f in files
        }
        print(f"Loaded {len(self.gallery_dict)} identities from {load_dir}.")
        return True


def _latest_checkpoint(ckpt_dir):
    """체크포인트 디렉터리에서 epoch 번호가 가장 높은 .pth 파일 반환."""
    files = glob.glob(os.path.join(ckpt_dir, "pointface2_epoch_*.pth"))
    if not files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return max(files, key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))


if __name__ == "__main__":
    print = get_logger(name='matching').info

    CKPT_DIR    = CONFIG["PATH"]["checkpoint_dir"]
    MATCHING_DIR = CONFIG["PATH"]["gallery_dir"]
    DATABASE_DIR = CONFIG["PATH"]["gallery_storage"]
    THRESHOLD    = CONFIG["MATCHING"]["threshold"]

    model_path = _latest_checkpoint(CKPT_DIR)
    print(f"Using checkpoint: {model_path}")

    recognizer = FaceRecognizer(model_path, device=CONFIG["DEVICE"])

    # 갤러리 등록 후 저장 (이미 저장됐으면 load_gallery 사용)
    recognizer.register_gallery(MATCHING_DIR)
    recognizer.save_gallery(DATABASE_DIR)
    # recognizer.load_gallery(DATABASE_DIR)

    identities = sorted([
        d for d in os.listdir(MATCHING_DIR)
        if os.path.isdir(os.path.join(MATCHING_DIR, d)) and not d.startswith('.')
    ])

    total = correct = wrong = 0
    TP = FP = TN = FN = 0

    for identity_id in identities:
        person_dir = os.path.join(MATCHING_DIR, identity_id)
        for fname in os.listdir(person_dir):
            if not fname.endswith('.npy'):
                continue
            total += 1
            fpath = os.path.join(person_dir, fname)

            # 1:1 인증
            pred_id, score = recognizer.recognize_id(fpath, identity_id,
                                                     threshold=THRESHOLD)
            print(f"[{identity_id}] pred={pred_id}  score={score:.4f}")

            match = (pred_id == identity_id)
            if match:
                correct += 1; TP += 1
            else:
                wrong += 1
                if pred_id == "Unknown":
                    FN += 1   # 맞는 사람인데 거부
                else:
                    FP += 1   # 틀린 사람으로 수락

    TAR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FRR = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    print(f"[Result] Total={total}  Correct={correct}  Wrong={wrong}")
    print(f"[Result] TAR={TAR:.4f}  FAR={FAR:.4f}  FRR={FRR:.4f}")
    print(f"[Result] Acc={correct/total*100:.2f}%" if total > 0 else "")
