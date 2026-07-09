import torch
import numpy as np
import os, glob
from model.pointface import PointFaceNet
from config import CONFIG


def _latest_checkpoint(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, "pointface_epoch_100.pth"))
    if not files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return max(files, key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))


class Rank1Evaluator:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.num_points = CONFIG["MODEL"]["num_points"]

        self.model = PointFaceNet(
            feature_dim=CONFIG["MODEL"]["feature_dim"]
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        self.model.eval()
        print(f"Model loaded: {model_path}")

    def extract_embedding(self, npy_path):
        if not os.path.exists(npy_path):
            return None
        try:
            points = np.load(npy_path)[:, :3]
            pre_data = PointFaceNet.preprocess(
                points, num_points=self.num_points, device=str(self.device)
            )
            with torch.no_grad():
                emb = self.model(pre_data).cpu().numpy().flatten()
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 1e-8 else emb
        except Exception as e:
            print(f"  Error: {npy_path}: {e}")
            return None

    def evaluate(self, data_root, gallery_per_id=1):
        """
        Rank-1 Accuracy 측정.

        각 ID에서 앞 gallery_per_id개 파일을 Gallery,
        나머지를 Probe로 사용한다.
        Neutral 표정 파일(_NE_)이 있으면 Gallery 우선 배치.
        """
        identities = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')
        ])
        print(f"Extracting embeddings from {len(identities)} identities "
              f"(gallery_per_id={gallery_per_id})...")

        gallery_embs, gallery_ids = [], []
        probe_embs,   probe_ids   = [], []

        for idx, identity in enumerate(identities):
            person_dir = os.path.join(data_root, identity)
            files = sorted([
                f for f in os.listdir(person_dir)
                if f.endswith('.npy') and not f.startswith('.')
            ])
            if len(files) < 2:
                continue

            # Neutral 파일(_NE_)을 앞으로 정렬
            neutral = [f for f in files if '_NE_' in f]
            others  = [f for f in files if '_NE_' not in f]
            ordered = neutral + others

            gal_files   = ordered[:gallery_per_id]
            probe_files = ordered[gallery_per_id:]

            for f in gal_files:
                emb = self.extract_embedding(os.path.join(person_dir, f))
                if emb is not None:
                    gallery_embs.append(emb)
                    gallery_ids.append(idx)

            for f in probe_files:
                emb = self.extract_embedding(os.path.join(person_dir, f))
                if emb is not None:
                    probe_embs.append(emb)
                    probe_ids.append(idx)

        if not gallery_embs or not probe_embs:
            print(f"Insufficient data: {len(gallery_embs)} gallery, "
                  f"{len(probe_embs)} probes.")
            return None

        G = np.array(gallery_embs)   # (N_g, feature_dim)
        P = np.array(probe_embs)     # (N_p, feature_dim)
        gl = np.array(gallery_ids)
        pl = np.array(probe_ids)

        print(f"Gallery: {len(G)}  Probes: {len(P)}  "
              f"feature_dim: {G.shape[1]}")

        # Cosine similarity matrix (P, G)
        sim_matrix  = P @ G.T
        pred_labels = gl[np.argmax(sim_matrix, axis=1)]

        correct  = int(np.sum(pred_labels == pl))
        total    = len(pl)
        accuracy = correct / total * 100

        print(f"Rank-1 Accuracy: {accuracy:.2f}%  ({correct}/{total})")
        return accuracy


if __name__ == "__main__":
    CKPT_DIR = CONFIG["PATH"]["checkpoint_dir"]
    DATA_DIR = CONFIG["PATH"]["gallery_dir"]

    model_path = _latest_checkpoint(CKPT_DIR)

    evaluator = Rank1Evaluator(model_path, device=CONFIG["DEVICE"])
    evaluator.evaluate(DATA_DIR, gallery_per_id=1)
