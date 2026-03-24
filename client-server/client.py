import os, sys
import tenseal as ts
import time
import numpy as np
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from logger import get_logger
from config import CONFIG
from model.pointface import PointFaceNet

class PointFaceClient:
    def __init__(self, context, logger=None):
        if logger:
            self.print = logger
        else:
            self.print = get_logger("matching_client").info

        self.load_context(context)

    def load_context(self, context_path):
        """ 외부에서 공개 키를 가져오기 (비밀키 O)
        
        :param context_path: `ckks.Context` file path 
        """
        if not os.path.exists(context_path):
            self.print("[Client] Error: 'secret.context' file not found in directory.")
            return False
        try:
            with open(context_path, "rb") as f:
                context_bytes = f.read()
                self.ctx = ts.context_from(context_bytes, n_threads=4)
            self.print("[Client] Context loaded successfully.")
            return True
        except Exception as e:
            self.print(f"[Client] Error loading context: {e}")
            raise Exception
        
    def preprocess(self, npy_path):
        """
        # Client - 전처리 Step

        각 SetAbstraction에 해당하는 Pre_data를 계산하고(s1~s4), rel_vec만 암호화함 
        
        idx는 평문으로 사용함(인덱스이므로)

        :param npy_path: .npy file path

        :return enc_data: >>> dict('rel': dict('s1': [], 's2': [], 's3': [], 's4': []), 'idx': dict('s1': [], 's2': [], 's3': [], 's4': []))
        """
        self.print("[Client Side #1 Start]")
        points = np.load(npy_path)[:, :3]
        start_time = time.time()
        pre_data = PointFaceNet.preprocess(points, num_points = CONFIG["MODEL"]["num_points"], device = 'cpu')
        end_time = time.time()
        self.print(f"[Client] 1. Preprocessing Time: {end_time - start_time:.4f}")

        enc_data = {'rel': {}, 'idx': {}}
        
        for stage in ['s1', 's2', 's3', 's4']:
            # 1. relation vector는 암호화: PyTorch Tensor -> Numpy -> List -> CKKSVector
            rel_tensor = pre_data['rel'][stage].squeeze(0).permute(1, 2, 0).reshape(-1, 10).numpy()
            enc_rel_list = []
            for c in range(10):
                enc_vec = ts.ckks_vector(self.ctx, rel_tensor[:, c].tolist())
                enc_rel_list.append(enc_vec.serialize())

            enc_data['rel'][stage] = enc_rel_list
            # 2. idx는 평문으로 보냄
            enc_data['idx'][stage] = pre_data['idx'][stage].numpy()

        enc_time = time.time()
        self.print(f"[Client] 2. Encryption Time: {enc_time - end_time:.4f}")

        self.print(f"[Client] Client #1 Total Time: {enc_time - start_time:.4f}")
        self.print("[Client Side #1 Finished]")        
        return enc_data
    
    def decrypt_result(self, enc_score):
        """
        # Client - Decryption

        서버에서 온 암호화된 점수를 복호화해서 인증 여부를 확인함

        :param enc_score: 암호화된 매칭 점수

        :return: >>> True if score-threshold > 0 else False
        """
        self.print("[Client Side #2 Start]")

        start_time = time.time()
        # 복호화
        score = ts.ckks_vector_from(self.ctx, enc_score).decrypt()[0]
        end_time = time.time()
        self.print(f"[Client] 1. Decryption Time: {end_time - start_time:.4f}")

        self.print(f"[Client] Client #2 Total Time: {end_time - start_time:.4f}")
        self.print("[Client Side #2 Finished]")
        return True if score > 0 else False