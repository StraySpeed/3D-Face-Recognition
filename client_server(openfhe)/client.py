import os, sys
import openfhe
import time
import numpy as np
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from logger import get_logger
from config import CONFIG
from model.pointface import PointFaceNet

class PointFaceClient:
    def __init__(self, key_dir, logger=None):
        if logger:
            self.print = logger
        else:
            self.print = get_logger("matching_client").info
        self.load_context(key_dir)

    def load_context(self, key_dir):
        if not os.path.exists(key_dir):
            self.print("[Client] Error: Key directory not found.")
            return False
            
        self.ctx, _ = openfhe.DeserializeCryptoContext(os.path.join(key_dir, "cryptocontext.txt"), openfhe.BINARY)
        self.public_key, _ = openfhe.DeserializePublicKey(os.path.join(key_dir, "key-public.txt"), openfhe.BINARY)
        self.secret_key, _ = openfhe.DeserializePrivateKey(os.path.join(key_dir, "key-secret.txt"), openfhe.BINARY)
        self.print("[Client] OpenFHE Context and Keys loaded.")
        return True
        
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
        points = PointFaceNet.morton_sort(points)
        pre_data = PointFaceNet.preprocess(points, num_points = CONFIG["MODEL"]["num_points"], device = 'cpu')
        end_time = time.time()
        self.print(f"[Client] 1. Preprocessing Time: {end_time - start_time:.4f}")

        enc_data = {'rel': {}, 'idx': {}}
        
        for stage in ['s1', 's2', 's3', 's4']:
            # 차원을 (1, npoint, nsample, 10) 에서 (npoint, nsample, 10) 으로 줄임
            # rel_tensor = pre_data['rel'][stage].squeeze(0).permute(1, 2, 0).reshape(-1, 10).numpy()
            rel_tensor = pre_data['rel'][stage].squeeze(0).permute(2, 1, 0).reshape(-1, 10).numpy()
            
            enc_rel_channels = []
            for c in range(10):
                # OpenFHE 평문 패킹 후 암호화
                ptxt = self.ctx.MakeCKKSPackedPlaintext(rel_tensor[:, c].tolist())
                ctxt = self.ctx.Encrypt(self.public_key, ptxt)
                
                # 시뮬레이션 최적화: 직렬화(Serialize) 생략하고 객체 직접 전달
                enc_rel_channels.append(ctxt)

            enc_data['rel'][stage] = enc_rel_channels
            enc_data['idx'][stage] = pre_data['idx'][stage].numpy()

        enc_enc_time = time.time()
        self.print(f"[Client] 2. Encryption Time: {enc_enc_time - end_time:.4f}")

        self.print(f"[Client] Client #1 Total Time: {enc_enc_time - start_time:.4f}")
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

        # OpenFHE 복호화
        ptxt_res = self.ctx.Decrypt(self.secret_key, enc_score)
        ptxt_res.SetLength(1) # 추출할 슬롯 고정
        score = ptxt_res.GetRealPackedValue()[0]
        
        end_time = time.time()
        self.print(f"[Client] 1. Decryption Time: {end_time - start_time:.4f}")
        self.print(f"[Client] Client #2 Total Time: {end_time - start_time:.4f}")
        self.print("[Client Side #2 Finished]")
        return True if score > 0 else False