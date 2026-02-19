import tenseal as ts
import glob, time, os
import sys
import numpy as np
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from logger import get_logger

class PointFaceServer:
    THRESHOLD = 0.7
    def __init__(self, context_bytes, logger=None):
        # 1. 클라이언트가 준 Context 로드 (Secret Key 미포함)
        self.context = ts.context_from(context_bytes)

        if logger:
            self.print = logger
        else:
            self.print = get_logger("matching_server").info

        self.encrypted_gallery = {}


    def load_gallery_individual(self, load_dir="./gallery_storage"):
        """
        폴더 내의 모든 .ts 파일을 읽어서 갤러리로 로드
        """
        if not os.path.exists(load_dir):
            self.print(f"Error: Directory '{load_dir}' not found.")
            return False

        # 1. Context 로드
        context_path = os.path.join(load_dir, "secret.context")
        if not os.path.exists(context_path):
            self.print("Error: 'secret.context' file not found in directory.")
            return False
        try:
            with open(context_path, "rb") as f:
                context_bytes = f.read()
                self.ctx = ts.context_from(context_bytes, n_threads=4)
            self.print("Context loaded successfully.")
        except Exception as e:
            self.print(f"Error loading context: {e}")
            return None
        
        ts_files = glob.glob(os.path.join(load_dir, "*.ts"))
        
        if not ts_files:
            self.print("Warning: No .ts files found in the directory.")
            return False
            
        self.print(f"Loading embeddings from '{load_dir}'...")
        
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
                self.print(f"Error loading {filename}: {e}")
            
        self.print(f"Loaded {len(self.encrypted_gallery)} identities.")
        return True
    
    def recognize_encrypted_id(self, enc_feature, id):
        """암호화된 상태에서 매칭 수행
        
        :return: score * Random Positive Value
        """

        self.print("[Server Side Start]")
        enc_probe = ts.ckks_vector_from(self.ctx, enc_feature)
        
        sim_start_time = time.time()
        # 갤러리와 동형 연산 (내적) 수행
        enc_gallery_vec = self.encrypted_gallery.get(id)
            
        # Homomorphic Dot Product
        # (A . B) = Cosine Similarity (||A||=1, ||B||=1)
        enc_score = enc_probe.dot(enc_gallery_vec).sub(PointFaceServer.THRESHOLD).mul([np.random.uniform(10.0, 1000.0)])
        end_time = time.time()
        self.print(f"Total Matching Time: {end_time - sim_start_time:.4f}")
        self.print(f"[Server Side Finished]")
        return enc_score.serialize()
