import tenseal as ts
from he_utils import load_server_weights
import glob, time, os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from logger import get_logger

class PointFaceServer:
    def __init__(self, model_path, context_bytes, logger=None):
        # 1. 클라이언트가 준 Context 로드 (Secret Key 미포함)
        self.context = ts.context_from(context_bytes)

        if logger:
            self.print = logger
        else:
            self.print = get_logger("matching_server").info

        # 2. 모델 가중치 로드 및 병합 (Linear + BN)
        # W shape: (512, 512), b shape: (512,)
        self.W, self.b, _ = load_server_weights(model_path)
        
        self.encrypted_gallery = {} # {id: {'vec': enc_vec, 'norm_sq': enc_norm_sq}}

    def process_feature(self, enc_feature_bytes):
        """
        Client Feature(512) -> FC Layer(Linear+BN) -> Embedding(512)
        """
        # 1. Deserialize
        enc_vec = ts.ckks_vector_from(self.context, enc_feature_bytes)
        # return enc_vec
        # 2. FC Layer Operation (y = xW^T + b)
        # enc_vec: (512,)
        # W: (512, 512) -> TenSEAL matmul expects matrix
        
        # TenSEAL의 matmul은 x.matmul(W) 형태
        # x(1, 512) @ W(512, 512)
        enc_embedding = enc_vec.matmul(self.W)
        enc_embedding.add_(self.b)
        
        return enc_embedding


    def register(self, identity, enc_feature_bytes):
        """
        갤러리 등록: 임베딩을 계산하고, 그 임베딩의 'Squared Norm'도 미리 계산해둠
        """
        # 1. FC Layer 통과
        enc_emb = self.process_feature(enc_feature_bytes)
        
        # 2. Squared Norm 계산 (||v||^2 = v . v)
        # 코사인 유사도 분모 계산용
        enc_norm_sq = enc_emb.dot(enc_emb)
        
        self.encrypted_gallery[identity] = {
            'vec': enc_emb,
            'norm_sq': enc_norm_sq
        }
        self.print(f"Server: Registered ID {identity}")

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
                self.context = ts.context_from(context_bytes, n_threads=4)
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
        vec = probe_norm_sq = None
        for file_path in ts_files:
            # 파일명에서 확장자 제거하여 ID로 사용 (예: id_001.ts -> id_001)
            filename = os.path.basename(file_path)
            identity = os.path.splitext(filename)[0][:7]

            # 로드
            try:
                with open(file_path, 'rb') as f:
                    vec_bytes = f.read()
                
                # 로드한 Context를 이용해 벡터 복원
                vec = ts.ckks_vector_from(self.context, vec_bytes)
                probe_norm_sq = vec.dot(vec)
                self.encrypted_gallery[identity] = {
                    'vec': vec,
                    'norm_sq': probe_norm_sq
                }
                
            except Exception as e:
                self.print(f"Error loading {filename}: {e}")
            
        self.print(f"Loaded {len(self.encrypted_gallery)} identities.")
        return True
    
    def match(self, enc_probe_bytes, id):
        """
        Probe Feature -> FC Layer -> Embedding

        :Return: (ID, Enc_Dot, Enc_Probe_NormSq, Enc_Gal_NormSq)
        """
        self.print("[Server Side Start]")
        start_time = time.time()
        # 1. FC Layer 통과
        probe_emb = self.process_feature(enc_probe_bytes)
        
        # 2. Probe Norm Squared 계산
        probe_norm_sq = probe_emb.dot(probe_emb)
        fc_end_time = time.time()
        self.print(f"1. FC Layer Time: {fc_end_time - start_time:.4f}")

        results = []
        
        # 3. 갤러리와 매칭
        gallery = self.encrypted_gallery.get(id, None)
        gal_emb = gallery['vec']
        gal_norm_sq = gallery['norm_sq']
        
        # Dot Product (분자)
        dot_prod = probe_emb.dot(gal_emb)
        
        # 클라이언트에게 보낼 결과 패키징 (모두 암호화된 상태)
        # (Identity, DotProduct, ProbeNorm^2, GalleryNorm^2)
        results.append((
            id,
            dot_prod.serialize(), 
            probe_norm_sq.serialize(), 
            gal_norm_sq.serialize()
        ))
        end_time = time.time()

        self.print(f"2. Server Total Time: {end_time - start_time:.4f}")
        self.print(f"[Server Side Finished]")
        return results