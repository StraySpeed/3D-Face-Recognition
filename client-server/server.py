import torch, pickle
import numpy as np
import tenseal as ts
import os, glob, time
from logger import get_logger
from config import CONFIG
from model.pointface import PointFaceNet

class PointFaceServer:
    THRESHOLD = CONFIG["MATCHING"]["threshold"]
    def __init__(self, contexts, logger=None):
        if logger:
            self.print = logger
        else:
            self.print = get_logger("matching_server").info
        # 1. 클라이언트가 준 Context 로드 (Secret Key 미포함)
        self.load_context(contexts)
        self.he_model = HE_PointFaceNet(self.ctx, self.print)
        self.encrypted_gallery = {}
        
    def load_context(self, context_path):
        """ 외부에서 공개 키를 가져오기 (비밀키 X) """
        if not os.path.exists(context_path):
            self.print("[Server] Error: 'secret.context' file not found in directory.")
            return False
        try:
            with open(context_path, "rb") as f:
                context_bytes = f.read()
                self.ctx = ts.context_from(context_bytes, n_threads=4)
            self.print("[Server] Context loaded successfully.")
            return True
        except Exception as e:
            self.print(f"[Server] Error loading context: {e}")
            raise Exception

    def load_gallery_indvidual_ts(self, load_dir="gallery_storage"):
        """
        # Server - 폴더 내의 모든 암호화된 .ts 파일을 읽어서 갤러리로 로드
        
        (사용 X)

        :param load_dir: 암호화된 .ts 파일들이 들어있는 폴더
        """
        if not os.path.exists(load_dir):
            self.print(f"[Server] Error: Directory '{load_dir}' not found.")
            return False        

        ts_files = glob.glob(os.path.join(load_dir, "*.ts"))
        
        if not ts_files:
            self.print("[Server] Warning: No .ts files found in the directory.")
            return False

        self.encrypted_gallery = {}
        for file_path in ts_files:
            # 파일명에서 확장자 제거하여 ID로 사용 (예: id_001.ts -> id_001)
            filename = os.path.basename(file_path)
            identity = os.path.splitext(filename)[0]
            
            # 로드
            try:
                with open(file_path, 'rb') as f:
                    vec_bytes = f.read()
                
                # 로드한 Context를 이용해 텐서 복원
                vec = ts.ckks_tensor_from(self.ctx, vec_bytes)
                self.encrypted_gallery[identity] = vec
                
            except Exception as e:
                self.print(f"Error loading {filename}: {e}")
            
        self.print(f"[Server] Loaded {len(self.encrypted_gallery)} encrypted identities.")
        return True

    def load_gallery_individual_npy(self, load_dir="./gallery_storage"):
        """
        # Server - 폴더 내의 모든 평문 .npy 파일을 읽어서 갤러리로 로드

        :param load_dir: .npy 파일들이 들어있는 폴더
        """
        if not os.path.exists(load_dir):
            self.print(f"[Server] Error: Directory '{load_dir}' not found.")
            return False        

        npy_files = glob.glob(os.path.join(load_dir, "*.npy"))
        if not npy_files:
            self.print("[Server] Warning: No .npy files found in the directory.")
            return False
            
        self.plain_gallery = {}
        for file_path in npy_files:
            identity = os.path.splitext(os.path.basename(file_path))[0]
            self.plain_gallery[identity] = np.load(file_path)
            
        self.print(f"[Server] Loaded {len(self.plain_gallery)} plain identities.")
        return True

    def load_gallery_individual_pkl(self, load_dir="gallery_storage"):
        """
        # Server - 폴더 내의 모든 암호화된 .pkl 파일을 읽어서 갤러리로 로드

        128개의 암호문이 들어있음

        :param load_dir: 암호화된 .pkl 파일들이 들어있는 폴더
        """
        if not os.path.exists(load_dir):
            self.print(f"[Server] Error: Directory '{load_dir}' not found.")
            return False        

        pkl_files = glob.glob(os.path.join(load_dir, "*.pkl"))
        
        if not pkl_files:
            self.print("[Server] Warning: No .pkl files found in the directory.")
            return False

        self.encrypted_gallery = {}
        for file_path in pkl_files:
            # 파일명에서 확장자 제거하여 ID로 사용 (예: id_001.ts -> id_001)
            identity = os.path.splitext(os.path.basename(file_path))[0]
            
            # 로드
            try:
                with open(file_path, "rb") as f:
                    serialized_list = pickle.load(f)
                
                # 역직렬화: 바이트 문자열들을 다시 ts.ckks_vector 128개 리스트로 복원
                enc_vec_list = [ts.ckks_vector_from(self.ctx, vec_bytes) for vec_bytes in serialized_list]
                self.encrypted_gallery[identity] = enc_vec_list
                
            except Exception as e:
                self.print(f"Error loading {identity}.pkl: {e}")
            
        self.print(f"[Server] Loaded {len(self.encrypted_gallery)} encrypted identities.")
        return True

    def recognize_encrypted_id2(self, enc_data, id):
        """
        # Server - 평문 데이터와 매칭 수행

        :param enc_data: Client에서 보낸 암호화된 데이터

        :param id: Server에서 비교할 데이터의 ID
        
        :return: score - Threshold * Random Positive Value
        """
        self.print("[Server Side Start]")
        start_time = time.time()
        
        # 1. 암호화된 특징 추출 (feature_dim개 ckks_vector 리스트 반환)
        enc_emb_list = self.he_model.get_embedding_enc(enc_data)
        emb_time = time.time()
        self.print(f"[Server] 1. Embedding Time(enc): {emb_time - start_time:.4f}")

        # 2. 평문 갤러리 로드
        plain_gallery_vec = self.plain_gallery.get(id)
            
        # 암호문 리스트와 평문 벡터의 내적(Dot Product)을 for 루프로 안전하게 계산

        # 매칭에 Depth 2를 소모함 (내적 계산 1 + 제곱 판별식 1)
        enc_score = enc_emb_list[0] * plain_gallery_vec[0].item()
        for i in range(1, CONFIG["MODEL"]["feature_dim"]):
            enc_score += enc_emb_list[i] * plain_gallery_vec[i].item()
        
        enc_mag_sq = enc_emb_list[0] * enc_emb_list[0]
        for i in range(1, CONFIG["MODEL"]["feature_dim"]):
            enc_mag_sq += enc_emb_list[i] * enc_emb_list[i]

        # feature를 정규화하지 못하므로 나눗셈 없는 제곱 판별식(D) 계산
        threshold_sq = PointFaceServer.THRESHOLD ** 2
        enc_score_sq = enc_score * enc_score
        
        # D = Score^2 - (Threshold^2 * Magnitude^2)
        enc_diff = enc_score_sq - (enc_mag_sq * threshold_sq)

        # 3. 임계값(Threshold) 빼고 블라인딩 무작위 수 곱하기
        enc_score = enc_score.sub(PointFaceServer.THRESHOLD).mul(np.random.uniform(10.0, 1000.0))
        end_time = time.time()

        # 원래 서버에는 비밀 키가 없음 - 점수를 알 수 없음
        # 여기서는 편의를 위해 서버에 비밀 키를 넣음
        self.print(f"[Server_Debug] {id} Matching Score : {enc_diff.decrypt()[0]}, Similarity : {enc_score_sq.decrypt()[0] / enc_mag_sq.decrypt()[0]:.4f}")
        self.print(f"[Server] 2. 1:1 Matching Time: {end_time - emb_time:.4f}")
        self.print(f"[Server] Total Matching Time: {end_time - start_time:.4f}")
        self.print(f"[Server Side Finished]")
        
        return enc_score.serialize()

    def recognize_encrypted_id(self, enc_data, id):
        """
        # Server - 암호화된 데이터와 매칭 수행
        
        128개의 암호문을 이용하여 매칭함
        
        :return: score - Threshold * Random Positive Value
        """
        self.print("[Server Side Start]")
        start_time = time.time()
        enc_emb_list = self.he_model.get_embedding_enc(enc_data)
        emb_time = time.time()
        self.print(f"[Server] 1. Embedding Time(enc): {emb_time - start_time:.4f}")
        enc_gallery_list = self.encrypted_gallery.get(id)

        # 완벽한 암호문 대 암호문(C x C) 내적 연산 수행
        # 128개의 채널에 대해 각각 곱셈(C x C) 발생 -> enc_score는 Depth 20 도달
        enc_score = enc_emb_list[0] * enc_gallery_list[0]
        for i in range(1, CONFIG["MODEL"]["feature_dim"]):
            enc_score += enc_emb_list[i] * enc_gallery_list[i]

        # 임베딩 크기 제곱(L2 Norm^2) 계산 (C x C, Depth 20)
        enc_mag_sq = enc_emb_list[0] * enc_emb_list[0]
        for i in range(1, CONFIG["MODEL"]["feature_dim"]):
            enc_mag_sq += enc_emb_list[i] * enc_emb_list[i]

        # 제곱 판별식 계산 (나눗셈 우회)
        threshold_sq = PointFaceServer.THRESHOLD ** 2
        
        # Score 제곱 (C x C) -> Depth 21 도달
        enc_score_sq = enc_score * enc_score
        
        # Mag_sq에 평문 임계값 곱셈 (C x P) -> Depth 21 도달
        enc_diff = enc_score_sq - (enc_mag_sq * threshold_sq)

        # 랜덤 블라인딩 처리 (C x P) -> 최종 Depth 22
        # 클라이언트가 원래 점수를 역산할 수 없도록 무작위 양수 팩터 곱셈
        blind_factor = np.random.uniform(10.0, 1000.0)
        enc_blinded_result = enc_diff * blind_factor

        end_time = time.time()

        # 원래 서버에는 비밀 키가 없음 - 점수를 알 수 없음
        # 여기서는 편의를 위해 서버에 비밀 키를 넣음
        self.print(f"[Server_Debug] {id} Matching Score : {enc_diff.decrypt()[0]}, Similarity : {enc_score_sq.decrypt()[0] / enc_mag_sq.decrypt()[0]:.4f}")
        
        self.print(f"[Server] 2. 1:1 Matching (C x C) Time: {end_time - emb_time:.4f}")
        self.print(f"[Server] Total Matching Time: {end_time - start_time:.4f}")
        self.print(f"[Server Side Finished]")
        
        return enc_blinded_result.serialize()



class HE_PointFaceNet:
    """
    동형암호 상태에서 모델을 통과시키기 위한 모듈
    """
    MODEL_PATH = os.path.join(CONFIG["PATH"]["checkpoint_dir"], "pointface_epoch_200.pth")
    def __init__(self, ctx, logger=None):
        if logger:
            self.print = logger
        else:
            self.print = get_logger("matching_server").info

        self.device = 'cuda'    # 'cpu'
        self.model = PointFaceNet(num_classes=CONFIG["MODEL"]["num_classes"]).to(self.device)
        checkpoint = torch.load(HE_PointFaceNet.MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("[Server] Extracting and Folding Weights for HE Server...")
        self.weights = self._extract_and_fold_weights()
        self.ctx = ctx

    def _fold_conv_bn_1x1(self, conv, bn):
        """
        # Server - 1x1 Conv와 BN을 하나의 W, b 로 병합
        """
        w = conv.weight.data.cpu().numpy() # (Out, In, 1, 1)
        b = conv.bias.data.cpu().numpy() if conv.bias is not None else np.zeros(w.shape[0])
        
        if bn.weight is not None:
            gamma = bn.weight.data.cpu().numpy()
            beta = bn.bias.data.cpu().numpy()
        else:
            gamma = np.ones(w.shape[0])
            beta = np.zeros(w.shape[0])

        mean = bn.running_mean.cpu().numpy()
        var = bn.running_var.cpu().numpy()
        eps = bn.eps
        
        std = np.sqrt(var + eps)
        multiplier = gamma / std
        
        # W_fold = W * (gamma/std), b_fold = (b - mean)*(gamma/std) + beta
        w_fold = w.reshape(w.shape[0], w.shape[1]) * multiplier[:, np.newaxis]
        b_fold = (b - mean) * multiplier + beta
        
        # (In, Out) 형태로 전치하여 반환
        return w_fold.T, b_fold

    def _extract_hermite_params(self, hermite_layer):
        """
        # Server - 학습된 에르미트 다항식의 a, b, c 계수를 추출
        """
        a = hermite_layer.a.item()
        b = hermite_layer.b.item()
        c = hermite_layer.c.item()
        return a, b, c

    def _extract_and_fold_weights(self):
        """
        # Server - 모든 SA 계층의 가중치를 추출
        """
        he_w = {}
        for i, sa_module in enumerate([self.model.encoder.sa1, self.model.encoder.sa2, 
                                       self.model.encoder.sa3, self.model.encoder.sa4]):
            rsconv = sa_module.rsconv
            
            # MLP 1단계 (Conv + BN)
            w1, b1 = self._fold_conv_bn_1x1(rsconv.mlp[0], rsconv.mlp[1])
            # Hermite 파라미터
            ha, hb, hc = self._extract_hermite_params(rsconv.mlp[2])
            # MLP 2단계 (Conv + BN)
            w2, b2 = self._fold_conv_bn_1x1(rsconv.mlp[3], rsconv.mlp[4])
            # Linear (Conv1d + BN1d)
            w_lin, b_lin = self._fold_conv_bn_1x1(rsconv.linear[0], rsconv.linear[1])
            
            # 다항식을 x^2 + (b/a)x + (c/a) 형태로 변경
            hb_new = hb / ha
            hc_new = hc / ha
            
            # 밖으로 빼낸 'ha'를 다음 레이어인 w2 행렬 전체에 스칼라 곱으로 미리 흡수시킴
            w2_new = w2 * ha

            he_w[f's{i+1}'] = {
                'w1': w1, 'b1': b1, 'hb_new': hb_new, 'hc_new': hc_new,
                'w2': w2_new, 'b2': b2, 'w_lin': w_lin, 'b_lin': b_lin
            }
            
        # 마지막 FC 레이어
        fc_layer = self.model.encoder.fc
        fc_w = fc_layer[0].weight.data.cpu().numpy().T
        fc_b = fc_layer[0].bias.data.cpu().numpy() if fc_layer[0].bias is not None else np.zeros(fc_w.shape[1])

        # BN1d 폴딩 
        bn_layer = fc_layer[1]

        # FC 레이어의 BN 처리에도 안전장치 추가
        if bn_layer.weight is not None:
            gamma = bn_layer.weight.data.cpu().numpy()
            beta = bn_layer.bias.data.cpu().numpy()
        else:
            gamma = np.ones(fc_w.shape[1])
            beta = np.zeros(fc_w.shape[1])
            
        mean = bn_layer.running_mean.cpu().numpy()
        var = bn_layer.running_var.cpu().numpy()
        eps = bn_layer.eps
        std = np.sqrt(var + eps)
        multiplier = gamma / std
        
        he_w['fc_w'] = fc_w * multiplier
        he_w['fc_b'] = (fc_b - mean) * multiplier + beta

        return he_w

    # ====================================================================
    # 동형암호 추론 연산부 (TenSEAL CKKSTensor 연산)
    # ====================================================================
    
    def _create_grouping_matrix(self, plain_idx, N_prev):
        """
        [인덱싱 우회 기법]
        이전 단계의 점 개수(N_prev)에서 현재 단계의 이웃점들을 뽑아내기 위해,
        (npoint * nsample, N_prev) 형태의 0과 1로 이루어진 평문 선택 행렬을 만듭니다.
        """
        # plain_idx 형태: (1, npoint, nsample)
        idx_flat = plain_idx.flatten()
        num_target = len(idx_flat)
        
        # 행은 타겟 크기(npoint * nsample), 열은 이전 단계 점 개수(N_prev)
        G = np.zeros((num_target, N_prev))
        for i, idx in enumerate(idx_flat):
            G[i, idx] = 1.0
            
        return G

    def _create_sum_pooling_matrix(self, npoint, nsample):
        """
        [Sum Pooling 우회 기법]
        각 npoint마다 nsample개의 데이터를 합치기 위해,
        (npoint, npoint * nsample) 형태의 평문 합산 행렬을 만듭니다.
        """
        S = np.zeros((npoint, npoint * nsample))
        for i in range(npoint):
            S[i, i*nsample : (i+1)*nsample] = 1.0
            
        return S
    
    # ====================================================================
    # 동형암호 로그 합산 함수
    # ====================================================================

    def _create_compaction_matrix(self, npoint, nsample):
        """
        [압축 행렬 (Compaction Matrix)]
        로그 스케일 합산 후 0, 16, 32... 번째 인덱스에 위치한 정답값들을
        0, 1, 2... 번째 인덱스로 당겨와 크기가 npoint인 꽉 찬 벡터로 압축합니다.
        기존 2,048개의 꽉 찬 대각선을 단 128개의 대각선으로 줄입니다.
        """
        total_len = npoint * nsample
        C = np.zeros((total_len, npoint))
        
        for i in range(npoint):
            C[i * nsample, i] = 1.0
            
        return C

    def _logarithmic_shift_and_add(self, enc_vector, nsample):
        """
        [로그 스케일 합산 (Logarithmic Shift-and-Add)]
        단 log2(nsample) 번의 회전과 덧셈만으로 이웃점들을 초고속으로 합산합니다.
        """
        steps = int(np.log2(nsample))
        result = enc_vector 
        for i in range(steps):
            shift = 2 ** i
            # TenSEAL의 벡터 슬롯 회전 메서드를 호출합니다.
            # 파이썬 기반 CKKSVector는 Rotate를 지원하지 않음
            rotated = result.copy()
            rotated.rotate(shift)
            result = result + rotated
            
        return result

    # ==========================================
    # 단일 SA(Set Abstraction) 블록 연산
    # ==========================================
    def _he_rsconv_forward(self, enc_features_list, enc_rel_list, plain_idx, w_dict, npoint, nsample, N_prev):
        """
        # Server - 동형암호 상태에서 SA 블록 연산
        """
        # 1. MLP 1 (10채널 -> hid_ch)
        enc_x = []
        for j in range(w_dict['w1'].shape[1]):
            channel_j = enc_rel_list[0] * w_dict['w1'][0, j].item()
            for i in range(1, 10): channel_j += enc_rel_list[i] * w_dict['w1'][i, j].item()
            channel_j += w_dict['b1'][j].item()
            enc_x.append(channel_j)
            
        # 2. Hermite Activation
        for j in range(len(enc_x)):
            # x^2 연산 (C*C, Depth 1 소모)
            enc_x_sq = enc_x[j] * enc_x[j]
            # (b/a)x 연산 (C*P, Depth 1 소모) -> enc_x_sq와 Depth 레벨이 동일하게 맞춰짐
            enc_x_lin = enc_x[j] * w_dict['hb_new']
            
            # 덧셈은 Depth를 소모하지 않음 (최종 소모 Depth = 1)
            enc_x[j] = enc_x_sq + enc_x_lin + w_dict['hc_new']
            
        # 3. MLP 2 (hid_ch -> in_ch)
        enc_weights = []
        for j in range(w_dict['w2'].shape[1]):
            channel_j = enc_x[0] * w_dict['w2'][0, j].item()
            for i in range(1, len(enc_x)): channel_j += enc_x[i] * w_dict['w2'][i, j].item()
            channel_j += w_dict['b2'][j].item()
            enc_weights.append(channel_j)
            
        # 4. Feature Grouping & Multiplication
        if enc_features_list is not None:
            # G 행렬을 전치(.T)하여 matmul 포맷으로 맞춤
            G = self._create_grouping_matrix(plain_idx, N_prev).T.tolist()
            enc_grouped = [feat.matmul(G) for feat in enc_features_list]
            enc_weighted = [enc_grouped[i] * enc_weights[i] for i in range(len(enc_weights))]
        else:
            enc_weighted = enc_weights
            
        # 5. Sum Pooling
        S = self._create_sum_pooling_matrix(npoint, nsample).T.tolist()
        # 여기가 제일 오래 걸림
        enc_pooled = [feat.matmul(S) for feat in enc_weighted]
        
        # =========== 현재 사용 불가 ============
        # 5-1. 초고속 로그 합산 연산 (행렬곱 대체, Depth 0 소모)
        #enc_shifted = [self._logarithmic_shift_and_add(feat, nsample) for feat in enc_weighted]
        # 5-2. 압축 행렬 곱셈 (128번의 대각선 연산만 발생, Depth 1 소모)
        # 듬성듬성한 벡터에서 유효한 값만 앞으로 당겨와 npoint 크기로 맞춥니다.
        #C = self._create_compaction_matrix(npoint, nsample).tolist()
        #enc_pooled = [feat.matmul(C) for feat in enc_shifted]
        # ======================================

        # 6. Final Linear Projection (in_ch -> out_ch)
        enc_new_features = []
        for j in range(w_dict['w_lin'].shape[1]):
            channel_j = enc_pooled[0] * w_dict['w_lin'][0, j].item()
            for i in range(1, len(enc_pooled)): channel_j += enc_pooled[i] * w_dict['w_lin'][i, j].item()
            channel_j += w_dict['b_lin'][j].item()
            enc_new_features.append(channel_j)
            
        return enc_new_features
    

    # ==========================================
    # 메인 추론 함수 (SA1 ~ SA4 파이프라인)
    # ==========================================
    def get_embedding_enc(self, enc_data):
        """
        # Server - 동형 암호 상태에서 특징을 추출하는 함수
        
        :param enc_data: Client에서 보낸 암호화된 데이터
        :return: `CONFIG["MODEL"]["feature_dim"] 개수의 암호문`
        """
        start_time = time.time()
        
        # 암호문 복원 (List of ckks_vectors)
        enc_rel = {}
        for stage in ['s1', 's2', 's3', 's4']:
            enc_rel[stage] = [ts.ckks_vector_from(self.ctx, vec_bytes) for vec_bytes in enc_data['rel'][stage]]
        plain_idx = enc_data['idx']
        
        sa_params = CONFIG['SA_PARAMS']
        
        # === Stage 1 ===
        self.print("[Server] SA1 연산 중...")
        l1_features = self._he_rsconv_forward(
            None, 
            enc_rel['s1'], 
            plain_idx['s1'], 
            self.weights['s1'],
            sa_params[0]['npoint'], 
            sa_params[0]['nsample'],
            N_prev=CONFIG["MODEL"]["num_points"] # Stage 1의 초기 점 개수
        )
        
        # === Stage 2 ===
        self.print("[Server] SA2 연산 중...")
        l2_features = self._he_rsconv_forward(l1_features, enc_rel['s2'], plain_idx['s2'], self.weights['s2'],
                                              sa_params[1]['npoint'], sa_params[1]['nsample'], sa_params[0]['npoint'])
        
        # === Stage 3 ===
        self.print("[Server] SA3 연산 중...")
        l3_features = self._he_rsconv_forward(l2_features, enc_rel['s3'], plain_idx['s3'], self.weights['s3'],
                                              sa_params[2]['npoint'], sa_params[2]['nsample'], sa_params[1]['npoint'])
        
        # === Stage 4 ===
        self.print("[Server] SA4 연산 중...")
        l4_features = self._he_rsconv_forward(l3_features, enc_rel['s4'], plain_idx['s4'], self.weights['s4'],
                                              sa_params[3]['npoint'], sa_params[3]['nsample'], sa_params[2]['npoint'])
        
        # === Global Sum Pooling ===
        self.print("[Server] Global Pooling 및 FC 계층 연산 중...")
        # 256개의 점(npoint)을 하나로 합칩니다.
        final_npoint = sa_params[3]['npoint']

        S_global = np.ones((final_npoint, 1)).tolist()
        
        enc_global = [feat.matmul(S_global) for feat in l4_features]
        # ========== 현재 사용 불가 ============
        # 1. 초고속 로그 합산 연산 (Depth 0 소모, 회전 4회)
        # 이미 정의해둔 _logarithmic_shift_and_add 함수를 재활용합니다.
        # enc_global_shifted = [self._logarithmic_shift_and_add(feat, final_npoint) for feat in l4_features]
        
        # 2. 극강의 1D 압축 행렬 (Depth 1 소모, 회전 1회)
        # 인덱스 0번의 값만 쏙 빼서 크기가 1인 암호문으로 압축합니다.
        # C_global = np.zeros((final_npoint, 1))
        # C_global[0, 0] = 1.0
        # C_global_list = C_global.tolist()
        # enc_global = [feat.matmul(C_global_list) for feat in enc_global_shifted]
        # ===================

        # === Final Fully Connected Layer ===
        # enc_global_feature @ fc_w + fc_b
        enc_emb = []
        for j in range(sa_params[-1]["out_ch"]):
            channel_j = enc_global[0] * self.weights['fc_w'][0, j].item()
            for i in range(1, CONFIG["MODEL"]["feature_dim"]):
                channel_j += enc_global[i] * self.weights['fc_w'][i, j].item()
            channel_j += self.weights['fc_b'][j].item()
            enc_emb.append(channel_j)
        
        self.print("[Server] Feature Extraction 완료!")

        end_time = time.time()
        self.print(f"[Server] Server Total Time: {end_time - start_time:.4f}")
        
        return enc_emb