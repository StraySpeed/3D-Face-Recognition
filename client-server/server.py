import torch
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
            
        self.print(f"[Server] Loading embeddings from '{load_dir}'...")

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
        enc_score = enc_emb_list[0] * plain_gallery_vec[0].item()
        for i in range(1, CONFIG["MODEL"]["feature_dim"]):
            enc_score += enc_emb_list[i] * plain_gallery_vec[i].item()
        
        # 3. 임계값(Threshold) 빼고 블라인딩 무작위 수 곱하기
        enc_score = enc_score.sub(PointFaceServer.THRESHOLD).mul(np.random.uniform(10.0, 1000.0))
        end_time = time.time()
        self.print(f"[Server] 2. 1:1 Matching Time: {end_time - emb_time:.4f}")
        self.print(f"[Server] Total Matching Time: {end_time - start_time:.4f}")
        self.print(f"[Server Side Finished]")
        
        return enc_score.serialize()

    def recognize_encrypted_id(self, enc_data, id):
        """
        # Server - 암호화된 데이터와 매칭 수행
         
        ### (현재는 사용 불가함 - 암호화된 갤러리가 아니므로)
        
        :return: score - Threshold * Random Positive Value
        """

        self.print("[Server Side Start]")
        start_time = time.time()
        enc_emb = self.he_model.get_embedding_enc(enc_data)
        emb_time = time.time()
        self.print(f"[Server] 1. Embedding Time(enc): {emb_time - start_time:.4f}")

        # 2. 갤러리와 동형 연산 (내적) 수행
        enc_gallery_vec = self.encrypted_gallery.get(id)
            
        # 암호문-암호문 간의 매칭 연산 최적화
        # 1. Element-wise 곱셈: (1, dim) * (1, dim) -> (1, dim) 암호문
        enc_element_wise = enc_emb * enc_gallery_vec
        # 2. 평문 행렬 (dim, 1)을 행렬곱하여 내부 슬롯 합산(Summation) 유도 -> (1, 1) 암호문 생성
        plain_ones = np.ones((CONFIG["MODEL"]["feature_dim"], 1)).tolist()
        enc_score_tensor = enc_element_wise.mm(plain_ones)
        
        # 3. 임계값(Threshold) 빼고 블라인딩 무작위 수 곱하기
        enc_score = enc_score_tensor.sub(PointFaceServer.THRESHOLD).mul(np.random.uniform(10.0, 1000.0))
        end_time = time.time()
        self.print(f"[Server] 2. 1:1 Matching Time: {end_time - emb_time:.4f}")
        self.print(f"[Server] Total Matching Time: {end_time - start_time:.4f}")
        self.print(f"[Server Side Finished]")
        return enc_score.serialize()



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
            
            he_w[f's{i+1}'] = {
                'w1': w1, 'b1': b1, 'ha': ha, 'hb': hb, 'hc': hc,
                'w2': w2, 'b2': b2, 'w_lin': w_lin, 'b_lin': b_lin
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
            step1 = enc_x[j] * w_dict['ha'] + w_dict['hb']
            enc_x[j] = enc_x[j] * step1 + w_dict['hc']
            
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
        self.print("[Server Side Finished]")
        
        return enc_emb