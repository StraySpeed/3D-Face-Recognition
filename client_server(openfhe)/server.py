import torch, pickle, tempfile
import numpy as np
import openfhe
import os, glob, time
from logger import get_logger
from config import CONFIG
from model.pointface import PointFaceNet

def deserialize_openfhe_from_bytes(data):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(data)
        temp_path = tmp.name
    obj, _ = openfhe.DeserializeCiphertext(temp_path, openfhe.BINARY)
    os.remove(temp_path)
    return obj

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

    def load_context(self, key_dir):
        """ 외부에서 공개 키를 가져오기 (비밀키 X) """
        if not os.path.exists(key_dir):
            self.print("[Server] Error: Key directory not found.")
            return False
        
        self.ctx, _ = openfhe.DeserializeCryptoContext(os.path.join(key_dir, "cryptocontext.txt"), openfhe.BINARY)
        # 서버는 연산을 위해 다중/회전 키만 로드
        self.ctx.DeserializeEvalMultKey(os.path.join(key_dir, "key-eval-mult.txt"), openfhe.BINARY)
        self.ctx.DeserializeEvalAutomorphismKey(os.path.join(key_dir, "key-eval-rot.txt"), openfhe.BINARY)

        # 테스트를 위해 비밀 키를 로드하기
        self.secret_key, _ = openfhe.DeserializePrivateKey(os.path.join(key_dir, "key-secret.txt"), openfhe.BINARY)
        
        self.print("[Server] OpenFHE Eval Keys loaded successfully.")
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
                
                # 역직렬화: 바이트 문자열들을 다시 리스트로 복원
                enc_vec_list = [deserialize_openfhe_from_bytes(b) for b in serialized_list]
                self.encrypted_gallery[identity] = enc_vec_list
                
            except Exception as e:
                self.print(f"Error loading {identity}.pkl: {e}")
            
        self.print(f"[Server] Loaded {len(self.encrypted_gallery)} encrypted identities.")
        return True

    def recognize_encrypted_id(self, enc_data, id):
        """
        # Server - 암호화된 데이터와 매칭 수행
        
        1개의 암호문을 이용하여 매칭함
        
        :return: score^2 - Threshold^2 * Random Positive Value
        """
        self.print("[Server Side Start]")
        start_time = time.time()
        enc_emb_list = self.he_model.get_embedding_enc(enc_data)
        emb_time = time.time()
        self.print(f"[Server] 1. Embedding Time(enc): {emb_time - start_time:.4f}")
        enc_gallery_list = self.encrypted_gallery.get(id)


        len_emb = len(enc_emb_list)
        len_gal = len(enc_gallery_list)
        
        if len_emb != len_gal:
            self.print(f"[Server_Warning] 차원 불일치! 모델 출력({len_emb}), 갤러리({len_gal})")
            
        enc_emb_single = enc_emb_list[0]
        enc_gallery_single = enc_gallery_list[0]

        # [128번 루프 제거 -> 요소별 곱셈 1회 + 로그 덧셈으로 대체]
        enc_mult = self.ctx.EvalMult(enc_emb_single, enc_gallery_single)
        enc_score = self.he_model._logarithmic_shift_and_add_standard(enc_mult, CONFIG["MODEL"]["feature_dim"])

        enc_mag_mult = self.ctx.EvalMult(enc_emb_single, enc_emb_single)
        enc_mag_sq = self.he_model._logarithmic_shift_and_add_standard(enc_mag_mult, CONFIG["MODEL"]["feature_dim"])

        # 제곱 판별식 계산 (나눗셈 우회)
        threshold_sq = float(PointFaceServer.THRESHOLD ** 2)
        # Score 제곱 (C x C) -> Depth 21 도달
        enc_score_sq = self.ctx.EvalMult(enc_score, enc_score)
        
        # Mag_sq에 평문 임계값 곱셈 (C x P) -> Depth 21 도달
        enc_mag_sq_th = self.ctx.EvalMult(enc_mag_sq, threshold_sq)
        enc_diff = self.ctx.EvalSub(enc_score_sq, enc_mag_sq_th)
        
        # 랜덤 블라인딩 처리 (C x P) -> 최종 Depth 22
        # 클라이언트가 원래 점수를 역산할 수 없도록 무작위 양수 팩터 곱셈
        #blind_factor = float(np.random.uniform(10.0, 1000.0))
        #enc_blinded_result = self.ctx.EvalMult(enc_diff, blind_factor)
        enc_blinded_result = enc_diff

        end_time = time.time()

        # 원래 서버에는 비밀 키가 없음 - 점수를 알 수 없음
        # 여기서는 편의를 위해 서버에 비밀 키를 넣음
        ptxt_diff = self.ctx.Decrypt(self.secret_key, enc_diff)
        ptxt_diff.SetLength(1)
        val_diff = ptxt_diff.GetRealPackedValue()[0]

        ptxt_score_sq = self.ctx.Decrypt(self.secret_key, enc_score_sq)
        ptxt_score_sq.SetLength(1)
        val_score_sq = ptxt_score_sq.GetRealPackedValue()[0]

        ptxt_mag_sq = self.ctx.Decrypt(self.secret_key, enc_mag_sq)
        ptxt_mag_sq.SetLength(1)
        val_mag_sq = ptxt_mag_sq.GetRealPackedValue()[0]
        self.print(f"[Server_Debug] {id} Matching Score : {val_diff:.4f}, Similarity : {val_score_sq / val_mag_sq:.4f}")

        # 최종 결과
        self.print(f"[Server] 2. 1:1 Matching (C x C) Time: {end_time - emb_time:.4f}")
        self.print(f"[Server] Total Matching Time: {end_time - start_time:.4f}")
        self.print(f"[Server Side Finished]")
        
        return enc_blinded_result



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
        
        self.print("[Server] Extracting and Folding Weights for HE Server...")
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
        """ 1D 패킹을 위한 단일 Grouping 행렬 생성 함수 """
        idx_flat = plain_idx.T.flatten() 
        G = np.zeros((len(idx_flat), N_prev))
        for i, idx in enumerate(idx_flat): 
            G[i, idx] = 1.0
        return G
    
    # ==========================================================
    # 2의 거듭제곱 조합형 초고속 회전 함수
    # ==========================================================
    def _custom_rotate(self, ctxt, shift):
        if shift == 0:
            return ctxt
            
        sign = 1 if shift > 0 else -1
        abs_shift = abs(shift)
        
        # NAF (Non-Adjacent Form) 알고리즘 적용
        naf = []
        while abs_shift > 0:
            if abs_shift % 2 == 1:
                z = 2 - (abs_shift % 4)
                naf.append(z)
                abs_shift -= z
            else:
                naf.append(0)
            abs_shift //= 2
            
        result = ctxt
        for i, z in enumerate(naf):
            if z != 0:
                # z는 1 또는 -1
                power_of_2_shift = sign * z * (2 ** i)
                result = self.ctx.EvalRotate(result, power_of_2_shift)
                
        return result

    def _he_matmul_openfhe(self, ctxt, plain_matrix):
        N, M = plain_matrix.shape
        max_dim = max(N, M)
        padded_matrix = np.zeros((max_dim, max_dim))
        padded_matrix[:N, :M] = plain_matrix
        
        result_ctxt = None
        for i in range(max_dim):
            diag = np.diagonal(padded_matrix, offset=i)
            if np.all(diag == 0): continue
            full_diag = np.zeros(max_dim)
            full_diag[:len(diag)] = diag
            ptxt_diag = self.ctx.MakeCKKSPackedPlaintext(full_diag.tolist())
            
            # EvalRotate 대신 안전한 custom_rotate 사용
            rotated_ctxt = self._custom_rotate(ctxt, i)
            
            term = self.ctx.EvalMult(rotated_ctxt, ptxt_diag)
            result_ctxt = term if result_ctxt is None else self.ctx.EvalAdd(result_ctxt, term)
            
        for i in range(1, max_dim):
            diag = np.diagonal(padded_matrix, offset=-i)
            if np.all(diag == 0): continue
            full_diag = np.zeros(max_dim)
            full_diag[i:i+len(diag)] = diag
            ptxt_diag = self.ctx.MakeCKKSPackedPlaintext(full_diag.tolist())
            
            # EvalRotate 대신 안전한 custom_rotate 사용
            rotated_ctxt = self._custom_rotate(ctxt, -i)
            
            term = self.ctx.EvalMult(rotated_ctxt, ptxt_diag)
            result_ctxt = term if result_ctxt is None else self.ctx.EvalAdd(result_ctxt, term)
            
        return result_ctxt

    def _he_matmul_openfhe_batched(self, ctxt_list, plain_matrix):
        """
        Baby-Step Giant-Step (BSGS) + Batched Matrix Multiplication
        """
        N, M = plain_matrix.shape
        max_dim = max(N, M)
        padded_matrix = np.zeros((max_dim, max_dim))
        padded_matrix[:N, :M] = plain_matrix
        
        # CKKS 슬롯 전체 크기 (register_enc.py의 BatchSize와 완벽히 일치해야 함)
        SLOT_COUNT = 16384 

        # 1. 값이 0이 아닌 유효한 대각선 오프셋만 수집
        unique_offsets = []
        for i in range(-max_dim + 1, max_dim):
            if np.any(np.diagonal(padded_matrix, offset=i)):
                unique_offsets.append(i)
                
        if not unique_offsets:
            return ctxt_list # 예외 처리

        # 2. Giant Step 크기 (g) 동적 설정 (루트 N이 이론상 최적)
        g = int(np.ceil(np.sqrt(len(unique_offsets))))
        g = max(1, g) 

        # 3. 오프셋을 j(Giant)와 k(Baby)로 그룹화
        # offset = j * g + k
        groups = {}
        baby_k_set = set()
        for offset in unique_offsets:
            j = offset // g
            k = offset % g
            if j not in groups:
                groups[j] = []
            groups[j].append((k, offset))
            baby_k_set.add(k)

        # =========================================================
        # 4. Baby Steps 연산 (모든 채널에 대해 k만큼 미리 회전)
        # =========================================================
        # baby_steps[c][k] = ctxt_list[c] rotated by k
        baby_steps = {c: {} for c in range(len(ctxt_list))}
        for k in baby_k_set:
            for c, ctxt in enumerate(ctxt_list):
                baby_steps[c][k] = self._custom_rotate(ctxt, k)

        # =========================================================
        # 5. Giant Steps 및 Inner Sum 누적
        # =========================================================
        result_ctxts = [None] * len(ctxt_list)

        for j, items in groups.items():
            inner_sum = [None] * len(ctxt_list)
            
            for k, offset in items:
                # 대각선 추출 및 패딩
                diag = np.diagonal(padded_matrix, offset=offset)
                full_diag = np.zeros(SLOT_COUNT)
                
                if offset >= 0:
                    full_diag[:len(diag)] = diag
                else:
                    full_diag[-offset:-offset+len(diag)] = diag
                    
                # 평문을 -j*g 만큼 미리 역회전!
                # FHE의 회전을 보상하기 위해 numpy 상에서 오른쪽(j*g)으로 시프트시킴
                shifted_plain = np.roll(full_diag, j * g)
                ptxt_diag = self.ctx.MakeCKKSPackedPlaintext(shifted_plain.tolist())
                
                # 채널별 Inner Sum 누적
                for c in range(len(ctxt_list)):
                    term = self.ctx.EvalMult(baby_steps[c][k], ptxt_diag)
                    if inner_sum[c] is None:
                        inner_sum[c] = term
                    else:
                        inner_sum[c] = self.ctx.EvalAdd(inner_sum[c], term)
                        
            # 생성된 Inner Sum을 j*g 만큼 Giant Step 훌쩍 뛰어서 회전!
            if j * g != 0:
                for c in range(len(ctxt_list)):
                    inner_sum[c] = self._custom_rotate(inner_sum[c], j * g)
                    
            # 최종 결과에 합산
            for c in range(len(ctxt_list)):
                if result_ctxts[c] is None:
                    result_ctxts[c] = inner_sum[c]
                else:
                    result_ctxts[c] = self.ctx.EvalAdd(result_ctxts[c], inner_sum[c])

        return result_ctxts

    # 초고속 로그 합산 연산 (Rotation 기반 덧셈)
    def _logarithmic_shift_and_add_standard(self, ctxt, steps_length):
        steps = int(np.log2(steps_length))
        result = ctxt
        for i in range(steps):
            shift = 2 ** i
            rotated = self.ctx.EvalRotate(result, shift)
            result = result + rotated
        return result


    def _logarithmic_shift_and_add_interleaved(self, ctxt, npoint, nsample):
        steps = int(np.log2(nsample))
        result = ctxt
        for i in range(steps):
            shift = npoint * (2 ** i)  # 이웃(1칸)이 아니라 npoint칸 간격으로 뛰며 덧셈
            rotated = self._custom_rotate(result, shift)
            result = self.ctx.EvalAdd(result, rotated)
        return result

    # ==========================================
    # 단일 SA(Set Abstraction) 블록 연산 (1D 패킹 전용)
    # ==========================================
    def _he_rsconv_forward(self, enc_features_list, enc_rel_list, plain_idx, w_dict, npoint, nsample, N_prev):
        # 1. MLP 1 (10채널 -> hid_ch) 
        # 1D 패킹이므로 k루프 없이 암호문 전체를 한 번에 연산
        enc_x = []
        for j in range(w_dict['w1'].shape[1]):
            channel_j = self.ctx.EvalMult(enc_rel_list[0], float(w_dict['w1'][0, j].item()))
            for i in range(1, 10): 
                term = self.ctx.EvalMult(enc_rel_list[i], float(w_dict['w1'][i, j].item()))
                channel_j = self.ctx.EvalAdd(channel_j, term)
            channel_j = self.ctx.EvalAdd(channel_j, float(w_dict['b1'][j].item()))
            enc_x.append(channel_j)

        # 2. Hermite Activation
        for j in range(len(enc_x)):
            enc_x_sq = self.ctx.EvalMult(enc_x[j], enc_x[j])
            enc_x_lin = self.ctx.EvalMult(enc_x[j], float(w_dict['hb_new']))
            term_add = self.ctx.EvalAdd(enc_x_sq, enc_x_lin)
            enc_x[j] = self.ctx.EvalAdd(term_add, float(w_dict['hc_new']))
            
        # 3. MLP 2 (hid_ch -> in_ch)
        enc_weights = []
        for j in range(w_dict['w2'].shape[1]):
            channel_j = self.ctx.EvalMult(enc_x[0], float(w_dict['w2'][0, j].item()))
            for i in range(1, len(enc_x)): 
                term = self.ctx.EvalMult(enc_x[i], float(w_dict['w2'][i, j].item()))
                channel_j = self.ctx.EvalAdd(channel_j, term)
            channel_j = self.ctx.EvalAdd(channel_j, float(w_dict['b2'][j].item()))
            enc_weights.append(channel_j)
            
        # 4. Feature Grouping & Multiplication
        if enc_features_list is not None:
            G = self._create_grouping_matrix(plain_idx, N_prev)
            # Batched 함수 1번 호출로 64개 채널 일괄 처리
            enc_grouped = self._he_matmul_openfhe_batched(enc_features_list, G)
            
            enc_weighted = [self.ctx.EvalMult(enc_grouped[i], enc_weights[i]) for i in range(len(enc_weights))]
        else:
            enc_weighted = enc_weights
            
        # 5. Sum Pooling (Rotation 기반 덧셈 적용)
        enc_pooled = [self._logarithmic_shift_and_add_interleaved(feat, npoint, nsample) for feat in enc_weighted]

        # 6. Final Linear Projection (in_ch -> out_ch)
        enc_new_features = []
        for j in range(w_dict['w_lin'].shape[1]):
            channel_j = self.ctx.EvalMult(enc_pooled[0], float(w_dict['w_lin'][0, j].item()))
            for i in range(1, len(enc_pooled)): 
                term = self.ctx.EvalMult(enc_pooled[i], float(w_dict['w_lin'][i, j].item()))
                channel_j = self.ctx.EvalAdd(channel_j, term)
            channel_j = self.ctx.EvalAdd(channel_j, float(w_dict['b_lin'][j].item()))
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

        # 통신 오버헤드 무시 (통신 오버헤드 - 약 100 sec)
        # 그대로 가져왔다고 가정
        #for stage in ['s1', 's2', 's3', 's4']:
        #    enc_rel_channels = []
        #    for c in range(10):
        #        enc_rel_neighbors = [ts.ckks_vector_from(self.ctx, vec_bytes) for vec_bytes in enc_data['rel'][stage][c]]
        #        enc_rel_channels.append(enc_rel_neighbors)
        #    enc_rel[stage] = enc_rel_channels
        
        enc_rel = enc_data['rel']
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
        enc_global_shifted = [self._logarithmic_shift_and_add_standard(feat, final_npoint) for feat in l4_features]

        # 인덱스 0번 값 추출용 극소형 압축 행렬
        C_global = np.zeros((final_npoint, 1))
        C_global[0, 0] = 1.0
        enc_global = self._he_matmul_openfhe_batched(enc_global_shifted, C_global)

        # ====================================================================
        # [혁신 최적화 1] Pre-Packing & Tiling (256개의 암호문을 1개로 압축)
        # ====================================================================
        dim = sa_params[-1]["out_ch"] # 256
        self.print(f"[Server] FC 계층을 위한 {dim} -> 1 Pre-Packing 및 BSGS 준비 중...")
        
        packed_x = None
        for i in range(dim):
            # enc_global[i]는 Batched Matmul(C_global)에 의해 이미 0번 외의 슬롯이 0으로 클리어된 상태입니다.
            # 따라서 곱셈(Masking) 없이 단 1번의 회전만으로 초고속 패킹이 가능합니다!
            rotated_ctxt = self._custom_rotate(enc_global[i], -i)
            if packed_x is None:
                packed_x = rotated_ctxt
            else:
                packed_x = self.ctx.EvalAdd(packed_x, rotated_ctxt)
                
        # [핵심] FHE 행렬곱에서 인덱스가 범위를 넘어가는(Wrap-around) 현상을 막기 위한 "Tiling"
        # 0~255 범위의 데이터를 256~511 범위에 똑같이 복제해 둡니다.
        packed_x_tiled = self.ctx.EvalAdd(packed_x, self._custom_rotate(packed_x, -dim))

        # ====================================================================
        # [혁신 최적화 2] Final Fully Connected Layer (BSGS 대각선 행렬곱)
        # ====================================================================
        self.print(f"[Server] FC 계층 연산 중 (BSGS O(√N) 적용)...")
        W = self.weights['fc_w'].T  # (256, 256) 형태의 가중치 행렬 전치
        b = self.weights['fc_b']    # (256,) 편향
        
        N = dim
        g = int(np.ceil(np.sqrt(N))) # Giant 보폭 = 16
        SLOT_COUNT = 16384           # parameters.SetBatchSize 와 동일해야 함
        
        # 1. Baby steps: 입력 x를 k만큼 미리 회전 (16회)
        baby_x = {}
        for k in range(g):
            baby_x[k] = self._custom_rotate(packed_x_tiled, k)
            
        result_ctxt = None
        
        # 2. Giant steps & Inner Sum
        for j in range(g):
            inner_sum = None
            for k in range(g):
                offset = j * g + k
                if offset >= N:
                    continue
                    
                # W 행렬의 대각선 추출 (d_offset[i] = W[i, (i + offset) % N])
                diag = np.zeros(N)
                for i in range(N):
                    diag[i] = W[i, (i + offset) % N]
                    
                full_diag = np.zeros(SLOT_COUNT)
                full_diag[:N] = diag
                
                # 평문 역회전 (-j*g) -> FHE 회전을 상쇄하기 위해 numpy를 반대로 밉니다
                shifted_diag = np.roll(full_diag, j * g)
                ptxt_diag = self.ctx.MakeCKKSPackedPlaintext(shifted_diag.tolist())
                
                term = self.ctx.EvalMult(baby_x[k], ptxt_diag)
                
                if inner_sum is None:
                    inner_sum = term
                else:
                    inner_sum = self.ctx.EvalAdd(inner_sum, term)
                    
            if inner_sum is not None:
                # 3. Giant step 회전 (j*g) (16회)
                rotated_inner = self._custom_rotate(inner_sum, j * g)
                if result_ctxt is None:
                    result_ctxt = rotated_inner
                else:
                    result_ctxt = self.ctx.EvalAdd(result_ctxt, rotated_inner)
                    
        # 4. Bias(편향) 더하기
        full_b = np.zeros(SLOT_COUNT)
        full_b[:N] = b
        ptxt_b = self.ctx.MakeCKKSPackedPlaintext(full_b.tolist())
        packed_emb = self.ctx.EvalAdd(result_ctxt, ptxt_b)
        
        self.print("[Server] Feature Extraction 및 FC 연산 완료!")

        end_time = time.time()
        self.print(f"[Server] Server Total Time: {end_time - start_time:.4f}")
        
        # BSGS 결과물이 이미 1개의 암호문으로 예쁘게 패킹되어 있으므로 리스트로 감싸서 바로 반환!
        return [packed_emb]