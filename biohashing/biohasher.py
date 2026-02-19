import numpy as np

class BioHasher:
    def __init__(self, input_dim=512, code_length=512):
        """
        :param input_dim: PointFace 임베딩 차원
        :param code_length: 생성할 BioCode 길이
        """
        self.input_dim = input_dim
        self.code_length = code_length

    def generate_projection_matrix(self, seed):
        """
        사용자별 시드(Seed)를 이용해 고유한 Projection 행렬 생성
        
        이 행렬이 'Token' 역할

        :param seed: seed
        """
        # 시드 고정으로 항상 동일한 랜덤 행렬 생성
        rng = np.random.default_rng(seed)
        
        # 정규분포(Gaussian) 랜덤 행렬 생성
        # (input_dim, code_length)
        projection_matrix = rng.standard_normal((self.input_dim, self.code_length))
        
        # 행렬 직교화 (Gram-Schmidt)
        q, _ = np.linalg.qr(projection_matrix)
        return q

    def get_biocode(self, embedding, seed):
        """
        Feature + Token -> BioCode

        이진화해서 암호화함

        :param embedding: (512,) Normalized Face Vector
        :param seed: Seed
        """
        # 1. 시드에 맞는 랜덤 Projection 행렬 생성
        proj_mat = self.generate_projection_matrix(seed)
        
        # 2. Projection: Feature vector와 행렬 곱
        # (1, 512) @ (512, 512) -> (1, 512)
        projected = np.dot(embedding, proj_mat)
        
        # 3. 이진화 (Binarization)
        # Threshold보다 크면 1, 아니면 0
        # 여기서는 0을 사용함
        biocode = (projected > 0).astype(int)
        
        return biocode

    @staticmethod
    def hamming_distance(code1, code2):
        """
        두 BioCode 간의 해밍 거리 계산 (0.0 ~ 1.0)

        값이 0에 가까울수록 유사함(코사인 유사도와 반대)

        0.0: 본인

        0.5: (일반적으로) 다른 사람
        
        1.0: 완전히 다른 사람
        """
        if len(code1) != len(code2):
            raise ValueError("Code lengths must match")
            
        # XOR 연산으로 다른 비트 개수 세기
        mismatch_count = np.sum(code1 != code2)
        
        # 전체 길이로 나누어 정규화 (Normalized Hamming Distance)
        return mismatch_count / len(code1)