import numpy as np
import torch
from config import CONFIG

class PointCloudAugmentation:
    def __init__(self, num_points=CONFIG["MODEL"]["num_points"], train=True):
        self.num_points = num_points
        self.train = train

    def __call__(self, points, train=True):
        """
        :param points: (N, 3) numpy array
        :param train: 학습 모드 여부
        """
        # 1. Resampling (5000개로 맞춤)
        points = self.random_sample(points)
        
        # 2. Normalization (Unit Sphere) 
        points = self.normalize(points)

        if train and self.train:
            # 3. Data Augmentation (각 기법을 독립적인 확률로 적용)
            if np.random.random() > 0.5:
                points = self.random_scale(points)
            if np.random.random() > 0.5:
                points = self.random_rotate(points)
            if np.random.random() > 0.5:
                points = self.random_translate(points)
            if np.random.random() > 0.5:
                points = self.random_jitter(points)      # [신규] 가우시안 노이즈
            if np.random.random() > 0.5:
                points = self.random_dropout(points)

        points = self.morton_sort(points)
                
        # (N, 3) -> (3, N) for PyTorch Conv1d
        return torch.from_numpy(points.astype(np.float32)).transpose(1, 0)

    def random_sample(self, points):
        # Random Sampling (논문은 Farthest Point Sampling 사용했으나, 너무 연산 많아져서 여기서는 랜덤 샘플링)
        choice = np.random.choice(points.shape[0], self.num_points, replace=True)
        return points[choice, :]

    def normalize(self, points):
        centroid = np.mean(points, axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        return points / m

    def random_scale(self, points):
        # Range [-0.66, 1.5] -> Scale factor로 변환 (0.66 ~ 1.5로 가정)
        # 논문 표기는 [-0.66, 1.5]이나 스케일 팩터는 양수여야 하므로 
        # 통상적인 0.8 ~ 1.25 범위 또는 논문 의도를 살려 0.66 ~ 1.5 배율 적용
        scale = np.random.uniform(0.66, 1.5)
        return points * scale

    def random_translate(self, points):
        # Range [-0.2, 0.2] 
        shift = np.random.uniform(-0.2, 0.2, size=(1, 3))
        return points + shift

    def random_rotate(self, points):
        # Yaw: [-45, 45], Pitch: [-15, 15] 
        theta_y = np.random.uniform(-45, 45) * np.pi / 180
        theta_p = np.random.uniform(-15, 15) * np.pi / 180
        
        # Rotation Matrix (Yaw)
        rot_y = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])
        
        # Rotation Matrix (Pitch - X축 회전으로 근사하거나 논문에 맞춰 축 설정)
        rot_p = np.array([
            [1, 0, 0],
            [0, np.cos(theta_p), -np.sin(theta_p)],
            [0, np.sin(theta_p), np.cos(theta_p)]
        ])
        
        rotation_matrix = np.dot(rot_y, rot_p)
        return np.dot(points, rotation_matrix)
    
    def random_jitter(self, points, std=0.02, clip=0.05):
        """ nsample=4 환경에서도 부피를 인식하도록 강한 노이즈 삽입 """
        noise = np.clip(np.random.normal(0, std, points.shape), -clip, clip)
        return points + noise

    def random_dropout(self, points, min_drop=0.1, max_drop=0.2):
        """ 무작위로 10~20%의 점을 지워, 이웃 점이 부족한 열악한 상황을 시뮬레이션 """
        dropout_ratio = np.random.uniform(min_drop, max_drop)
        num_drop = int(dropout_ratio * points.shape[0])
        
        # 삭제되지 않고 남길 점들 선택
        keep_indices = np.random.choice(points.shape[0], points.shape[0] - num_drop, replace=False)
        kept_points = points[keep_indices, :]
        
        # 1024개를 맞추기 위해 남은 점들 중에서 랜덤 복제하여 채움
        dup_indices = np.random.choice(kept_points.shape[0], num_drop, replace=True)
        dup_points = kept_points[dup_indices, :]
        
        return np.vstack([kept_points, dup_points])

    def morton_sort(self, points):
        """ Client 전처리와 동일하게 1D 배열 인덱스를 정렬 """
        coords = points[:, :3]
        p_min = np.min(coords, axis=0)
        p_max = np.max(coords, axis=0)
        
        norm_points = (coords - p_min) / (p_max - p_min + 1e-8)
        quantized = np.clip(np.floor(norm_points * 1024), 0, 1023).astype(np.uint32)
        
        def part1by2(n):
            n &= 0x000003ff
            n = (n ^ (n << 16)) & 0xff0000ff
            n = (n ^ (n <<  8)) & 0x0300f00f
            n = (n ^ (n <<  4)) & 0x030c30c3
            n = (n ^ (n <<  2)) & 0x09249249
            return n
        
        x = np.vectorize(part1by2)(quantized[:, 0])
        y = np.vectorize(part1by2)(quantized[:, 1])
        z = np.vectorize(part1by2)(quantized[:, 2])
        
        codes = (z << 2) | (y << 1) | x
        return points[np.argsort(codes)]