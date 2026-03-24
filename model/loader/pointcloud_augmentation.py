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
            # 3. Data Augmentation
            if np.random.random() > 0.5:
                points = self.random_scale(points)
                points = self.random_rotate(points)
                points = self.random_translate(points)
                
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
        # Yaw: [-90, 90], Pitch: [-30, 30] 
        theta_y = np.random.uniform(-90, 90) * np.pi / 180
        theta_p = np.random.uniform(-30, 30) * np.pi / 180
        
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