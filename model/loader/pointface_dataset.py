from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from .pointcloud_augmentation import PointCloudAugmentation

class PointFaceDataset(Dataset):
    def __init__(self, data_root, train=True):
        """
        data_root: 데이터 폴더 (구조: root/ID/sample.npy)
        """
        self.data_root = data_root
        self.train = train
        self.transform = PointCloudAugmentation(num_points=5000)
        self.pairs = [] # (path_anchor, path_positive, label_idx)
        
        # 1. 데이터 로드 및 정리
        # subjects: {label_idx: [path1, path2, ...]}
        self.subjects = self._load_data(data_root)
        
        # 2. Pair Generation (Algorithm 1 Implementation)
        self._generate_pairs()
        
        print(f"[DEBUG] Dataset Loaded: {len(self.subjects)} subjects, {len(self.pairs)} pairs generated.")

    def _load_data(self, root):
        # 폴더 구조를 읽어서 ID별로 파일 경로를 저장
        subjects = {}
        # 폴더명이 ID (0, 1, 2...)
        if not os.path.exists(root):
            return {} # 빈 딕셔너리 반환

        for label_idx, folder_name in enumerate(sorted(os.listdir(root))):
            folder_path = os.path.join(root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
            if len(files) > 0:
                subjects[label_idx] = sorted(files)
        return subjects

    def _generate_pairs(self):
        """
        Pair Selection Strategy

        각 Subject에 대해 리스트를 순회하며 (i, i+1) 형태로 쌍을 생성
        """
        self.pairs = []
        for label_idx, file_list in self.subjects.items():
            n = len(file_list)
            if n < 1: continue
            
            # 데이터가 1개뿐인 경우 자기 자신을 Positive로 (예외 처리)
            if n == 1:
                self.pairs.append((file_list[0], file_list[0], label_idx))
                continue

            # 순환 연결 방식: (P_i, P_{(i) mod len + 1})
            for i in range(n):
                anchor = file_list[i]
                positive = file_list[(i + 1) % n] # 다음 인덱스 (마지막이면 처음으로)
                self.pairs.append((anchor, positive, label_idx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_anchor, path_positive, label = self.pairs[idx]
        
        # 1. 파일 로드 (여기서는 .npy 가정)
        # 실제 데이터 포맷에 따라 np.load, plyfile 등 사용
        pc_anchor = np.load(path_anchor)[:, :3] # (N, 3) XYZ 좌표만 사용
        pc_positive = np.load(path_positive)[:, :3]
        
        # 2. Augmentation & Preprocessing
        # Anchor와 Positive에 대해 서로 다른 랜덤 증강 적용
        tensor_anchor = self.transform(pc_anchor, train=self.train)
        tensor_positive = self.transform(pc_positive, train=self.train)
        
        return tensor_anchor, tensor_positive, label
    

def get_dataloader(data_root, batch_size=32, num_workers=4):
    # Dataset 인스턴스 생성
    dataset = PointFaceDataset(data_root=data_root, train=True)
    
    # DataLoader 생성
    # shuffle=True는 필수 (배치 내에서 Hardest Negative를 찾기 위해 섞여야 함)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, # GPU 전송 속도 향상
        drop_last=True   # 배치 크기가 일정해야 Loss 계산이 용이함
    )
    
    return loader