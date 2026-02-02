import os
import numpy as np
import shutil

"""
가상 데이터 셋 생성기

근데 그냥 랜덤 포인트를 생성하기만 함 (얼굴 모양 아님)

Code by Gemini
"""

def generate_dummy_data(root_dir, num_identities=10, samples_per_id=5, num_points=6000):
    """
    PointFace 학습을 위한 가상 데이터셋 생성기
    
    Args:
        root_dir: 데이터셋이 저장될 루트 경로
        num_identities: 생성할 가상의 사람 수 (Class 개수)
        samples_per_id: 한 사람당 생성할 3D 스캔 데이터 개수 (Pair 구성을 위해 최소 2개 이상 권장)
        num_points: 각 스캔 데이터의 점 개수 (Loader에서 5000개로 리샘플링함)
    """
    
    # 1. 기존 폴더가 있다면 삭제 후 새로 생성 (초기화)
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)
    
    print(f"Generating dummy dataset at '{root_dir}'...")
    
    for person_idx in range(num_identities):
        # 사람(Identity)별 폴더 생성 (예: id_000, id_001 ...)
        person_id = f"id_{person_idx:03d}"
        person_dir = os.path.join(root_dir, person_id)
        os.makedirs(person_dir)
        
        for sample_idx in range(samples_per_id):
            # 2. 가상의 3D 포인트 클라우드 생성 (Random XYZ)
            # 정규화되지 않은 임의의 좌표 (-100 ~ 100 범위 가정)
            raw_points = np.random.uniform(-100, 100, size=(num_points, 3)).astype(np.float32)
            
            # 가짜 구(Sphere) 형태를 띄게 만들고 싶다면:
            points = np.random.randn(num_points, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
            # 3. .npy 파일로 저장
            file_name = f"scan_{sample_idx:03d}.npy"
            save_path = os.path.join(person_dir, file_name)
            np.save(save_path, raw_points)
            
    print(f"Done! Created {num_identities} identities with {samples_per_id} samples each.")

# --- 실행 부분 ---
if __name__ == "__main__":
    # 데이터셋 생성 경로 설정
    DATASET_PATH = "./dummy_dataset"
    
    # 데이터 생성 실행
    generate_dummy_data(DATASET_PATH, num_identities=5, samples_per_id=4)
    
    # --- 생성된 데이터 확인 (검증) ---
    print("\n[Data Verification]")
    sample_file = os.path.join(DATASET_PATH, "id_000", "scan_000.npy")
    if os.path.exists(sample_file):
        data = np.load(sample_file)
        print(f"Sample file: {sample_file}")
        print(f"Shape: {data.shape}")  # (6000, 3) 출력 예상
        print(f"Data Type: {data.dtype}")
        print("First 5 points:\n", data[:5])
    else:
        print("Failed to verify data.")