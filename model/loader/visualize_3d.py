import numpy as np
import open3d as o3d

"""
.npy 3D 데이터를 시각적으로 확인하기 위한 코드

Code by Gemini
"""

def visualize_point_cloud(npy_path):
    # 1. .npy 파일 로드
    try:
        # (N, 3) 또는 (N, 6) 형태의 데이터
        data = np.load(npy_path)
        print(f"[DEBUG] Data Loaded: {data.shape}")
    except Exception as e:
        print(f"파일을 읽는 중 오류 발생: {e}")
        return

    # 2. XYZ 좌표만 추출 (N, 3)
    # 데이터가 (N, 6)인 경우 앞의 3개만 좌표로 사용
    xyz = data[:, :3]

    # 3. Open3D PointCloud 객체 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # 색상 입히기 (높이값 Z에 따라 색상 변화)
    # Z축 기준으로 그라데이션
    z_vals = xyz[:, 2]
    # 0~1 사이로 정규화
    colors = np.zeros_like(xyz)
    if z_vals.max() - z_vals.min() > 0:
        z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())
        colors[:, 0] = z_norm      # R (Red)
        colors[:, 1] = 1 - z_norm  # G (Green)
        colors[:, 2] = 0.5         # B (Blue)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 4. 시각화 실행
    print("뷰어 창이 열립니다. (마우스 드래그: 회전, 휠: 줌, Ctrl+드래그: 이동)")
    print("종료하려면 'Q'를 누르세요.")
    
    # 좌표축(Coordinate Frame)을 함께 표시 (사이즈 10)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries([pcd, axis])

# 사용 예시
if __name__ == "__main__":
    # 보고 싶은 파일 경로 입력
    # 예: 앞서 생성한 dummy_dataset의 파일
    target_file = "./dummy_dataset/id_000/scan_000.npy" 
    
    # 파일이 존재하면 시각화
    visualize_point_cloud(target_file)