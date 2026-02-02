import torch

"""
원래는 GPU를 이용해서 계산하는 코드 (연산량 많음)
해당 코드는 CPU로 계산하는 코드 (느림)

Code by Gemini
"""

def square_distance(src, dst):
    """
    두 점 집합 간의 유클리드 거리 제곱 계산 (Pure PyTorch)
    src: (B, N, C)
    dst: (B, M, C)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def farthest_point_sample(xyz, npoint):
    """
    가장 먼 점 샘플링 (FPS) - CPU 호환 구현
    xyz: (B, N, 3)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        
    return centroids

def index_points(points, idx):
    """
    인덱스에 해당하는 점 추출
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball Query (반경 내 이웃 찾기) - CPU 호환 구현
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    # 거리 계산 (메모리 소모가 크므로 주의)
    sqrdists = square_distance(new_xyz, xyz)
    
    # 반경 내의 점들 마스킹
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius ** 2] = N # 반경 밖은 dummy index(N) 처리
    
    # 랜덤하게 nsample 개 선택 (간소화된 방식) 또는 앞의 nsample개 선택
    # 실제로는 정렬 후 자르는 방식을 많이 씀
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    # 부족한 점 처리는(복제 등) 추가 로직 필요하지만, 여기선 단순히 첫 번째 점으로 채움
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint: 샘플링할 중심점 개수 (예: 2048)
        radius: Ball Query 반경 (예: 0.1)
        nsample: 각 그룹당 포인트 개수 (예: 32)
        xyz: (B, 3, N) - 입력 좌표
        points: (B, C, N) - 입력 특징 (없으면 None)
    Output:
        new_xyz: (B, 3, npoint) - 샘플링된 중심점
        grouped_xyz: (B, 3, npoint, nsample) - 그룹핑된 좌표
        grouped_points: (B, C+3, npoint, nsample) - 그룹핑된 특징 (좌표 포함 가능)
    """
    B, C, N = xyz.shape
    
    # 1. Sampling Layer: FPS (Farthest Point Sampling)
    # 전체 점(N) 중에서 npoint개의 중심점을 뽑습니다.
    # xyz를 (B, N, 3)으로 transpose해서 넘겨줘야 함 (구현에 따라 다름)
    xyz_t = xyz.permute(0, 2, 1) # (B, N, 3)
    fps_idx = farthest_point_sample(xyz_t, npoint) # (B, npoint)
    
    # 중심점 좌표 추출
    new_xyz = index_points(xyz_t, fps_idx) # (B, npoint, 3)
    new_xyz_trans = new_xyz.permute(0, 2, 1) # (B, 3, npoint) - 리턴용

    # 2. Grouping Layer: Ball Query
    # 각 중심점 주변 radius 내의 점들을 nsample개 찾습니다.
    # idx: (B, npoint, nsample) - 이웃 점들의 인덱스
    idx = query_ball_point(radius, nsample, xyz_t, new_xyz)
    
    # 3. Grouping: 인덱스를 이용해 실제 좌표와 특징을 모음
    # grouped_xyz: (B, npoint, nsample, 3)
    grouped_xyz = index_points(xyz_t, idx) 
    grouped_xyz = grouped_xyz.permute(0, 3, 1, 2) # (B, 3, npoint, nsample)
    
    # 4. Feature Gathering
    # 특징 벡터(points)가 있다면 똑같이 그룹핑합니다.
    grouped_points = grouped_xyz # 기본적으로 좌표를 특징으로 사용
    
    if points is not None:
        # points: (B, C, N) -> (B, N, C)
        points_t = points.permute(0, 2, 1)
        grouped_features = index_points(points_t, idx) # (B, npoint, nsample, C)
        grouped_features = grouped_features.permute(0, 3, 1, 2) # (B, C, npoint, nsample)
        
        # 좌표 정보와 기존 특징을 합칠 수도 있고, 기존 특징만 쓸 수도 있음
        # PointFace의 RSConv는 좌표 관계를 다시 계산하므로 여기선 features만 리턴하거나 합침
        grouped_points = grouped_features

    return new_xyz_trans, grouped_xyz, grouped_points

# --- 의존성 헬퍼 함수 (앞서 설명한 내용의 요약) ---
def index_points(points, idx):
    """
    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points