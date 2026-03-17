import torch

"""
원래는 GPU를 이용해서 계산하는 코드 (연산량 많음)
해당 코드는 CPU로 계산하는 코드 (느림)

Code by Gemini
"""

def square_distance(src, dst):
    """
    두 점 집합 간의 유클리드 거리 제곱 계산 (Pure PyTorch)
    
    :param src: (B, N, C)
    :param dst: (B, M, C)
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
    
    :param xyz: (B, N, 3)
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
    # 인덱스 넘치면 0으로 바꾸기
    idx[idx >= points.shape[1]] = 0
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

def sample_and_group(npoint, radius, nsample, xyz, points, precomputed_centroids=None, precomputed_indices=None):
    """
    :param precomputed_centroids: (B, 3, npoint) - 미리 계산된 중심점 좌표
    :param precomputed_indices: (B, npoint, nsample) - 미리 계산된 Ball Query 인덱스
    """
    B, C, N = xyz.shape
    xyz_t = xyz.permute(0, 2, 1) # (B, N, 3)
    
    # 1. Sampling Layer
    if precomputed_centroids is not None:
        # FPS를 생략하고 입력받은 중심점을 그대로 사용
        new_xyz_trans = precomputed_centroids
        new_xyz = new_xyz_trans.permute(0, 2, 1) # (B, npoint, 3)
    else:
        # 기존 로직 (FPS 연산 수행)
        fps_idx = farthest_point_sample(xyz_t, npoint) # (B, npoint)
        new_xyz = index_points(xyz_t, fps_idx) # (B, npoint, 3)
        new_xyz_trans = new_xyz.permute(0, 2, 1) # (B, 3, npoint)

    # 2. Grouping Layer: Ball Query
    if precomputed_indices is not None:
        # 미리 계산된 인덱스 사용
        idx = precomputed_indices
    else:
        idx = query_ball_point(radius, nsample, xyz_t, new_xyz)
    
    # 3. Grouping
    grouped_xyz = index_points(xyz_t, idx) 
    grouped_xyz = grouped_xyz.permute(0, 3, 1, 2)
    grouped_xyz = grouped_xyz - new_xyz_trans.unsqueeze(-1)
    
    if points is not None:
        points_t = points.permute(0, 2, 1)
        grouped_points = index_points(points_t, idx)
        grouped_points = grouped_points.permute(0, 3, 1, 2)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=1)
    else:
        new_points = grouped_xyz

    return new_xyz_trans, grouped_xyz, new_points