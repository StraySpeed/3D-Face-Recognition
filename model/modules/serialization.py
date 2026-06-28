import numpy as np
import torch


def _part1by2_vec(arr):
    """10-bit 정수 배열을 Morton 인터리브 형태로 비트 분산."""
    n = arr.astype(np.uint32) & 0x000003ff
    n = (n ^ (n << 16)) & 0xff0000ff
    n = (n ^ (n <<  8)) & 0x0300f00f
    n = (n ^ (n <<  4)) & 0x030c30c3
    n = (n ^ (n <<  2)) & 0x09249249
    return n


def morton_codes(points_np, bits=10):
    """
    (N, 3) numpy 점들에 대한 Morton(Z-order) 코드 계산.
    points_np는 미리 normalize되어 있다고 가정하지 않음 — 내부에서 min-max 정규화.
    """
    p_min = points_np.min(axis=0)
    p_max = points_np.max(axis=0)
    norm = (points_np - p_min) / (p_max - p_min + 1e-8)
    res = (1 << bits)
    q = np.clip(np.floor(norm * res), 0, res - 1).astype(np.uint32)
    return (_part1by2_vec(q[:, 2]) << 2) | (_part1by2_vec(q[:, 1]) << 1) | _part1by2_vec(q[:, 0])


def morton_sort_tensor(curr_xyz):
    """
    (3, N) 텐서를 Z-order로 정렬해서 반환.
    클라이언트 사전 처리 단계에서 호출되며, 서버는 정렬된 시퀀스만 받음.
    """
    points_np = curr_xyz.detach().cpu().numpy().T  # (N, 3)
    codes = morton_codes(points_np)
    sorted_np = points_np[np.argsort(codes)]
    return torch.tensor(sorted_np, dtype=curr_xyz.dtype, device=curr_xyz.device).t().contiguous()


def morton_sort_np(points_np):
    """numpy 인풋(N, 3) → Z-order 정렬된 (N, 3) numpy."""
    codes = morton_codes(points_np)
    return points_np[np.argsort(codes)]
