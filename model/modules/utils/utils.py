import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ────────────────────────────────────────────────
#  Triton FPS kernel
#  B programs (one per batch element).
#  All npoint iterations run on-GPU; no Python
#  round-trips between centroid selections.
# ────────────────────────────────────────────────

if _TRITON_AVAILABLE:
    @triton.jit
    def _fps_kernel(
        xyz_ptr,               # (B, N, 3) float32, C-contiguous
        out_ptr,               # (B, npoint) int32
        dist_ptr,              # (B, N) float32 scratch
        start_ptr,             # (B,) int32 – random start indices
        N,                     # runtime: number of input points
        npoint,                # runtime: number of output centroids
        BLOCK_N: tl.constexpr, # vectorisation width (power of 2)
    ):
        b    = tl.program_id(0)
        xyz_b  = xyz_ptr  + b * N * 3
        out_b  = out_ptr  + b * npoint
        dist_b = dist_ptr + b * N

        offs  = tl.arange(0, BLOCK_N)
        n_blk = tl.cdiv(N, BLOCK_N)

        # Initialise distances to +∞
        for bn in range(n_blk):
            off  = bn * BLOCK_N + offs
            mask = off < N
            tl.store(dist_b + off,
                     tl.full([BLOCK_N], 1e10, dtype=tl.float32),
                     mask=mask)

        farthest = tl.load(start_ptr + b)

        for i in range(npoint):
            tl.store(out_b + i, farthest)

            # Load centroid coordinates (scalar loads)
            cx = tl.load(xyz_b + farthest * 3 + 0).to(tl.float32)
            cy = tl.load(xyz_b + farthest * 3 + 1).to(tl.float32)
            cz = tl.load(xyz_b + farthest * 3 + 2).to(tl.float32)

            best_dist = -1.0
            best_idx  = 0

            for bn in range(n_blk):
                off  = bn * BLOCK_N + offs
                mask = off < N

                base = off * 3
                px = tl.load(xyz_b + base,     mask=mask, other=cx)
                py = tl.load(xyz_b + base + 1, mask=mask, other=cy)
                pz = tl.load(xyz_b + base + 2, mask=mask, other=cz)

                dx = px - cx;  dy = py - cy;  dz = pz - cz
                d2 = dx * dx + dy * dy + dz * dz

                # Update minimum distance (fused minimum)
                cur = tl.load(dist_b + off, mask=mask, other=1e10)
                upd = tl.minimum(cur, d2)
                tl.store(dist_b + off, upd, mask=mask)

                # Track global argmax across blocks
                upd_m   = tl.where(mask, upd, -1.0)
                blk_max = tl.max(upd_m, axis=0)

                # Uniform conditional: blk_max is a scalar reduction result
                if blk_max > best_dist:
                    best_dist = blk_max
                    # First index in this block where upd == blk_max
                    cand     = tl.where(upd_m >= blk_max, offs, N)
                    best_idx = bn * BLOCK_N + tl.min(cand, axis=0)

            farthest = best_idx

    def _fps_triton(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        B, N, _ = xyz.shape
        xyz_c = xyz.contiguous()
        out   = torch.empty(B, npoint, dtype=torch.int32, device=xyz.device)
        dist  = torch.empty(B, N,      dtype=torch.float32, device=xyz.device)
        start = torch.randint(0, N, (B,), dtype=torch.int32, device=xyz.device)

        BLOCK_N = 512
        _fps_kernel[(B,)](xyz_c, out, dist, start, N, npoint, BLOCK_N=BLOCK_N)
        return out.long()


def _fps_pytorch(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Optimised pure-PyTorch fallback (CPU or non-float32 input)."""
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance  = torch.full((B, N), 1e10, device=device)
    farthest  = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest].unsqueeze(1)           # (B,1,3)
        dist     = torch.sum((xyz - centroid) ** 2, dim=-1)
        torch.minimum(distance, dist, out=distance)
        farthest = distance.argmax(dim=-1)

    return centroids


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling.

    GPU path  – Triton kernel (single launch, all iterations on-GPU).
    CPU path  – optimised PyTorch loop.

    xyz    : (B, N, 3) float32
    npoint : number of centroids to sample
    returns: (B, npoint) long
    """
    if _TRITON_AVAILABLE and xyz.is_cuda and xyz.dtype == torch.float32:
        return _fps_triton(xyz, npoint)
    return _fps_pytorch(xyz, npoint)


# ─────────────────────────────────────────────────
#  Remaining utilities (unchanged)
# ─────────────────────────────────────────────────

def square_distance(src, dst):
    """
    src: (B, N, C)
    dst: (B, M, C)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape   = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape   = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (torch.arange(B, dtype=torch.long, device=device)
                     .view(view_shape).repeat(repeat_shape))
    return points[batch_indices, idx, :]


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = square_distance(new_xyz, xyz)

    group_idx = (torch.arange(N, dtype=torch.long, device=device)
                 .view(1, 1, N).repeat(B, S, 1))
    group_idx[sqrdists > radius ** 2] = N

    group_idx  = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    xyz    : (B, 3, N)
    points : (B, C, N)
    """
    B, C, N = xyz.shape

    xyz_t   = xyz.permute(0, 2, 1)                        # (B, N, 3)
    fps_idx = farthest_point_sample(xyz_t, npoint)        # (B, npoint)

    new_xyz       = index_points(xyz_t, fps_idx)          # (B, npoint, 3)
    new_xyz_trans = new_xyz.permute(0, 2, 1)              # (B, 3, npoint)

    idx         = query_ball_point(radius, nsample, xyz_t, new_xyz)
    grouped_xyz = index_points(xyz_t, idx).permute(0, 3, 1, 2)  # (B,3,npoint,nsample)

    grouped_points = grouped_xyz
    if points is not None:
        points_t        = points.permute(0, 2, 1)
        grouped_features = index_points(points_t, idx).permute(0, 3, 1, 2)
        grouped_points  = grouped_features

    return new_xyz_trans, grouped_xyz, grouped_points
