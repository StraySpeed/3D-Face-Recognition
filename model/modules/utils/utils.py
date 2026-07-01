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

    # ────────────────────────────────────────────────
    #  Triton Ball Query kernel
    #  One program per (batch, centroid) pair.
    #  Tiles N input points in BLOCK_N chunks —
    #  no (B, S, N) intermediate tensor allocated.
    # ────────────────────────────────────────────────
    @triton.jit
    def _ball_query_kernel(
        xyz_ptr,               # (B, N, 3) float32 contiguous
        ctr_ptr,               # (B*S, 3)  float32 contiguous
        out_ptr,               # (B*S*nsample,) int32 — pre-filled with N (sentinel)
        radius_sq,             # float32 = radius²
        N,                     # runtime: input points per batch element
        S,                     # runtime: centroids per batch element
        nsample: tl.constexpr, # compile-time: neighbours to collect (e.g. 32)
        BLOCK_N: tl.constexpr, # compile-time: tile width
    ):
        bs    = tl.program_id(0)       # ∈ [0, B*S)
        b     = bs // S
        xyz_b  = xyz_ptr + b  * N * 3  # base of batch b's points
        ctr_bs = ctr_ptr + bs * 3      # this centroid's xyz
        out_bs = out_ptr + bs * nsample

        cx = tl.load(ctr_bs + 0)
        cy = tl.load(ctr_bs + 1)
        cz = tl.load(ctr_bs + 2)

        n_blk = tl.cdiv(N, BLOCK_N)

        # count: 0-d int32 accumulator — how many valid neighbours stored so far
        count = tl.sum(tl.zeros([1], dtype=tl.int32), axis=0)

        for bn in range(n_blk):
            offs = bn * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = offs < N

            px = tl.load(xyz_b + offs * 3 + 0, mask=mask, other=cx)
            py = tl.load(xyz_b + offs * 3 + 1, mask=mask, other=cy)
            pz = tl.load(xyz_b + offs * 3 + 2, mask=mask, other=cz)

            dx = px - cx; dy = py - cy; dz = pz - cz
            d2 = dx*dx + dy*dy + dz*dz

            valid = (d2 <= radius_sq) & mask          # (BLOCK_N,) bool

            # 0-indexed rank of each valid element within this tile
            loc  = tl.cumsum(valid.to(tl.int32), axis=0) - 1  # (BLOCK_N,) int32
            gpos = count + loc                                  # global write pos

            # Clamp address for inactive lanes (negative gpos) to avoid UB
            gpos_safe = tl.where(gpos >= 0, gpos, 0)

            write = valid & (gpos >= 0) & (gpos < nsample)
            tl.store(out_bs + gpos_safe, offs.to(tl.int32), mask=write)

            count = count + tl.sum(valid.to(tl.int32), axis=0)

    def _ball_query_triton(radius: float, nsample: int,
                           xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz     : (B, N, 3) float32 CUDA
        new_xyz : (B, S, 3) float32 CUDA
        returns : (B, S, nsample) long
        """
        B, N, _ = xyz.shape
        _, S, _ = new_xyz.shape

        xyz_c = xyz.contiguous()
        ctr_c = new_xyz.contiguous().view(B * S, 3)

        # Pre-fill with N (sentinel = "no valid neighbour")
        out = torch.full((B * S * nsample,), N, dtype=torch.int32, device=xyz.device)

        BLOCK_N = 256
        _ball_query_kernel[(B * S,)](
            xyz_c, ctr_c, out,
            float(radius ** 2),
            N, S,
            nsample=nsample,
            BLOCK_N=BLOCK_N,
        )

        out = out.view(B, S, nsample).long()

        # Replace sentinel N with first valid entry; clamp for the degenerate
        # case where no points fall within radius (original has the same behaviour)
        first_safe = out[:, :, 0].clamp(max=N - 1).unsqueeze(-1).expand_as(out)
        out = torch.where(out >= N, first_safe, out)
        return out


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
    # Triton path: no (B, S, N) intermediate tensor
    if _TRITON_AVAILABLE and xyz.is_cuda and xyz.dtype == torch.float32:
        return _ball_query_triton(radius, nsample, xyz, new_xyz)

    # PyTorch fallback: chunked + topk to limit peak memory
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    r2 = radius ** 2

    CHUNK = 256
    out = torch.empty(B, S, nsample, dtype=torch.long, device=device)

    for s0 in range(0, S, CHUNK):
        s1   = min(s0 + CHUNK, S)
        dists = square_distance(new_xyz[:, s0:s1], xyz)         # (B, chunk, N)
        idx   = torch.arange(N, device=device).view(1, 1, N).expand(B, s1-s0, N).clone()
        idx[dists > r2] = N
        idx = idx.topk(nsample, dim=-1, largest=False)[0]       # (B, chunk, nsample)
        first = idx[:, :, 0:1].expand_as(idx)
        idx[idx == N] = first[idx == N]
        out[:, s0:s1] = idx

    return out


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
