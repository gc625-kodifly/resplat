import torch
from torch import Tensor


def local_knn_query(
    k: int,
    points_3d: Tensor,          # [N, 3] flattened (v*h*w)
    extrinsics: Tensor,          # [V, 4, 4] camera-to-world
    intrinsics: Tensor,          # [V, 3, 3] normalized (0-1)
    v: int, h: int, w: int,
    spatial_radius: int = 3,
    num_neighbor_views: int = 4,
    cross_view_radius: int = 3,
) -> Tensor:
    """Local KNN for structured point clouds from multi-view depth maps.

    Builds a candidate set from:
    1. Per-view 2D spatial neighbors (grid offsets within radius r)
    2. Cross-view neighbors (project to nearest cameras, sample window)

    Then selects top-k nearest from the combined candidate set.

    Returns [N, k] int32 tensor, same interface as pointops.knn_query.

    This is the batched implementation — cross-view projection is fully
    vectorized over all source views and neighbor views simultaneously.
    """
    N = points_3d.shape[0]
    assert N == v * h * w
    device = points_3d.device

    points_grid = points_3d.reshape(v, h, w, 3)

    # --- 1. Camera neighbor list ---
    cam_centers = extrinsics[:, :3, 3]  # [V, 3]
    cam_dist = torch.cdist(cam_centers.unsqueeze(0), cam_centers.unsqueeze(0), p=2)[0]  # [V, V]
    cam_dist_sorted = torch.argsort(cam_dist, dim=1)  # [V, V]
    # Exclude self (index 0 is self), take num_neighbor_views
    nv = min(num_neighbor_views, v - 1)
    neighbor_views = cam_dist_sorted[:, 1:nv + 1]  # [V, nv]

    # --- 2. Spatial candidate indices [N, S] ---
    sr = spatial_radius
    offsets_r = torch.arange(-sr, sr + 1, device=device)
    offsets_c = torch.arange(-sr, sr + 1, device=device)
    dr, dc = torch.meshgrid(offsets_r, offsets_c, indexing="ij")
    dr = dr.reshape(-1)
    dc = dc.reshape(-1)
    # Remove center (0, 0)
    center_mask = (dr != 0) | (dc != 0)
    dr = dr[center_mask]
    dc = dc[center_mask]
    S = dr.shape[0]  # (2r+1)^2 - 1

    # Point grid indices: [V, H, W] for row and col
    view_idx = torch.arange(v, device=device).view(v, 1, 1).expand(v, h, w)
    row_idx = torch.arange(h, device=device).view(1, h, 1).expand(v, h, w)
    col_idx = torch.arange(w, device=device).view(1, 1, w).expand(v, h, w)

    # Flatten to [N]
    view_flat = view_idx.reshape(N)
    row_flat = row_idx.reshape(N)
    col_flat = col_idx.reshape(N)

    # Neighbor rows/cols: [N, S]
    nb_row = row_flat.unsqueeze(1) + dr.unsqueeze(0)  # [N, S]
    nb_col = col_flat.unsqueeze(1) + dc.unsqueeze(0)  # [N, S]

    # Clamp and track validity
    spatial_valid = (nb_row >= 0) & (nb_row < h) & (nb_col >= 0) & (nb_col < w)
    nb_row = nb_row.clamp(0, h - 1)
    nb_col = nb_col.clamp(0, w - 1)

    # Flat indices: view * h * w + row * w + col
    spatial_indices = view_flat.unsqueeze(1) * (h * w) + nb_row * w + nb_col  # [N, S]

    # --- 3. Cross-view candidate indices [N, nv * C] (batched) ---
    cr = cross_view_radius
    cv_offsets_r = torch.arange(-cr, cr + 1, device=device)
    cv_offsets_c = torch.arange(-cr, cr + 1, device=device)
    cv_dr, cv_dc = torch.meshgrid(cv_offsets_r, cv_offsets_c, indexing="ij")
    cv_dr = cv_dr.reshape(-1)  # [C] where C = (2*cr+1)^2
    cv_dc = cv_dc.reshape(-1)
    C = cv_dr.shape[0]

    # Precompute world-to-camera transforms
    w2c = torch.linalg.inv(extrinsics)  # [V, 4, 4]

    # Homogeneous points per view: [V, HW, 4]
    hw = h * w
    ones = torch.ones(v, hw, 1, device=device, dtype=points_3d.dtype)
    pts_h = torch.cat([points_grid.reshape(v, hw, 3), ones], dim=-1)  # [V, HW, 4]

    # Gather target transforms for all (source_view, neighbor) pairs
    w2c_tgt = w2c[neighbor_views]          # [V, nv, 4, 4]
    intr_tgt = intrinsics[neighbor_views]  # [V, nv, 3, 3]

    # Project all source points to all target cameras simultaneously
    # [V, 1, HW, 4] @ [V, nv, 4, 4] -> [V, nv, HW, 4]
    cam_pts = pts_h.unsqueeze(1) @ w2c_tgt.transpose(-1, -2)
    cam_xyz = cam_pts[..., :3]             # [V, nv, HW, 3]
    valid_z = cam_xyz[..., 2] > 0          # [V, nv, HW]

    # Perspective projection + intrinsics
    uv = cam_xyz[..., :2] / cam_xyz[..., 2:3].clamp(min=1e-6)  # [V, nv, HW, 2]
    uv_ones = torch.ones(*uv.shape[:-1], 1, device=device, dtype=uv.dtype)
    uv_h = torch.cat([uv, uv_ones], dim=-1)          # [V, nv, HW, 3]
    proj = uv_h @ intr_tgt.transpose(-1, -2)          # [V, nv, HW, 3]

    # Pixel coords + window offsets
    proj_col = (proj[..., 0] * w - 0.5).round().long()  # [V, nv, HW]
    proj_row = (proj[..., 1] * h - 0.5).round().long()

    nb_r = proj_row.unsqueeze(-1) + cv_dr  # [V, nv, HW, C]
    nb_c = proj_col.unsqueeze(-1) + cv_dc

    cross_valid = ((nb_r >= 0) & (nb_r < h) & (nb_c >= 0) & (nb_c < w)
                   & valid_z.unsqueeze(-1))
    nb_r.clamp_(0, h - 1)
    nb_c.clamp_(0, w - 1)

    tgt_view_idx = neighbor_views.reshape(v, nv, 1, 1)  # [V, nv, 1, 1]
    flat_idx = tgt_view_idx * hw + nb_r * w + nb_c       # [V, nv, HW, C]

    # Rearrange: [V, nv, HW, C] -> [V, HW, nv, C] -> [N, nv*C]
    cross_indices = flat_idx.permute(0, 2, 1, 3).reshape(N, nv * C)
    cross_valid = cross_valid.permute(0, 2, 1, 3).reshape(N, nv * C)

    # --- 4. Combine candidates and select top-k ---
    all_indices = torch.cat([spatial_indices, cross_indices], dim=1)  # [N, S + nv*C]
    all_valid = torch.cat([spatial_valid, cross_valid], dim=1)  # [N, S + nv*C]
    total_candidates = all_indices.shape[1]

    # Gather candidate 3D positions
    # Clamp indices for safe gather (invalid ones will be masked anyway)
    gather_idx = all_indices.clamp(0, N - 1)  # [N, total_candidates]
    candidate_pos = points_3d[gather_idx]  # [N, total_candidates, 3]

    # Squared distances
    query_pos = points_3d.unsqueeze(1)  # [N, 1, 3]
    sq_dist = ((candidate_pos - query_pos) ** 2).sum(dim=-1)  # [N, total_candidates]

    # Mask invalid candidates
    sq_dist[~all_valid] = float("inf")

    # Select top-k
    if total_candidates <= k:
        # Not enough candidates, take all and pad with self
        topk_dist, topk_local = sq_dist.topk(min(k, total_candidates), largest=False)
        result = gather_idx.gather(1, topk_local)
        if total_candidates < k:
            self_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, k - total_candidates)
            result = torch.cat([result, self_idx], dim=1)
    else:
        topk_dist, topk_local = sq_dist.topk(k, largest=False)
        result = gather_idx.gather(1, topk_local)

    # Replace any inf-distance slots with self-index
    inf_mask = topk_dist == float("inf")
    if inf_mask.any():
        self_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(result)
        result[inf_mask] = self_idx[inf_mask]

    return result.int()

