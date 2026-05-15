import torch
import os
import numpy as np
import torch.nn as nn
from time import time
import torch.nn.functional as F
from torch_cluster import grid_cluster,knn
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_add

def offset2batch(offset):

    batch = torch.zeros(offset[-1].item(), dtype=torch.long, device=offset.device)
    for i in range(len(offset) - 1):
        batch[offset[i]:offset[i+1]] = i
    return batch

def batch2offset(batch):

    return torch.cumsum(batch.bincount(), dim=0)

def voxel_grid(pos, size, batch, start=0):

    if not torch.is_tensor(size):
        size = torch.tensor([size, size, size], device=pos.device)
    
    pos = pos - start
    grid_index = torch.floor(pos / size).long()

    max_grid = grid_index.max(dim=0)[0] + 1
    grid_index_1d = grid_index[:, 0] * (max_grid[1] * max_grid[2]) + \
                    grid_index[:, 1] * max_grid[2] + \
                    grid_index[:, 2]
    
    batch_offset = batch * (max_grid[0] * max_grid[1] * max_grid[2]).item()
    return grid_index_1d + batch_offset

def segment_csr(src, indptr, reduce="mean"):
    if reduce == "mean":
        return scatter_mean(src, torch.arange(len(indptr)-1, device=src.device).repeat_interleave(indptr[1:] - indptr[:-1]), dim=0)
    elif reduce == "max":
        return scatter_max(src, torch.arange(len(indptr)-1, device=src.device).repeat_interleave(indptr[1:] - indptr[:-1]), dim=0)[0]
    elif reduce == "min":
        return scatter_min(src, torch.arange(len(indptr)-1, device=src.device).repeat_interleave(indptr[1:] - indptr[:-1]), dim=0)[0]
    else:
        raise ValueError(f"Unsupported reduce type: {reduce}")

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
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


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
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


# def query_ball_point(radius, nsample, xyz, new_xyz):
#     """
#     Input:
#         radius: local region radius 
#         nsample: 
#         xyz: all points, [B, N, 3]
#         new_xyz: query points, [B, S, 3]
#     Return:
#         group_idx: grouped points index, [B, S, nsample]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     _, S, _ = new_xyz.shape
    
#     group_idx = torch.full((B, S, nsample), N, dtype=torch.long, device=device)
#     chunk_size = 256
    
#     for si in range(0, S, chunk_size):
#         end_si = min(si + chunk_size, S)

#         sqrdists = square_distance(new_xyz[:, si:end_si, :], xyz)
        
#         _, idx = torch.topk(sqrdists, k=nsample, dim=-1, largest=False, sorted=False)
        
#         group_idx[:, si:end_si, :] = idx
    
#     return group_idx

# def query_ball_point_knn_aligned(radius, nsample, xyz, new_xyz):
#     B, N, _ = xyz.shape
#     _, S, _ = new_xyz.shape
#     device = xyz.device
    
#     batch_xyz = torch.arange(B, device=device).repeat_interleave(N)
#     batch_new = torch.arange(B, device=device).repeat_interleave(S)
    
#     idx = knn(xyz.reshape(-1, 3), new_xyz.reshape(-1, 3), 
#               k=nsample, batch_x=batch_xyz, batch_y=batch_new)[1]
#     if idx.shape[0] == B * S * nsample:
#         idx = idx.view(B, S, nsample)  # [B, S, K]
#     else:
#         return None
    
#     # perm = torch.stack([torch.randperm(nsample, device=device) for _ in range(B * S)])
#     # idx = idx.view(B * S, nsample)[torch.arange(B * S, device=device).unsqueeze(1), perm]
    
#     return idx.view(B, S, nsample)
def query_ball_point_knn_aligned(radius, nsample, xyz, new_xyz, min_k=1):
    """
    radius-aware KNN:
    1) 先用 KNN 找候选邻居
    2) 用 radius 过滤过远邻居
    3) 若有效邻居不足，则用最近邻补齐，保证输出 shape 恒定 [B,S,nsample]
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device

    if N <= 0 or S <= 0:
        return None,None

    k_safe = min(nsample, N)
    min_k = min(min_k, k_safe)

    # knn: x=database(xyz), y=query(new_xyz), 返回 query->database 的邻接
    batch_x = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(N)
    batch_y = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(S)

    knn_col = knn(
        x=xyz.reshape(-1, 3),
        y=new_xyz.reshape(-1, 3),
        k=k_safe,
        batch_x=batch_x,
        batch_y=batch_y
    )[1]  # [B*S*k_safe]，为 database 索引（batch 内局部索引）

    if knn_col.numel() != B * S * k_safe:
        return None, None

    idx = knn_col.view(B, S, k_safe)  # [B,S,K]

    # 计算每个 query 到候选邻居的距离平方
    grouped_xyz = index_points(xyz, idx)                      # [B,S,K,3]
    center_xyz = new_xyz.unsqueeze(2)                         # [B,S,1,3]
    d2 = torch.sum((grouped_xyz - center_xyz) ** 2, dim=-1)  # [B,S,K]
    r2 = float(radius) * float(radius)

    valid = d2 <= r2  # [B,S,K]

    # 至少保留 min_k 个最近邻，避免稀疏区域全被 mask
    # d2 已与 idx 对齐（KNN 本身一般按近到远）
    force_keep = torch.zeros_like(valid)
    force_keep[:, :, :min_k] = True
    keep = valid | force_keep

    # 为了按“有效优先、无效靠后”重排：有效项key=0，无效项key=1
    # 再叠加小距离项，确保有效里仍按距离近优先
    sort_key = (~keep).float() * 1e6 + d2
    order = torch.argsort(sort_key, dim=-1)                   # [B,S,K]
    idx_sorted = torch.gather(idx, 2, order)                  # [B,S,K]
    keep_sorted = torch.gather(keep, 2, order)                # [B,S,K]

    if k_safe < nsample:
        # N 比 nsample 还小，补齐到 nsample
        pad = nsample - k_safe
        pad_idx = idx_sorted[:, :, :1].expand(B, S, pad)      # 最近邻重复
        pad_keep = keep_sorted[:, :, :1].expand(B, S, pad)
        idx_sorted = torch.cat([idx_sorted, pad_idx], dim=2)  # [B,S,nsample]
        keep_sorted = torch.cat([keep_sorted, pad_keep], dim=2)
    elif k_safe > nsample:
        idx_sorted = idx_sorted[:, :, :nsample]
        keep_sorted = keep_sorted[:, :, :nsample]

    # 如果某些 query 在 radius 内几乎没有点，也至少保留第一个邻居（已由 min_k 保证）
    # 这里再保险：若全False，强制第一个True
    all_invalid = ~keep_sorted.any(dim=-1, keepdim=True)      # [B,S,1]
    if all_invalid.any():
        keep_sorted[:, :, 0:1] = keep_sorted[:, :, 0:1] | all_invalid

    # 注意：下游当前只用 idx，不用 mask；若你后续想做严格密度加权，可返回 keep_sorted
    return idx_sorted, keep_sorted

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


# class PointNetSetAbstractionMsg(nn.Module):
#     def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
#         super(PointNetSetAbstractionMsg, self).__init__()
#         self.npoint = npoint
#         self.radius_list = radius_list
#         self.nsample_list = nsample_list
#         self.conv_blocks = nn.ModuleList()
#         self.bn_blocks = nn.ModuleList()
#         for i in range(len(mlp_list)):
#             convs = nn.ModuleList()
#             bns = nn.ModuleList()
#             last_channel = in_channel + 3
#             for out_channel in mlp_list[i]:
#                 convs.append(nn.Conv2d(last_channel, out_channel, 1))
#                 bns.append(nn.BatchNorm2d(out_channel))
#                 last_channel = out_channel
#             self.conv_blocks.append(convs)
#             self.bn_blocks.append(bns)

#     def forward(self, xyz, points,self_define_samples=None):
#         """
#         Input:
#             xyz: input points position data, [B, C, N]
#             points: input points data, [B, D, N]
#         Return:
#             new_xyz: sampled points position data, [B, C, S]
#             new_points_concat: sample points feature data, [B, D', S]
#         """
#         xyz = xyz.permute(0, 2, 1)
#         if points is not None:
#             points = points.permute(0, 2, 1)

#         B, N, C = xyz.shape
#         S = self_define_samples if self_define_samples else self.npoint
#         new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
#         new_points_list = []
#         for i, radius in enumerate(self.radius_list):
#             K = self.nsample_list[i]
#             group_idx = query_ball_point(radius, K, xyz, new_xyz)
#             # group_idx = pointnet2_utils.query_ball_point(radius, K, xyz, new_xyz)
#             grouped_xyz = index_points(xyz, group_idx)
#             grouped_xyz -= new_xyz.view(B, S, 1, C)
#             if points is not None:
#                 grouped_points = index_points(points, group_idx)
#                 grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
#             else:
#                 grouped_points = grouped_xyz

#             grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
#             for j in range(len(self.conv_blocks[i])):
#                 conv = self.conv_blocks[i][j]
#                 bn = self.bn_blocks[i][j]
#                 grouped_points =  F.relu(bn(conv(grouped_points)))
#             new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
#             new_points_list.append(new_points)

#         new_xyz = new_xyz.permute(0, 2, 1)
#         new_points_concat = torch.cat(new_points_list, dim=1)
#         return new_xyz, new_points_concat

class PointNetSetAbstractionMsg_GridPool(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, grid_size=None):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.grid_size = grid_size

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                # bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
    def _global_fps_indices(self, xyz, npoint):
        """
        xyz: [B,N,3]
        return: [B,npoint] (unique within each batch)
        """
        B, N, _ = xyz.shape
        device = xyz.device
        npoint = min(int(npoint), N)
        out = []
        for b in range(B):
            pts = xyz[b:b+1]                          # [1,N,3]
            idx = farthest_point_sample(pts, npoint) # [1,npoint]
            out.append(idx[0])
        return torch.stack(out, dim=0)   
    # ############## sample min index point in every voxel############
    # def grid_pool_downsample(self, xyz, target_npoint):
    #     """
    #     Args:
    #         xyz: [B, N, 3] - GPU tensor
    #         target_npoint: int 
    #     Returns:
    #         indices: [B, target_npoint]
    #     """
    #     B, N, _ = xyz.shape
    #     device = xyz.device
        

    #     if self.grid_size is None:
    #         bbox = xyz.max(dim=1)[0] - xyz.min(dim=1)[0]  # [B, 3]
    #         vol = bbox.prod(dim=1).mean()

    #         target_npoint = torch.Tensor(target_npoint).to('cuda')
    #         init_gs = max(1e-4, (vol / target_npoint) ** (1/3) * 0.15)
    #         self.grid_size = float(init_gs) 
        

    #     min_gs = self.grid_size * 0.01   
    #     max_gs = self.grid_size * 1000.0 
        

    #     best_cluster = None
    #     best_gs = self.grid_size
    #     best_num = 0
        
    #     for _ in range(20):  
    #         gs_val = float(best_gs) 
    #         grid_size_tensor = torch.full((3,), gs_val, device=device, dtype=xyz.dtype)
            
    #         cluster = grid_cluster(xyz.reshape(-1, 3), grid_size_tensor)  # [B*N]
    #         unique_clusters, inverse = torch.unique(cluster, return_inverse=True)
    #         num_clusters = unique_clusters.shape[0]
            
            
    #         if abs(num_clusters - target_npoint) < abs(best_num - target_npoint):
    #             best_num = num_clusters
    #             best_gs = gs_val
    #             best_cluster = (cluster, unique_clusters, inverse)
            
    
    #         if num_clusters < target_npoint * 0.995:
    #             max_gs = gs_val  
    #             best_gs = (min_gs + gs_val) / 2
    #         elif num_clusters > target_npoint * 1.005:
    #             min_gs = gs_val  
    #             best_gs = (gs_val + max_gs) / 2
    #         else:
    #             break  
        
    #     if best_cluster == None:
    #         print("Can't find a satisfied grid_size for grid_pool_sample")
    #         return None
    #     cluster, unique_clusters, inverse = best_cluster
    #     num_clusters = unique_clusters.shape[0]
        

    #     min_indices = torch.full_like(unique_clusters, 10**9, device=device)
    #     point_indices = torch.arange(B * N, device=device)
    #     min_indices.scatter_reduce_(0, inverse, point_indices, reduce='min')
        
    #     representative_indices = min_indices % N
    #     batch_of_reps = torch.arange(B, device=device).repeat_interleave(N)[min_indices]
        

    #     final_indices = []
    #     for b in range(B):
    #         mask = batch_of_reps == b
    #         indices_b = representative_indices[mask]
    #         current_n = indices_b.shape[0]
            
    #         if current_n > target_npoint:

    #             perm = torch.randperm(current_n, device=device)
    #             indices_b = indices_b[perm][:target_npoint]
    #         elif current_n < target_npoint:
    #             deficit = target_npoint - current_n

    #             if deficit <= 0.3 * target_npoint:
    #                 rand_indices = torch.randint(0, N, (deficit,), device=device)
    #                 indices_b = torch.cat([indices_b, rand_indices], dim=0)
    #             else:
    #                 print(f"[WARN] Batch {b}: Only {current_n} clusters. "
    #                     f"Consider reducing target_npoint or checking data distribution.")

    #                 repeat_needed = deficit // current_n + 1
    #                 extra = representative_indices[mask].repeat(repeat_needed)[:deficit]
    #                 indices_b = torch.cat([indices_b, extra], dim=0)
            
    #         final_indices.append(indices_b)
        
    #     return torch.stack(final_indices)  # [B, target_npoint]
            
    # ############## sample min distance point from centroid in every voxel############
    # def grid_pool_downsample(self, xyz, target_npoint):
    #     """
    #     Args:
    #         xyz: [B, N, 3] - GPU tensor
    #         target_npoint: int 
    #     Returns:
    #         indices: [B, target_npoint]
    #     """
    #     B, N, _ = xyz.shape
    #     device = xyz.device
        

    #     if self.grid_size is None:
    #         bbox = xyz.max(dim=1)[0] - xyz.min(dim=1)[0]  # [B, 3]
    #         vol = bbox.prod(dim=1).mean()

    #         target_npoint = torch.Tensor(target_npoint).to('cuda')
    #         init_gs = max(1e-4, (vol / target_npoint) ** (1/3) * 0.15)
    #         self.grid_size = float(init_gs) 
        

    #     min_gs = self.grid_size * 0.01   
    #     max_gs = self.grid_size * 1000.0 
        

    #     best_cluster = None
    #     best_gs = self.grid_size
    #     best_num = 0
        
    #     for _ in range(20):  
    #         gs_val = float(best_gs) 
    #         grid_size_tensor = torch.full((3,), gs_val, device=device, dtype=xyz.dtype)
            
    #         cluster = grid_cluster(xyz.reshape(-1, 3), grid_size_tensor)  # [B*N]
    #         unique_clusters, inverse = torch.unique(cluster, return_inverse=True)
    #         num_clusters = unique_clusters.shape[0]
            
            
    #         if abs(num_clusters - target_npoint) < abs(best_num - target_npoint):
    #             best_num = num_clusters
    #             best_gs = gs_val
    #             best_cluster = (cluster, unique_clusters, inverse)
            
    
    #         if num_clusters < target_npoint * 0.995:
    #             max_gs = gs_val  
    #             best_gs = (min_gs + gs_val) / 2
    #         elif num_clusters > target_npoint * 1.005:
    #             min_gs = gs_val  
    #             best_gs = (gs_val + max_gs) / 2
    #         else:
    #             break  
        
    #     if best_cluster == None:
    #         print("Can't find a satisfied grid_size for grid_pool_sample")
    #         return None
    #     cluster, unique_clusters, inverse = best_cluster
    #     num_clusters = unique_clusters.shape[0]
        

    #     centroids = scatter_mean(xyz.reshape(-1, 3), inverse, dim=0)
    #     point_to_center = xyz.reshape(-1, 3) - centroids[inverse]
    #     distances = torch.norm(point_to_center, dim=1)
    #     min_dist_indices = scatter_min(distances, inverse, dim=0)[1]

    #     representative_indices = min_dist_indices % N
    #     batch_of_reps = min_dist_indices // N
        

    #     final_indices = []
    #     for b in range(B):
    #         mask = batch_of_reps == b
    #         indices_b = representative_indices[mask]
    #         current_n = indices_b.shape[0]
            
    #         if current_n > target_npoint:

    #             perm = torch.randperm(current_n, device=device)
    #             indices_b = indices_b[perm][:target_npoint]
    #         elif current_n < target_npoint:
    #             deficit = target_npoint - current_n

    #             if deficit <= 0.005 * target_npoint:
    #                 rand_indices = torch.randint(0, N, (deficit,), device=device)
    #                 indices_b = torch.cat([indices_b, rand_indices], dim=0)
    #             else:
    #                 print(f"[WARN] Batch {b}: Only {current_n} clusters. "
    #                     f"Consider reducing target_npoint or checking data distribution.")

    #                 repeat_needed = deficit // current_n + 1
    #                 extra = representative_indices[mask].repeat(repeat_needed)[:deficit]
    #                 indices_b = torch.cat([indices_b, extra], dim=0)
            
    #         final_indices.append(indices_b)
    #     result = {'sampled_index': torch.stack(final_indices)}
    #     return result  # [B, target_npoint]
            
    # ############## sample centroid in every voxel as the new xyz############
    # 放在 PointNetSetAbstractionMsg_GridPool 类内（forward 前）
    def _compute_cover_from_group(self, group_idx, group_mask, N):
        # group_idx/group_mask: [B,S,K]
        B = group_idx.shape[0]
        covers = []
        for b in range(B):
            idx_b = group_idx[b].reshape(-1)
            msk_b = group_mask[b].reshape(-1)
            valid_idx = idx_b[msk_b]
            if valid_idx.numel() == 0:
                covers.append(0.0)
            else:
                covers.append(torch.unique(valid_idx).numel() / float(N))
        return float(sum(covers) / max(1, len(covers))), covers
    # def grid_pool_downsample_density_adaptive(self, xyz, target_npoint, alpha=0.7, coarse_scale=2.0, mix_fps_ratio=0.3):
    #     """
    #     严格无重复版本（per-batch unique）
    #     xyz: [B, N, 3]
    #     return: {'sampled_index': [B, target_npoint]}
    #     """
    #     B, N, _ = xyz.shape
    #     device = xyz.device

    #     if target_npoint > N:
    #         raise ValueError(f"target_npoint({target_npoint}) cannot be larger than N({N}) for unique sampling.")

    #     out_idx = []

    #     for b in range(B):
    #         pts = xyz[b]  # [N,3]

    #         # 1) coarse grid
    #         bbox = pts.max(dim=0)[0] - pts.min(dim=0)[0]
    #         vol = torch.clamp(bbox.prod(), min=1e-12)
    #         base_gs = (vol / float(target_npoint)) ** (1.0 / 3.0)
    #         gs = float(base_gs * coarse_scale)
    #         grid_size_tensor = torch.full((3,), gs, device=device, dtype=pts.dtype)

    #         cluster = grid_cluster(pts, grid_size_tensor)          # [N]
    #         uniq, inv = torch.unique(cluster, return_inverse=True) # M
    #         M = uniq.numel()

    #         cnt = torch.bincount(inv, minlength=M).float()         # [M]

    #         # 2) density budget
    #         w = torch.pow(cnt, alpha)
    #         k_float = w / (w.sum() + 1e-12) * float(target_npoint)
    #         k = torch.floor(k_float).long().clamp_min(1)           # [M]

    #         # 修正总和到 target_npoint
    #         cur = int(k.sum().item())
    #         if cur > target_npoint:
    #             extra = cur - target_npoint
    #             order = torch.argsort(k, descending=True)
    #             for j in order:
    #                 if extra == 0:
    #                     break
    #                 dec = min(int(k[j].item()) - 1, extra)
    #                 if dec > 0:
    #                     k[j] -= dec
    #                     extra -= dec
    #         elif cur < target_npoint:
    #             need = target_npoint - cur
    #             frac = (k_float - torch.floor(k_float))
    #             order = torch.argsort(frac, descending=True)
    #             t = 0
    #             while need > 0:
    #                 j = order[t % M]
    #                 k[j] += 1
    #                 need -= 1
    #                 t += 1

    #         # 3) per-cell sampling（不重复）
    #         picked = []

    #         for cid in range(M):
    #             pidx = torch.nonzero(inv == cid, as_tuple=False).squeeze(1)  # 原始全局索引
    #             ncell = pidx.numel()
    #             if ncell == 0:
    #                 continue

    #             kb = int(k[cid].item())
    #             kb = min(kb, ncell)  # 关键：不在cell内重复采样

    #             if kb == ncell:
    #                 picked.append(pidx)
    #             else:
    #                 # 局部 FPS（无重复）
    #                 local = pts[pidx]  # [ncell,3]
    #                 sel = torch.empty(kb, dtype=torch.long, device=device)
    #                 far = torch.randint(0, ncell, (1,), device=device).item()
    #                 dist = torch.full((ncell,), 1e10, device=device)
    #                 for t in range(kb):
    #                     sel[t] = far
    #                     c = local[far:far+1]
    #                     d = torch.sum((local - c) ** 2, dim=1)
    #                     dist = torch.minimum(dist, d)
    #                     far = torch.argmax(dist).item()
    #                 picked.append(pidx[sel])

    #         if len(picked) == 0:
    #             # 极端兜底：从全局无放回采样
    #             perm = torch.randperm(N, device=device)[:target_npoint]
    #             out_idx.append(perm)
    #             continue

    #         idx_b = torch.cat(picked, dim=0)              # 可能 < 或 > target
    #         idx_b = torch.unique(idx_b)                   # 全局去重（保持无重复）

    #         # 4) 调整到 target_npoint（仍保持无重复）
    #         if idx_b.numel() > target_npoint:
    #             perm = torch.randperm(idx_b.numel(), device=device)[:target_npoint]
    #             idx_b = idx_b[perm]
    #         elif idx_b.numel() < target_npoint:
    #             needed = target_npoint - idx_b.numel()

    #             used = torch.zeros(N, dtype=torch.bool, device=device)
    #             used[idx_b] = True
    #             remain = torch.nonzero(~used, as_tuple=False).squeeze(1)  # 未选中的池（无重复来源）

    #             if remain.numel() < needed:
    #                 # 理论上不会发生（target<=N），兜底
    #                 needed = remain.numel()

    #             perm = torch.randperm(remain.numel(), device=device)[:needed]
    #             idx_b = torch.cat([idx_b, remain[perm]], dim=0)

    #             # 再保险（应已等于 target）
    #             if idx_b.numel() > target_npoint:
    #                 idx_b = idx_b[:target_npoint]
    #                     # ====== 新增：grid + global FPS 混采，提升覆盖 ======
    #         s_fps = int(target_npoint * mix_fps_ratio)
    #         s_fps = max(0, min(s_fps, target_npoint))
    #         if s_fps > 0:
    #             fps_idx_b = torch.randperm(N, device=device)[:s_fps]  # [s_fps]
    #             # 保留一部分 grid 点，和 FPS 点拼起来
    #             s_grid = target_npoint - s_fps
    #             if idx_b.numel() > s_grid:
    #                 perm = torch.randperm(idx_b.numel(), device=device)[:s_grid]
    #                 idx_grid = idx_b[perm]
    #             else:
    #                 idx_grid = idx_b

    #             idx_b = torch.unique(torch.cat([idx_grid, fps_idx_b], dim=0))

    #             # 不足则补未选点
    #             if idx_b.numel() < target_npoint:
    #                 need = target_npoint - idx_b.numel()
    #                 used = torch.zeros(N, dtype=torch.bool, device=device)
    #                 used[idx_b] = True
    #                 remain = torch.nonzero(~used, as_tuple=False).squeeze(1)
    #                 if remain.numel() > 0:
    #                     perm = torch.randperm(remain.numel(), device=device)[:need]
    #                     idx_b = torch.cat([idx_b, remain[perm]], dim=0)

    #             # 超了则裁剪
    #             if idx_b.numel() > target_npoint:
    #                 perm = torch.randperm(idx_b.numel(), device=device)[:target_npoint]
    #                 idx_b = idx_b[perm]
    #         # ====== 新增结束 ======
    #         out_idx.append(idx_b)
    #     sampled_index = torch.stack(out_idx, dim=0)  # [B, target_npoint]
    #     return {'sampled_index': sampled_index}
    def grid_pool_downsample_density_adaptive(
        self,
        xyz,                    # [B,N,3]
        target_npoint,
        feat=None,              # [B,N,D] or None
        alpha=0.7,
        coarse_scale=2.0,
        mix_fps_ratio=0.2,
        feat_weight=0.7,     
        geo_weight=0.3      
    ):
        B, N, _ = xyz.shape
        device = xyz.device
        if target_npoint > N:
            raise ValueError(f"target_npoint({target_npoint}) > N({N})")

        out_idx = []

        for b in range(B):
            pts = xyz[b]                    # [N,3]
            fts = feat[b] if feat is not None else None  # [N,D]

            # 1) coarse grid
            bbox = pts.max(dim=0)[0] - pts.min(dim=0)[0]
            vol = torch.clamp(bbox.prod(), min=1e-12)
            base_gs = (vol / float(target_npoint)) ** (1.0 / 3.0)
            gs = float(base_gs * coarse_scale)
            grid_size_tensor = torch.full((3,), gs, device=device, dtype=pts.dtype)

            cluster = grid_cluster(pts, grid_size_tensor)              # [N]
            uniq, inv = torch.unique(cluster, return_inverse=True)
            M = uniq.numel()
            cnt = torch.bincount(inv, minlength=M).float()             # [M]

            # 2) density budget
            w = torch.pow(cnt, alpha)
            k_float = w / (w.sum() + 1e-12) * float(target_npoint)
            k = torch.floor(k_float).long().clamp_min(1)

            cur = int(k.sum().item())
            if cur > target_npoint:
                extra = cur - target_npoint
                order = torch.argsort(k, descending=True)
                for j in order:
                    if extra == 0: break
                    dec = min(int(k[j].item()) - 1, extra)
                    if dec > 0:
                        k[j] -= dec
                        extra -= dec
            elif cur < target_npoint:
                need = target_npoint - cur
                frac = (k_float - torch.floor(k_float))
                order = torch.argsort(frac, descending=True)
                t = 0
                while need > 0:
                    j = order[t % M]
                    k[j] += 1
                    need -= 1
                    t += 1

            # 3) per-cell sample: 先feature-medoid，再几何补点
            picked = []
            for cid in range(M):
                pidx = torch.nonzero(inv == cid, as_tuple=False).squeeze(1)
                ncell = pidx.numel()
                if ncell == 0:
                    continue

                kb = min(int(k[cid].item()), ncell)
                if fts is not None:
                    cell_f = fts[pidx]                                 # [ncell,D]
                    mu_f = cell_f.mean(dim=0, keepdim=True)            # [1,D]
                    f_dist = torch.sum((cell_f - mu_f) ** 2, dim=1)    # [ncell]

                    cell_x = pts[pidx]
                    mu_x = cell_x.mean(dim=0, keepdim=True)
                    x_dist = torch.sum((cell_x - mu_x) ** 2, dim=1)    # [ncell]
                    f_dist_n = f_dist / (f_dist.mean() + 1e-12)
                    x_dist_n = x_dist / (x_dist.mean() + 1e-12)
                    score = feat_weight * f_dist_n + geo_weight * x_dist_n
                    first_local = torch.argmin(score)
                else:
                    cell_x = pts[pidx]
                    mu_x = cell_x.mean(dim=0, keepdim=True)
                    x_dist = torch.sum((cell_x - mu_x) ** 2, dim=1)
                    first_local = torch.argmin(x_dist)

                selected = [first_local.item()]

            
                if kb > 1:
                    cell_x = pts[pidx]                                  # [ncell,3]
                    dist = torch.sum((cell_x - cell_x[first_local:first_local+1]) ** 2, dim=1)
                    dist[first_local] = -1.0
                    for _ in range(kb - 1):
                        far = torch.argmax(dist).item()
                        selected.append(far)
                        d = torch.sum((cell_x - cell_x[far:far+1]) ** 2, dim=1)
                        dist = torch.minimum(torch.where(dist < 0, d, dist), d)
                        dist[selected] = -1.0

                picked.append(pidx[torch.tensor(selected, device=device, dtype=torch.long)])

            if len(picked) == 0:
                idx_b = torch.randperm(N, device=device)[:target_npoint]
                out_idx.append(idx_b)
                continue

            idx_b = torch.unique(torch.cat(picked, dim=0))

            if idx_b.numel() > target_npoint:
                idx_b = idx_b[torch.randperm(idx_b.numel(), device=device)[:target_npoint]]
            elif idx_b.numel() < target_npoint:
                need = target_npoint - idx_b.numel()
                used = torch.zeros(N, dtype=torch.bool, device=device)
                used[idx_b] = True
                remain = torch.nonzero(~used, as_tuple=False).squeeze(1)
                if remain.numel() > 0:
                    idx_b = torch.cat([idx_b, remain[torch.randperm(remain.numel(), device=device)[:need]]], dim=0)

            s_mix = int(target_npoint * mix_fps_ratio)
            if s_mix > 0:
                s_mix = min(s_mix, target_npoint)
                s_grid = target_npoint - s_mix
                if idx_b.numel() > s_grid:
                    idx_grid = idx_b[torch.randperm(idx_b.numel(), device=device)[:s_grid]]
                else:
                    idx_grid = idx_b
                idx_rand = torch.randperm(N, device=device)[:s_mix]
                idx_b = torch.unique(torch.cat([idx_grid, idx_rand], dim=0))
                if idx_b.numel() < target_npoint:
                    need = target_npoint - idx_b.numel()
                    used = torch.zeros(N, dtype=torch.bool, device=device); used[idx_b] = True
                    remain = torch.nonzero(~used, as_tuple=False).squeeze(1)
                    if remain.numel() > 0:
                        idx_b = torch.cat([idx_b, remain[torch.randperm(remain.numel(), device=device)[:need]]], dim=0)
                if idx_b.numel() > target_npoint:
                    idx_b = idx_b[torch.randperm(idx_b.numel(), device=device)[:target_npoint]]

            out_idx.append(idx_b)

        return {'sampled_index': torch.stack(out_idx, dim=0)}
    def grid_pool_downsample(self, xyz, target_npoint, return_counts=True):
        """
        Args:
            xyz: [B, N, 3]
            target_npoint: int
            return_counts: bool
        Returns:
            {
                'sampled_xyz': [B, target_npoint, 3],  
                'inverse': List[Tensor[N]]            
            }
        """
        B, N, _ = xyz.shape
        device = xyz.device

        if self.grid_size is None:
            bbox = xyz.max(dim=1)[0] - xyz.min(dim=1)[0]
            vol = bbox.prod(dim=1).mean()
            target_npoint_tensor = torch.tensor(target_npoint, device=device, dtype=xyz.dtype)
            init_gs = max(1e-4, (vol / target_npoint_tensor) ** (1/3) * 0.15)
            self.grid_size = float(init_gs) 
        
        min_gs = self.grid_size * 0.01   
        max_gs = self.grid_size * 1000.0 
        
        best_gs = self.grid_size
        best_result = None
        best_num = 0

        for _ in range(20):  
            gs_val = float(best_gs) 
            grid_size_tensor = torch.full((3,), gs_val, device=device, dtype=xyz.dtype)
            
            all_centroids, all_counts, all_inverses = [], [], []
            total_clusters = 0
            
            for b in range(B):
                points_b = xyz[b]  # [N, 3]
                cluster_b = grid_cluster(points_b, grid_size_tensor)  # [N]
                unique_clusters, inverse_b = torch.unique(cluster_b, return_inverse=True)  # [N], 0~num_voxels-1
                
                centroids_b = scatter_mean(points_b, inverse_b, dim=0)  # [num_voxels, 3]
                
                if return_counts:
                    ones = torch.ones_like(inverse_b, dtype=xyz.dtype)
                    counts_b = scatter_add(ones, inverse_b, dim=0, dim_size=unique_clusters.shape[0])
                else:
                    counts_b = None
                    
                all_centroids.append(centroids_b)
                all_counts.append(counts_b)
                all_inverses.append(inverse_b) 
                total_clusters += centroids_b.shape[0]
            
            if abs(total_clusters - target_npoint) < abs(best_num - target_npoint):
                best_num = total_clusters
                best_gs = gs_val
                best_result = (all_centroids, all_counts if return_counts else None, all_inverses)

            if total_clusters < target_npoint * 0.995:
                max_gs = gs_val
                best_gs = (min_gs + gs_val) / 2
            elif total_clusters > target_npoint * 1.005:
                min_gs = gs_val
                best_gs = (gs_val + max_gs) / 2
            else:
                break
        
        if best_result is None:
            print("Can't find a satisfied grid_size for grid_pool_sample")
            return None
        
        centroids_list, counts_list, inverses_list = best_result

        final_xyz, final_counts, final_inverses = [], [], []
        
        for b in range(B):
            centroids_b = centroids_list[b]       
            counts_b = counts_list[b] if return_counts else None  # [current_n]
            inverse_b = inverses_list[b]           
            current_n = centroids_b.shape[0]
            
            if current_n > target_npoint:
                perm = torch.randperm(current_n, device=device)
                sel_idx = perm[:target_npoint]  
                
                sampled_xyz = centroids_b[sel_idx]
                sampled_cnt = counts_b[sel_idx] if return_counts else None
                
                old2new = torch.full((current_n,), -1, dtype=torch.long, device=device)
                for new_idx, old_idx in enumerate(sel_idx):
                    old2new[old_idx] = new_idx
                
                new_inverse = old2new[inverse_b]  # [N]

                mask = (new_inverse == -1)
                if mask.any():
                    new_inverse[mask] = torch.randint(0, int(target_npoint), (mask.sum(),), device=device)
                    
            elif current_n < target_npoint:

                deficit = target_npoint - current_n
                if current_n > 0 and deficit <= 0.3 * target_npoint:
              
                    rand_idx = torch.randint(0, current_n, (deficit,), device=device)
                    sampled_xyz = torch.cat([centroids_b, centroids_b[rand_idx]], dim=0)
                    sampled_cnt = torch.cat([counts_b, counts_b[rand_idx]], dim=0) if return_counts else None
                    
                   
                    new_inverse = inverse_b  
                    
                elif current_n > 0:
        
                    print(f"[WARN] Batch {b}: Only {current_n} clusters, target {target_npoint}.")
                    repeat_needed = (deficit + current_n - 1) // current_n
                    extra_idx = torch.tensor([i % current_n for i in range(deficit)], device=device)
                    
                    extra_xyz = centroids_b[extra_idx]
                    extra_cnt = counts_b[extra_idx] if return_counts else None
                    sampled_xyz = torch.cat([centroids_b, extra_xyz], dim=0)
                    sampled_cnt = torch.cat([counts_b, extra_cnt], dim=0) if return_counts else None
                    
                    new_inverse = inverse_b 
                else:
                    
                    sampled_xyz = torch.zeros((target_npoint, 3), device=device, dtype=xyz.dtype)
                    sampled_cnt = torch.zeros(target_npoint, device=device, dtype=torch.long) if return_counts else None
                    new_inverse = torch.zeros(N, dtype=torch.long, device=device)  
            else:
          
                sampled_xyz = centroids_b
                sampled_cnt = counts_b if return_counts else None
                new_inverse = inverse_b  
            
            final_xyz.append(sampled_xyz)
            if return_counts:
                final_counts.append(sampled_cnt)
            final_inverses.append(new_inverse) 
        
        result = {
            'sampled_xyz': torch.stack(final_xyz),  # [B, target_npoint, 3]
            'inverse': final_inverses            
        }
        if return_counts:
            result['counts'] = torch.stack(final_counts)  # [B, target_npoint]
        
        return result
    def export_zero_hit_xyz(self,xyz, group_idx, group_mask, out_prefix="zero_hit"):
        """
        xyz: [B,N,3] (torch, cuda/cpu)
        group_idx: [B,S,K]
        group_mask: [B,S,K] bool
        导出每个batch的:
        - {out_prefix}_b{b}_all.xyz
        - {out_prefix}_b{b}_zero_hit.xyz
        - {out_prefix}_b{b}_hit.xyz
        """
        xyz = xyz.detach()
        B, N, _ = xyz.shape

        for b in range(B):
            idx_b = group_idx[b].reshape(-1)
            msk_b = group_mask[b].reshape(-1)

            valid_idx = idx_b[msk_b]  # 仅有效邻居命中
            hit = torch.zeros(N, dtype=torch.bool, device=xyz.device)
            if valid_idx.numel() > 0:
                hit[torch.unique(valid_idx)] = True
            zero = ~hit

            all_xyz = xyz[b].detach().cpu().numpy()
            hit_xyz = xyz[b][hit].detach().cpu().numpy()
            zero_xyz = xyz[b][zero].detach().cpu().numpy()

            np.savetxt(f"{out_prefix}_b{b}_all.xyz", all_xyz, fmt="%.6f")
            np.savetxt(f"{out_prefix}_b{b}_hit.xyz", hit_xyz, fmt="%.6f")
            np.savetxt(f"{out_prefix}_b{b}_zero_hit.xyz", zero_xyz, fmt="%.6f")

            print(f"[b={b}] all={len(all_xyz)} hit={len(hit_xyz)} zero={len(zero_xyz)} ratio={len(zero_xyz)/max(1,len(all_xyz)):.4f}")
    def _save_sampled_points_txt(self, xyz, fps_idx, tag="sa", out_dir="./debug_samples"):
        """
        xyz: [B,N,3] (after permute)
        fps_idx: [B,S]
        保存:
        - {tag}_b{b}_sampled_xyz.txt  (x y z)
        - {tag}_b{b}_sampled_idx.txt  (index)
        """
        os.makedirs(out_dir, exist_ok=True)

        with torch.no_grad():
            B = xyz.shape[0]
            for b in range(B):
                idx_b = fps_idx[b].detach().cpu().long().numpy()   # [S]
                pts_b = xyz[b, idx_b, :].detach().cpu().numpy()    # [S,3]

                xyz_path = os.path.join(out_dir, f"{tag}_b{b}_sampled_xyz.txt")
                idx_path = os.path.join(out_dir, f"{tag}_b{b}_sampled_idx.txt")

                # CloudCompare可直接读 x y z 三列
                np.savetxt(xyz_path, pts_b, fmt="%.6f")
                np.savetxt(idx_path, idx_b, fmt="%d")
    # def forward(self, xyz, points, self_define_samples=None, debug_cover=True):
    #     """
    #     Input:
    #         xyz: [B, C, N] -> [B, N, 3]
    #         points: [B, D, N] -> [B, N, D]
    #     Return:
    #         new_xyz: [B, 3, S]
    #         new_points_concat: [B, D', S]
    #     """
    #     xyz = xyz.permute(0, 2, 1).contiguous()  # [B, N, 3]
    #     if points is not None:
    #         points = points.permute(0, 2, 1).contiguous()  # [B, N, D]

    #     B, N, C = xyz.shape
    #     S = self_define_samples if self_define_samples else self.npoint

    #     # results = self.grid_pool_downsample_density_adaptive(xyz, S)
    #     results = self.grid_pool_downsample_density_adaptive(xyz, S, alpha=0.7, coarse_scale=2.0, mix_fps_ratio=0.2)
    #     if results is None:
    #         return None, None

    #     fps_idx = results['sampled_index']          # [B, S]
    #     new_xyz = index_points(xyz, fps_idx)        # [B, S, 3]
    #     if new_xyz is None:
    #         return None, None

    #     new_points_list = []
    #     finfo_min = torch.finfo(xyz.dtype).min

    #     for i, radius in enumerate(self.radius_list):
    #         K = self.nsample_list[i]

    #         out = query_ball_point_knn_aligned(radius, K, xyz, new_xyz)
    #         group_idx, group_mask = out
    #         if group_idx is None or group_mask is None:
    #             return None, None

    #         # ===== 覆盖率守门重试（建议先只开第一层 i==0）=====
    #         enable_cover_gate = (i == 0)   # 先只守第一层，稳定后可放开
    #         cover_thr = 0.95
    #         max_retry = 2

    #         if enable_cover_gate:
    #             avg_cover, _ = self._compute_cover_from_group(group_idx, group_mask, N)

    #             retry = 0
    #             radius_try = float(radius)
    #             while (avg_cover < cover_thr) and (retry < max_retry):
    #                 # 每次放大半径再重算
    #                 radius_try = radius_try * (1.25 if retry == 0 else 1.15)
    #                 out_retry = query_ball_point_knn_aligned(radius_try, K, xyz, new_xyz)
    #                 gi, gm = out_retry
    #                 if gi is None or gm is None:
    #                     break

    #                 group_idx, group_mask = gi, gm
    #                 avg_cover, _ = self._compute_cover_from_group(group_idx, group_mask, N)
    #                 retry += 1

    #             if debug_cover:
    #                 print(f"[COVER-GATE] scale={i} final_radius={radius_try:.5f}, avg_cover={avg_cover:.4f}, retry={retry}")
    #         # ===== 守门结束 =====
    #         if group_idx is None or group_mask is None:
    #             return None, None
    #         if debug_cover:
    #             self.export_zero_hit_xyz(xyz,group_idx,group_mask)
    #             self._save_sampled_points_txt(xyz, fps_idx, tag=f"sa_n{S}", out_dir="./debug_samples")
    #             # with torch.no_grad():
    #             #     for b in range(B):
    #             #         idx_b = group_idx[b].reshape(-1)
    #             #         msk_b = group_mask[b].reshape(-1)
    #             #         valid_idx_b = idx_b[msk_b]
    #             #         if valid_idx_b.numel() == 0:
    #             #             cover = 0.0
    #             #             zero_hit = N
    #             #             max_hit = 0
    #             #             mean_hit = 0.0
    #             #         else:
    #             #             uniq = torch.unique(valid_idx_b)
    #             #             cover = uniq.numel() / float(N)
    #             #             hit_count = torch.bincount(valid_idx_b, minlength=N)
    #             #             zero_hit = int((hit_count == 0).sum().item())
    #             #             max_hit = int(hit_count.max().item())
    #             #             mean_hit = float(hit_count.float().mean().item())
    #             #         print(f"[VALID-COVER] scale={i} b={b} cover={cover:.4f}, zero_hit={zero_hit}/{N} ({zero_hit/float(N):.4f}), max_hit={max_hit}, mean_hit={mean_hit:.4f}")

    #         grouped_xyz = index_points(xyz, group_idx)              # [B,S,K,3]
    #         grouped_xyz = grouped_xyz - new_xyz.view(B, S, 1, C)

    #         if points is not None:
    #             grouped_points = index_points(points, group_idx)    # [B,S,K,D]
    #             grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B,S,K,D+3]
    #         else:
    #             grouped_points = grouped_xyz

    #         grouped_points = grouped_points.permute(0, 3, 2, 1).contiguous()  # [B,D+3,K,S]

    #         for j in range(len(self.conv_blocks[i])):
    #             grouped_points = F.leaky_relu(
    #                 (self.conv_blocks[i][j](grouped_points)),
    #                 negative_slope=0.01,
    #                 inplace=True
    #             )
    #         # masked soft aggregation (replace masked max pooling)
    #         mask = group_mask.permute(0, 2, 1).unsqueeze(1)  # [B,1,K,S], bool

    #         # 用局部几何距离做attention logits（更稳）
    #         # grouped_xyz: [B,S,K,3] -> dist2: [B,1,K,S]
    #         dist2 = torch.sum(grouped_xyz ** 2, dim=-1).permute(0, 2, 1).unsqueeze(1)

    #         tau = 0.07  # 可调: 0.03~0.1（越小越接近hard选择）
    #         logits = -dist2 / tau

    #         neg_inf = torch.finfo(logits.dtype).min
    #         logits = logits.masked_fill(~mask, neg_inf)

    #         attn = torch.softmax(logits, dim=2)                 # [B,1,K,S]
    #         attn = torch.where(mask, attn, torch.zeros_like(attn))

    #         # 对邻居维K加权求和，替代max
    #         new_points = torch.sum(grouped_points * attn, dim=2)  # [B,D',S]

    #         # 若某query没有任何有效邻居，置零兜底
    #         valid_any = mask.any(dim=2)  # [B,1,S]
    #         new_points = torch.where(valid_any, new_points, torch.zeros_like(new_points))
    #         # masked max pooling
    #         # mask = group_mask.permute(0, 2, 1).unsqueeze(1)        # [B,1,K,S]
    #         # grouped_points = grouped_points.masked_fill(~mask, finfo_min)
    #         # new_points = torch.max(grouped_points, 2)[0]            # [B,D',S]
    #         # new_points = torch.where(torch.isfinite(new_points), new_points, torch.zeros_like(new_points))

    #         new_points_list.append(new_points)

    #     new_xyz = new_xyz.permute(0, 2, 1).contiguous()            # [B,3,S]
    #     new_points_concat = torch.cat(new_points_list, dim=1)      # [B,D'_total,S]
    #     return new_xyz, new_points_concat
    def forward(self, xyz, points, self_define_samples=None):
        """
        Input:
            xyz: [B, C, N] -> [B, N, 3]
            points: [B, D, N] -> [B, N, D]
        Return:
            new_xyz: [B, C, S]
            new_points_concat: [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, 3]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]
        
        B, N, C = xyz.shape
        S = self_define_samples if self_define_samples else self.npoint
        results = self.grid_pool_downsample(xyz, S)  # [B, S]
        if results == None:
            return None,None
        # fps_idx = results['sampled_index']  # [B, S]
        # if fps_idx == None:
        #     return None,None
        new_xyz = results['sampled_xyz']  # [B, S, 3]

        ################return centoid xyz#################
        # new_xyz = results['sampled_xyz']
        # ones = torch.ones_like(results['counts'])
        # mask = results['counts'] >= 3
        # counts_up_one = ones[mask].sum()
        # max_num_in_voxel , min_num_in_voxel, total_num_all_voxel = results['counts'].max(), results['counts'].min(), results['counts'].sum()

        # import matplotlib.pyplot as plt
        # import numpy as np
        
        # counts = results['counts'].detach().cpu().numpy()
        
        # counts_flat = counts.flatten()
        # x_axis = np.arange(len(counts_flat))  
        # y_axis = counts_flat                   
        
        # # counts_flat = counts[0]              
        # # x_axis = np.arange(len(counts_flat))
        # # y_axis = counts_flat
        
        # plt.figure(figsize=(12, 4))
        # plt.plot(x_axis, y_axis, linewidth=0.6, color='#2E86AB', label='Points per Voxel')
        
        # plt.title(f"Voxel Index vs Point Count\nBatch={counts.shape[0]}, Voxels={len(counts_flat)}, Max={y_axis.max()}, Min={y_axis.min()}, Mean={y_axis.mean():.2f}")
        # plt.xlabel("Voxel Index (Flattened)")
        # plt.ylabel("Number of Points in Voxel")
        # plt.grid(True, linestyle='--', alpha=0.3)
        # plt.tight_layout()
        
        # plt.savefig("debug_voxel_index_curve.png", dpi=150, bbox_inches='tight')
        # plt.close()
        if new_xyz == None:
            return None,None
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]

            group_idx, group_mask = query_ball_point_knn_aligned(radius, K, xyz, new_xyz)  # [B, S, K]
            if group_idx == None:
                return None, None
            counts_all = group_idx.shape[1] * group_idx.shape[2]
            coverage_ratio = torch.unique(group_idx).numel()/N
            # 在 group_idx 得到后插入这段调试代码
            # group_idx: [B, S, K]
            # N: 当前层输入点数（前面已有 B, N, C = xyz.shape）

            # with torch.no_grad():
            #     # 1) 每个 batch 的 unique 覆盖率
            #     cover_rates = []
            #     hit_stats = []  # 保存 (min_hit, max_hit, mean_hit)

            #     for b in range(B):
            #         idx_b = group_idx[b].reshape(-1)                 # [S*K]
            #         unique_idx_b = torch.unique(idx_b)               # unique indices in [0, N-1]
            #         cover_rate_b = unique_idx_b.numel() / float(N)
            #         cover_rates.append(cover_rate_b)

            #         # 2) 每个点被命中次数（看是否大量点被忽略）
            #         hit_count_b = torch.bincount(idx_b, minlength=N) # [N]
            #         min_hit = int(hit_count_b.min().item())
            #         max_hit = int(hit_count_b.max().item())
            #         mean_hit = float(hit_count_b.float().mean().item())
            #         zero_hit = int((hit_count_b == 0).sum().item())  # 完全未被选中的点数
            #         zero_rate = zero_hit / float(N)

            #         hit_stats.append((min_hit, max_hit, mean_hit, zero_hit, zero_rate))

            #     # 3) 打印当前层统计
            #     avg_cover = sum(cover_rates) / len(cover_rates)
            #     print(f"[KNN-COVER] avg_cover={avg_cover:.4f}, per_batch={['%.4f'%x for x in cover_rates]}")
            #     for b, (mn, mx, meanv, zh, zr) in enumerate(hit_stats):
            #         print(f"[KNN-HIT] batch={b} min={mn} max={mx} mean={meanv:.4f} zero_hit={zh}/{N} ({zr:.4f})")
            with torch.no_grad():
                for b in range(B):
                    idx_b = group_idx[b].reshape(-1)          # [S*K]
                    msk_b = group_mask[b].reshape(-1)         # [S*K] bool

                    valid_idx_b = idx_b[msk_b]                # 仅有效邻居
                    if valid_idx_b.numel() == 0:
                        cover = 0.0
                        zero_hit = N
                        max_hit = 0
                        mean_hit = 0.0
                    else:
                        uniq = torch.unique(valid_idx_b)
                        cover = uniq.numel() / float(N)

                        hit_count = torch.bincount(valid_idx_b, minlength=N)
                        zero_hit = int((hit_count == 0).sum().item())
                        max_hit = int(hit_count.max().item())
                        mean_hit = float(hit_count.float().mean().item())

                    print(f"[VALID-COVER] b={b} cover={cover:.4f}, zero_hit={zero_hit}/{N} ({zero_hit/float(N):.4f}), max_hit={max_hit}, mean_hit={mean_hit:.4f}")

            grouped_xyz = index_points(xyz, group_idx)  # [B, S, K, 3]
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            
            if points is not None:
                grouped_points = index_points(points, group_idx)  # [B, S, K, D]
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B, S, K, D+3]
            else:
                grouped_points = grouped_xyz
            
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D+3, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                # bn = self.bn_blocks[i][j]
                grouped_points = F.leaky_relu((conv(grouped_points)))
             # masked soft aggregation (replace masked max pooling)
            mask = group_mask.permute(0, 2, 1).unsqueeze(1)  # [B,1,K,S], bool

            # 用局部几何距离做attention logits（更稳）
            # grouped_xyz: [B,S,K,3] -> dist2: [B,1,K,S]
            # mask_sum = mask.sum(dim=-2)
            # print(mask_sum.tolist())
            dist2 = torch.sum(grouped_xyz ** 2, dim=-1).permute(0, 2, 1).unsqueeze(1)

            tau = 0.07  # 可调: 0.03~0.1（越小越接近hard选择）
            logits = -dist2 / tau

            neg_inf = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, neg_inf)

            attn = torch.softmax(logits, dim=2)                 # [B,1,K,S]

            attn = torch.where(mask, attn, torch.zeros_like(attn))

            # 对邻居维K加权求和，替代max
            new_points = torch.sum(grouped_points * attn, dim=2)  # [B,D',S]

            # 若某query没有任何有效邻居，置零兜底
            valid_any = mask.any(dim=2)  # [B,1,S]
            new_points = torch.where(valid_any, new_points, torch.zeros_like(new_points))
            # masked max pooling
            # mask = group_mask.permute(0, 2, 1).unsqueeze(1)        # [B,1,K,S]
            # grouped_points = grouped_points.masked_fill(~mask, finfo_min)
            # new_points = torch.max(grouped_points, 2)[0]            # [B,D',S]
            # new_points = torch.where(torch.isfinite(new_points), new_points, torch.zeros_like(new_points))

            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1).contiguous()            # [B,3,S]
        new_points_concat = torch.cat(new_points_list, dim=1)      # [B,D'_total,S]
        return new_xyz, new_points_concat
        #     new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
        #     new_points_list.append(new_points)
        
        # new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, S]
        # new_points_concat = torch.cat(new_points_list, dim=1)  # [B, D'_total, S]
        # return new_xyz, new_points_concat
# class PointNetFeaturePropagation(nn.Module):
#     def __init__(self, in_channel, mlp):
#         super(PointNetFeaturePropagation, self).__init__()
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm1d(out_channel))
#             last_channel = out_channel

#     def forward(self, xyz1, xyz2, points1, points2):
#         """
#         Input:
#             xyz1: input points position data, [B, C, N]
#             xyz2: sampled input points position data, [B, C, S]
#             points1: input points data, [B, D, N]
#             points2: input points data, [B, D, S]
#         Return:
#             new_points: upsampled points data, [B, D', N]
#         """
#         xyz1 = xyz1.permute(0, 2, 1)
#         xyz2 = xyz2.permute(0, 2, 1)

#         points2 = points2.permute(0, 2, 1)
#         B, N, C = xyz1.shape
#         _, S, _ = xyz2.shape

#         if S == 1:
#             interpolated_points = points2.repeat(1, N, 1)
#         else:
#             dists = square_distance(xyz1, xyz2)
#             dists, idx = dists.sort(dim=-1)
#             dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

#             dist_recip = 1.0 / (dists + 1e-8)
#             norm = torch.sum(dist_recip, dim=2, keepdim=True)
#             weight = dist_recip / norm
#             interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

#         if points1 is not None:
#             points1 = points1.permute(0, 2, 1)
#             new_points = torch.cat([points1, interpolated_points], dim=-1)
#         else:
#             new_points = interpolated_points

#         new_points = new_points.permute(0, 2, 1)
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             new_points = F.relu(bn(conv(new_points)))
#         return new_points
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def square_distance(src, dst):
#     """
#     Input:
#         src: [B, N, C]
#         dst: [B, M, C]
#     Output:
#         dist: [B, N, M]
#     """
#     B, N, _ = src.shape
#     _, M, _ = dst.shape
#     dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
#     dist += torch.sum(src ** 2, -1).view(B, N, 1)
#     dist += torch.sum(dst ** 2, -1).view(B, 1, M)
#     return dist

# def index_points(points, idx):
#     """
#     Input:
#         points: input points data, [B, D, N] or [B, N, D]
#         idx: sample index data, [B, S] or [B, S, K]
#     Return:
#         new_points: indexed points data, [B, S, D] or [B, S, K, D]
#     """
#     device = points.device
#     B = points.shape[0]
#     view_shape = list(idx.shape)
#     view_shape[1:] = [1] * (len(view_shape) - 1)
#     repeat_shape = list(idx.shape)
#     repeat_shape[0] = 1
#     batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
#     new_points = points[batch_indices, idx, :]
#     return new_points

# class PointNetFeaturePropagation(nn.Module):
#     def __init__(self, in_channel, mlp):
#         super(PointNetFeaturePropagation, self).__init__()
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm1d(out_channel))
#             last_channel = out_channel

#     def forward(self, xyz1, xyz2, points1, points2):
#         xyz1 = xyz1.permute(0, 2, 1)  # [B, N, 3]
#         xyz2 = xyz2.permute(0, 2, 1)  # [B, S, 3]
#         points2 = points2.permute(0, 2, 1)  # [B, S, D]
        
#         B, N, C = xyz1.shape
#         _, S, _ = xyz2.shape
#         if S == 1:
#             interpolated_points = points2.repeat(1, N, 1)
#         else:
#             interpolated_points = torch.zeros(B, N, points2.shape[-1], device=xyz1.device, dtype=points2.dtype)
            
#             chunk_size = 1024 
            
#             for i in range(0, N, chunk_size):
#                 end_i = min(i + chunk_size, N)
                
#                 dists = square_distance(xyz1[:, i:end_i, :], xyz2)  
                
#                 dists, idx = torch.topk(dists, k=4, dim=-1, largest=False)

#                 dist_recip = 1.0 / (dists + 1e-8)
#                 norm = torch.sum(dist_recip, dim=2, keepdim=True)
#                 weight = dist_recip / norm
                
#                 nearest_points = index_points(points2, idx) 
#                 interpolated_chunk = torch.sum(nearest_points * weight.view(B, -1, 4, 1), dim=2)
                
#                 interpolated_points[:, i:end_i, :] = interpolated_chunk

#         if points1 is not None:
#             points1 = points1.permute(0, 2, 1)
#             new_points = torch.cat([points1, interpolated_points], dim=-1)
#         else:
#             new_points = interpolated_points

#         new_points = new_points.permute(0, 2, 1)
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             new_points = F.leaky_relu(bn(conv(new_points)))
            
#         return new_points
class PointNetFeaturePropagationFast(nn.Module):
    def __init__(self, in_channel, mlp, k=3):
        super().__init__()
        self.k = k
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            # self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)  # [B, N, 3]
        xyz2 = xyz2.permute(0, 2, 1)  # [B, S, 3]
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)  # [B, N, D1]
        points2 = points2.permute(0, 2, 1)      # [B, S, D2]
        
        B, N, _ = xyz1.shape
        S = xyz2.shape[1]
        device = xyz1.device

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            k_safe = min(self.k, S)

            batch_x = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(S)
            batch_y = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(N)
            
            edge_index = knn(xyz2.reshape(-1, 3), xyz1.reshape(-1, 3),
                             k=k_safe, batch_x=batch_x, batch_y=batch_y)[1]
            
            idx = edge_index.view(B, N, k_safe)  # [B, N, K]
            batch_offset = torch.arange(B, device=device).repeat_interleave(N * k_safe) * S
            global_idx = batch_offset + idx.reshape(-1)  # [B*N*K]

            xyz2_flat = xyz2.reshape(B * S, 3)
            xyz2_neigh = xyz2_flat[global_idx].view(B, N, k_safe, 3)
            dists = torch.sum((xyz1.unsqueeze(2) - xyz2_neigh) ** 2, dim=-1)  # [B, N, K]

            dist_recip = 1.0 / (dists + 1e-8)
            weight = dist_recip / dist_recip.sum(dim=2, keepdim=True)  # [B, N, K]

            pts2_flat = points2.reshape(B * S, -1)
            pts2_neigh = pts2_flat[global_idx].view(B, N, k_safe, -1)
            interpolated_points = torch.sum(pts2_neigh * weight.unsqueeze(-1), dim=2)  # [B, N, D2]

        new_points = torch.cat([points1, interpolated_points], dim=-1) if points1 is not None else interpolated_points
        new_points = new_points.permute(0, 2, 1)  # [B, D, N]
        for conv in self.mlp_convs:
            new_points = F.leaky_relu(conv(new_points))
            
        return new_points