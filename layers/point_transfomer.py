import torch
import numpy as np
import torch.nn as nn
from time import time
import torch.nn.functional as F
from torch_cluster import grid_cluster,knn
from torch_scatter import scatter_max, scatter_mean

class PointTransformerLayerForSA(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=3):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        
        # Q/K/V 投影
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        
        # 位置编码 MLP
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            nn.LayerNorm(3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(3, out_planes)  # 输出 out_planes
        )
        
        # 🔧 修复：linear_w 最后输出 out_planes（与 x_v + p_r 对齐）
        self.linear_w = nn.Sequential(
            nn.LayerNorm(mid_planes),
            nn.LeakyReLU(inplace=True),
            nn.Linear(mid_planes,out_planes)
        )
        self.softmax = nn.Softmax(dim=2)  # 在 nsample 维度 softmax
        
    def forward(self, xyz, feat):
        B, S, _ = xyz.shape
        device = xyz.device
        
        # 1️⃣ KNN 定义 attention 邻域
        batch_xyz = torch.arange(B, device=device).repeat_interleave(S)
        edge_index = knn(
            xyz.reshape(-1, 3), xyz.reshape(-1, 3),
            k=self.nsample, batch_x=batch_xyz, batch_y=batch_xyz
        )
        # q_idx = edge_index[1].view(B, S, self.nsample)
        k_idx = edge_index[1].view(B, S, self.nsample)
        
        # 2️⃣ Gather 特征
        x_q = self.linear_q(feat)  # [B, S, mid_planes]
        feat_flat = feat.view(B * S, -1)
        k_feat = feat_flat[k_idx.view(-1)].view(B, S, self.nsample, -1)
        x_k = self.linear_k(k_feat)  # [B, S, nsample, mid_planes]
        x_v = self.linear_v(k_feat)  # [B, S, nsample, out_planes]
        
        # 3️⃣ 相对位置编码
        xyz_flat = xyz.view(B * S, 3)
        k_xyz = xyz_flat[k_idx.view(-1)].view(B, S, self.nsample, 3)
        q_xyz = xyz.unsqueeze(2)  # [B, S, 1, 3]
        rel_pos = k_xyz - q_xyz  # [B, S, nsample, 3]
        
        p_r = self.linear_p(rel_pos)  # [B, S, nsample, out_planes]
        
        # 4️⃣ 计算 attention weight
        w = x_k - x_q.unsqueeze(2) + p_r  # [B, S, nsample, out_planes]
        w = self.linear_w(w)  # 🔧 现在输出 [B, S, nsample, out_planes]
        w = self.softmax(w)  # [B, S, nsample, out_planes]
        
        # 5️⃣ 加权聚合（维度现在对齐！）
        out = ((x_v + p_r) * w).sum(dim=2)  # [B, S, out_planes] ✓
        return out

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

def query_ball_point_knn_aligned(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device
    
    batch_xyz = torch.arange(B, device=device).repeat_interleave(N)
    batch_new = torch.arange(B, device=device).repeat_interleave(S)
    
    # 1. 获取按距离排序的索引
    idx = knn(xyz.reshape(-1, 3), new_xyz.reshape(-1, 3), 
              k=nsample, batch_x=batch_xyz, batch_y=batch_new)[1]
    idx = idx.view(B, S, nsample)  # [B, S, K]
    
    # 2. 打乱顺序，对齐原版 sorted=False 的无序行为
    # perm = torch.stack([torch.randperm(nsample, device=device) for _ in range(B * S)])
    # idx = idx.view(B * S, nsample)[torch.arange(B * S, device=device).unsqueeze(1), perm]
    
    return idx.view(B, S, nsample)

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

# class PointNetSetAbstractionMsg_GridPool(nn.Module):
#     def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, grid_size=None):
#         super().__init__()
#         self.npoint = npoint
#         self.radius_list = radius_list
#         self.nsample_list = nsample_list
#         self.grid_size = grid_size

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

#     # def grid_pool_downsample(self, xyz, target_npoint):
#     #     """
#     #     Args:
#     #         xyz: [B, N, 3] - GPU tensor
#     #         target_npoint: int 
#     #     Returns:
#     #         indices: [B, target_npoint]
#     #     """
#     #     B, N, _ = xyz.shape
#     #     device = xyz.device
 
#     #     if self.grid_size is None:
#     #         bbox = xyz.max(dim=1)[0] - xyz.min(dim=1)[0]
#     #         vol = bbox.prod(dim=1).mean()                
#     #         target_npoint = torch.Tensor(target_npoint).to('cuda')
#     #         self.grid_size = max(1e-4, (vol / target_npoint) ** (1/3) * 0.15)
            
#     #     max_attempts = 3
#     #     for attempt in range(max_attempts):
#     #         grid_size_tensor = torch.full((3,), self.grid_size.item(), device=device, dtype=xyz.dtype)
#     #         cluster = grid_cluster(xyz.reshape(-1, 3), grid_size_tensor)  # [B*N]
#     #         unique_clusters, inverse = torch.unique(cluster, return_inverse=True)
#     #         num_clusters = unique_clusters.shape[0]
            
#     #         if num_clusters >= target_npoint * 0.8:
#     #             break
#     #         self.grid_size *= 0.5 
#     #     else:
#     #         print(f"[WARN] Grid clustering yielded only {num_clusters} clusters "
#     #             f"(target: {target_npoint}). Fallback to random sampling for remaining.")
            
#     #     min_indices = torch.full_like(unique_clusters, 10**9, device=device)
#     #     point_indices = torch.arange(B * N, device=device)
#     #     min_indices.scatter_reduce_(0, inverse, point_indices, reduce='min')
        
#     #     representative_indices = min_indices % N
#     #     batch_of_reps = torch.arange(B, device=device).repeat_interleave(N)[min_indices]
        
#     #     final_indices = []
#     #     for b in range(B):
#     #         mask = batch_of_reps == b
#     #         indices_b = representative_indices[mask]
#     #         current_n = indices_b.shape[0]
            
#     #         if current_n > target_npoint:
#     #             perm = torch.randperm(current_n, device=device)
#     #             indices_b = indices_b[perm][:target_npoint]
#     #         elif current_n < target_npoint:
#     #             deficit = target_npoint - current_n
#     #             if deficit > 0.5 * target_npoint:
#     #                 print(f"[WARN] Batch {b}: Only {current_n} clusters. "
#     #                     f"Padding {deficit} points may hurt spatial uniformity.")
                
#     #             rand_indices = torch.randint(0, N, (deficit,), device=device)
#     #             indices_b = torch.cat([indices_b, rand_indices], dim=0)
                
#     #         final_indices.append(indices_b)
            
#     #     return torch.stack(final_indices)  # [B, target_npoint]
#     def grid_pool_downsample(self, xyz, target_npoint):
#         """
#         Args:
#             xyz: [B, N, 3] - GPU tensor
#             target_npoint: int 
#         Returns:
#             indices: [B, target_npoint]
#         """
#         B, N, _ = xyz.shape
#         device = xyz.device
        

#         if self.grid_size is None:
#             bbox = xyz.max(dim=1)[0] - xyz.min(dim=1)[0]  # [B, 3]
#             vol = bbox.prod(dim=1).mean()

#             target_npoint = torch.Tensor(target_npoint).to('cuda')
#             init_gs = max(1e-4, (vol / target_npoint) ** (1/3) * 0.15)
#             self.grid_size = float(init_gs) 
        

#         min_gs = self.grid_size * 0.01   
#         max_gs = self.grid_size * 1000.0 
        

#         best_cluster = None
#         best_gs = self.grid_size
#         best_num = 0
        
#         for _ in range(20):  
#             gs_val = float(best_gs) 
#             grid_size_tensor = torch.full((3,), gs_val, device=device, dtype=xyz.dtype)
            
#             cluster = grid_cluster(xyz.reshape(-1, 3), grid_size_tensor)  # [B*N]
#             unique_clusters, inverse = torch.unique(cluster, return_inverse=True)
#             num_clusters = unique_clusters.shape[0]
            
            
#             if abs(num_clusters - target_npoint) < abs(best_num - target_npoint):
#                 best_num = num_clusters
#                 best_gs = gs_val
#                 best_cluster = (cluster, unique_clusters, inverse)
            
    
#             if num_clusters < target_npoint * 0.995:
#                 max_gs = gs_val  
#                 best_gs = (min_gs + gs_val) / 2
#             elif num_clusters > target_npoint * 1.005:
#                 min_gs = gs_val  
#                 best_gs = (gs_val + max_gs) / 2
#             else:
#                 break  
        

#         cluster, unique_clusters, inverse = best_cluster
#         num_clusters = unique_clusters.shape[0]
        

#         min_indices = torch.full_like(unique_clusters, 10**9, device=device)
#         point_indices = torch.arange(B * N, device=device)
#         min_indices.scatter_reduce_(0, inverse, point_indices, reduce='min')
        
#         representative_indices = min_indices % N
#         batch_of_reps = torch.arange(B, device=device).repeat_interleave(N)[min_indices]
        

#         final_indices = []
#         for b in range(B):
#             mask = batch_of_reps == b
#             indices_b = representative_indices[mask]
#             current_n = indices_b.shape[0]
            
#             if current_n > target_npoint:

#                 perm = torch.randperm(current_n, device=device)
#                 indices_b = indices_b[perm][:target_npoint]
#             elif current_n < target_npoint:
#                 deficit = target_npoint - current_n

#                 if deficit <= 0.3 * target_npoint:
#                     rand_indices = torch.randint(0, N, (deficit,), device=device)
#                     indices_b = torch.cat([indices_b, rand_indices], dim=0)
#                 else:
#                     print(f"[WARN] Batch {b}: Only {current_n} clusters. "
#                         f"Consider reducing target_npoint or checking data distribution.")

#                     repeat_needed = deficit // current_n + 1
#                     extra = representative_indices[mask].repeat(repeat_needed)[:deficit]
#                     indices_b = torch.cat([indices_b, extra], dim=0)
            
#             final_indices.append(indices_b)
        
#         return torch.stack(final_indices)  # [B, target_npoint]

#     def forward(self, xyz, points, self_define_samples=None):
#         """
#         Input:
#             xyz: [B, C, N] -> 转换为 [B, N, 3]
#             points: [B, D, N] -> 转换为 [B, N, D]
#         Return:
#             new_xyz: [B, C, S]
#             new_points_concat: [B, D', S]
#         """
#         xyz = xyz.permute(0, 2, 1)  # [B, N, 3]
#         if points is not None:
#             points = points.permute(0, 2, 1)  # [B, N, D]
        
#         B, N, C = xyz.shape
#         S = self_define_samples if self_define_samples else self.npoint
        
#         fps_idx = self.grid_pool_downsample(xyz, S)  # [B, S]
#         new_xyz = index_points(xyz, fps_idx)  # [B, S, 3]

#         new_points_list = []
#         for i, radius in enumerate(self.radius_list):
#             K = self.nsample_list[i]

#             group_idx = query_ball_point_knn_aligned(radius, K, xyz, new_xyz)  # [B, S, K]
#             grouped_xyz = index_points(xyz, group_idx)  # [B, S, K, 3]
#             grouped_xyz -= new_xyz.view(B, S, 1, C)
            
#             if points is not None:
#                 grouped_points = index_points(points, group_idx)  # [B, S, K, D]
#                 grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B, S, K, D+3]
#             else:
#                 grouped_points = grouped_xyz
            
#             grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D+3, K, S]
#             for j in range(len(self.conv_blocks[i])):
#                 conv = self.conv_blocks[i][j]
#                 bn = self.bn_blocks[i][j]
#                 grouped_points = F.leaky_relu(bn(conv(grouped_points)))
#             new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
#             new_points_list.append(new_points)
        
#         new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, S]
#         new_points_concat = torch.cat(new_points_list, dim=1)  # [B, D'_total, S]
#         return new_xyz, new_points_concat
class PointNetSetAbstractionMsg_GridPool_PTv1(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, 
                 grid_size=None, pt_out_channel=None, pt_nsample=8):
        """
        Args:
            npoint: 下采样后的中心点数
            radius_list: 每个尺度的半径
            nsample_list: 每个尺度的 KNN 邻居数
            in_channel: 输入特征通道
            mlp_list: 每个尺度的 MLP 配置 [[32,32,64], [64,64,128], ...]
            pt_out_channel: PTv1 输出通道（如果为 None，则跳过 PTv1）
            pt_nsample: PTv1 注意力层的邻居数（用于 center points 之间的 attention）
        """
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.grid_size = grid_size
        self.pt_out_channel = pt_out_channel
        self.pt_nsample = pt_nsample

        # 1️⃣ 简化的 MLP（每个尺度独立，仅 1-2 层 Conv）
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3  # 特征 + 相对坐标
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

        # 2️⃣ PTv1 Attention Layer（在 Max Pooling 之后，作用于中心点）
        if pt_out_channel is not None:
            # print("arive here")
            total_in_channel = sum([mlp_list[i][-1] for i in range(len(mlp_list))])
            self.pt_layer = PointTransformerLayerForSA(
                in_planes=total_in_channel,
                out_planes=pt_out_channel,
                nsample=pt_nsample
            )
        else:
            self.pt_layer = None

    def grid_pool_downsample(self, xyz, target_npoint):
        """
        Args:
            xyz: [B, N, 3] - GPU tensor
            target_npoint: int 
        Returns:
            indices: [B, target_npoint]
        """
        B, N, _ = xyz.shape
        device = xyz.device
        

        if self.grid_size is None:
            bbox = xyz.max(dim=1)[0] - xyz.min(dim=1)[0]  # [B, 3]
            vol = bbox.prod(dim=1).mean()

            target_npoint = torch.Tensor(target_npoint).to('cuda')
            init_gs = max(1e-4, (vol / target_npoint) ** (1/3) * 0.15)
            self.grid_size = float(init_gs) 
        

        min_gs = self.grid_size * 0.01   
        max_gs = self.grid_size * 1000.0 
        

        best_cluster = None
        best_gs = self.grid_size
        best_num = 0
        
        for _ in range(20):  
            gs_val = float(best_gs) 
            grid_size_tensor = torch.full((3,), gs_val, device=device, dtype=xyz.dtype)
            
            cluster = grid_cluster(xyz.reshape(-1, 3), grid_size_tensor)  # [B*N]
            unique_clusters, inverse = torch.unique(cluster, return_inverse=True)
            num_clusters = unique_clusters.shape[0]
            
            
            if abs(num_clusters - target_npoint) < abs(best_num - target_npoint):
                best_num = num_clusters
                best_gs = gs_val
                best_cluster = (cluster, unique_clusters, inverse)
            
    
            if num_clusters < target_npoint * 0.995:
                max_gs = gs_val  
                best_gs = (min_gs + gs_val) / 2
            elif num_clusters > target_npoint * 1.005:
                min_gs = gs_val  
                best_gs = (gs_val + max_gs) / 2
            else:
                break  
        

        cluster, unique_clusters, inverse = best_cluster
        num_clusters = unique_clusters.shape[0]
        

        min_indices = torch.full_like(unique_clusters, 10**9, device=device)
        point_indices = torch.arange(B * N, device=device)
        min_indices.scatter_reduce_(0, inverse, point_indices, reduce='min')
        
        representative_indices = min_indices % N
        batch_of_reps = torch.arange(B, device=device).repeat_interleave(N)[min_indices]
        

        final_indices = []
        for b in range(B):
            mask = batch_of_reps == b
            indices_b = representative_indices[mask]
            current_n = indices_b.shape[0]
            
            if current_n > target_npoint:

                perm = torch.randperm(current_n, device=device)
                indices_b = indices_b[perm][:target_npoint]
            elif current_n < target_npoint:
                deficit = target_npoint - current_n

                if deficit <= 0.3 * target_npoint:
                    rand_indices = torch.randint(0, N, (deficit,), device=device)
                    indices_b = torch.cat([indices_b, rand_indices], dim=0)
                else:
                    print(f"[WARN] Batch {b}: Only {current_n} clusters. "
                        f"Consider reducing target_npoint or checking data distribution.")

                    repeat_needed = deficit // current_n + 1
                    extra = representative_indices[mask].repeat(repeat_needed)[:deficit]
                    indices_b = torch.cat([indices_b, extra], dim=0)
            
            final_indices.append(indices_b)
        
        return torch.stack(final_indices)  # [B, target_npoint]

    def forward(self, xyz, points, self_define_samples=None):
        """
        Input:
            xyz: [B, 3, N]
            points: [B, D, N]
        Return:
            new_xyz: [B, 3, S]
            new_points: [B, pt_out_channel, S] 或 [B, total_mlp_out, S]
        """
        # 1️⃣ 转置为 [B, N, 3] / [B, N, D]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        
        B, N, C = xyz.shape
        S = self_define_samples if self_define_samples else self.npoint
        
        # 2️⃣ Grid Pool 下采样
        fps_idx = self.grid_pool_downsample(xyz, S)
        new_xyz = index_points(xyz, fps_idx)  # [B, S, 3]

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]

            # 3️⃣ KNN 分组
            group_idx = query_ball_point_knn_aligned(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)  # [B, S, K, 3]
            grouped_xyz_rel = grouped_xyz - new_xyz.view(B, S, 1, 3)
            
            # 4️⃣ 特征分组 + 拼接相对坐标
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_features = torch.cat([grouped_points, grouped_xyz_rel], dim=-1)
            else:
                grouped_features = grouped_xyz_rel
            
            # 5️⃣ 简化的 MLP 处理
            grouped_features = grouped_features.permute(0, 3, 2, 1)  # [B, D+3, K, S]
            for j, conv in enumerate(self.conv_blocks[i]):
                bn = self.bn_blocks[i][j]
                grouped_features = F.leaky_relu(bn(conv(grouped_features)))
            
            # 6️⃣ Local Max Pooling over K
            new_points = torch.max(grouped_features, dim=2)[0]  # [B, mlp_out, S]
            new_points_list.append(new_points)
        
        # 7️⃣ 多尺度特征拼接
        new_points_concat = torch.cat(new_points_list, dim=1)  # [B, total_mlp_out, S]
        
        # 8️⃣ 应用 PTv1 Attention（在中心点之间）
        if self.pt_layer is not None:
            # 转置为 [B, S, total_mlp_out]
            new_points_concat = new_points_concat.permute(0, 2, 1)
            new_xyz_for_attn = new_xyz  # [B, S, 3]
            
            # PTv1 Attention
            new_points_concat = self.pt_layer(new_xyz_for_attn, new_points_concat)
            # 输出: [B, S, pt_out_channel]
            
            # 转回 [B, pt_out_channel, S]
            new_points_concat = new_points_concat.permute(0, 2, 1)
        
        # 9️⃣ 输出
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, S]
        return new_xyz, new_points_concat
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
#     计算两点云之间的平方距离
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
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
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
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.leaky_relu(bn(conv(new_points)))
            
        return new_points