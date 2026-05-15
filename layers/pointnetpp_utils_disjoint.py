import torch
import numpy as np
import torch.nn as nn
from time import time
import torch.nn.functional as F
from torch_cluster import grid_cluster,knn
from torch_scatter import scatter_max, scatter_mean, scatter_min

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

def query_ball_point_knn_aligned(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device
    
    batch_xyz = torch.arange(B, device=device).repeat_interleave(N)
    batch_new = torch.arange(B, device=device).repeat_interleave(S)
    
    idx = knn(xyz.reshape(-1, 3), new_xyz.reshape(-1, 3), 
              k=nsample, batch_x=batch_xyz, batch_y=batch_new)[1]
    if idx.shape[0] == B * S * nsample:
        idx = idx.view(B, S, nsample)  # [B, S, K]
    else:
        return None
    
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
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
            
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
        
        if best_cluster == None:
            print("Can't find a satisfied grid_size for grid_pool_sample")
            return None
        cluster, unique_clusters, inverse = best_cluster
        num_clusters = unique_clusters.shape[0]
        

        centroids = scatter_mean(xyz.reshape(-1, 3), inverse, dim=0)
        point_to_center = xyz.reshape(-1, 3) - centroids[inverse]
        distances = torch.norm(point_to_center, dim=1)
        min_dist_indices = scatter_min(distances, inverse, dim=0)[1]

        representative_indices = min_dist_indices % N
        batch_of_reps = min_dist_indices // N
        

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

                if deficit <= 0.005 * target_npoint:
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
        
        fps_idx = self.grid_pool_downsample(xyz, S)  # [B, S]
        if fps_idx == None:
            return None,None
        new_xyz = index_points(xyz, fps_idx)  # [B, S, 3]

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]

            group_idx = query_ball_point_knn_aligned(radius, K, xyz, new_xyz)  # [B, S, K]
            if group_idx == None:
                return None, None
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
                bn = self.bn_blocks[i][j]
                grouped_points = F.leaky_relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)
        
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, S]
        new_points_concat = torch.cat(new_points_list, dim=1)  # [B, D'_total, S]
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