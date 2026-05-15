import torch
import os
import numpy as np
import torch.nn as nn
from torch import Tensor
from time import time
import torch.nn.functional as F
from model.grid_utils import _grid_creater,_grid_encoder
from torch_cluster import grid_cluster,knn
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_add

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

def queryandgroup(nsample, xyz, new_xyz, feat, idx, use_xyz=True):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
    output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        B, N, _ = xyz.shape
        _, S, _ = new_xyz.shape
        device = xyz.device
        batch_x = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(N)
        batch_y = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(S)

        knn_col = knn(
            x=xyz.reshape(-1, 3),
            y=new_xyz.reshape(-1, 3),
            k=nsample,
            batch_x=batch_x,
            batch_y=batch_y
        )[1]

        if knn_col.numel() != B * S * nsample and use_xyz:
            return None, None
        if knn_col.numel() != B * S * nsample:
            return None
        
        idx = knn_col.view(B, S, nsample)

    else:
        idx = idx
    
    grouped_xyz = index_points(xyz, idx)                 # [B,S,K,3]
    grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)     # [B,S,K,3]
    grouped_feat = index_points(feat, idx)
    # n, m, c = xyz.shape[1], new_xyz.shape[1], feat.shape[2]
    # grouped_xyz = xyz[:, idx.view(-1).long(), :].view(B,m, nsample, 3) # (b,m, nsample, 3)
    # #grouped_xyz = grouping(xyz, idx) # (b,m, nsample, 3)
    # grouped_xyz -= new_xyz.unsqueeze(2) # (b,m, nsample, 3)
    # grouped_feat = feat[:, idx.view(-1).long(), :].view(B, m, nsample, c) # (b,m, nsample, 3)
    # #grouped_feat = grouping(feat, idx) # (b,m, nsample, 3)

    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1), idx # (m, nsample, 3+c)
    else:
        return grouped_feat
def get_resolution_list(resolution_list:list,dim:int = 3):
        
        offset_list = [0]
        offset=0
        for i in range(0,len(resolution_list)):
            offset += resolution_list[i]**dim
            offset_list.append(offset)
        offset_list = torch.tensor(offset_list,device='cuda',dtype=torch.int)
        resolution_list = torch.tensor(resolution_list,device='cuda',dtype=torch.int)

        return resolution_list,offset_list
@torch.no_grad()
def build_static_gs_mask(g_geo, g_feac, keep_ratio=0.3, use_feat_energy=True):
    """
    g_geo:  [1, N, 8]  -> opacity, scaling(3), rotation(4)
    g_feac: [1, N, 48] -> dc(3), rest(45)

    return:
        mask_bool: [N], True 
    """

    geo = g_geo[0]
    feac = g_feac[0]

    N = geo.shape[0]

    raw_opacity = geo[:, 0]
    raw_scaling = geo[:, 1:4]

    opacity = torch.sigmoid(raw_opacity)
    scale = torch.exp(raw_scaling)
    scale = torch.clamp(scale, max=torch.quantile(scale.detach(), 0.99))

    sx, sy, sz = scale[:, 0], scale[:, 1], scale[:, 2]
    area = sx * sy + sx * sz + sy * sz
    area_score = torch.log1p(area)

    score = opacity * area_score

    if use_feat_energy:
        dc_energy = feac[:, :3].abs().mean(dim=-1)
        sh_energy = feac[:, 3:].abs().mean(dim=-1)

        feat_energy = dc_energy + 0.1 * sh_energy
        feat_energy = feat_energy / (feat_energy.mean() + 1e-6)

        beta = 0.2
        score = score * (1.0 + beta * feat_energy)

    k = int(N * keep_ratio)
    k = max(1, min(k, N - 1))

    topk_idx = torch.topk(score, k=k, largest=False).indices

    mask_bool = torch.zeros(N, dtype=torch.bool, device=g_geo.device)
    mask_bool[topk_idx] = True

    return mask_bool
class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.LayerNorm(3), nn.LeakyReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.LayerNorm(mid_planes), nn.LeakyReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.LayerNorm(mid_planes // share_planes), nn.LeakyReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, all_feats) -> torch.Tensor:
        position, feats = all_feats  # (n, 3), (n, c), (b)
        b, _, _ = position.shape
        feats_q, feats_k, feats_v = self.linear_q(feats), self.linear_k(feats), self.linear_v(feats)  # (b,n,c)
        feats_k, idx = queryandgroup(self.nsample, position, position, feats_k, None, use_xyz=True)  # (b,m, nsample, 3+c)
        feats_v = queryandgroup(self.nsample, position, position, feats_v, idx, use_xyz=False)  # (b,m, nsample, c)
        if feats_k == None or feats_v == None:
            return None
        p_r, feats_k = feats_k[:, :, :, 0:3], feats_k[:, :, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r)    # (b,m, nsample, c)
        w = feats_k - feats_q.unsqueeze(2) + p_r.view(b, p_r.shape[1], p_r.shape[2], self.out_planes // self.mid_planes, self.mid_planes).sum(3)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        b, n, nsample, c = feats_v.shape; s = self.share_planes
        x = ((feats_v + p_r).view(b, n, nsample, s, c // s) * w.unsqueeze(3)).sum(2).view(b, n, c)
        return x

class PointTransformerBlock(nn.Module):
    def __init__(self, in_planes, planes, share_planes=8, nsample=8):
        self.expansion = 1
        super(PointTransformerBlock, self).__init__()
        self.linear_identity = nn.Linear(in_planes, planes)
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.ln1 = nn.LayerNorm(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.ln2 = nn.LayerNorm(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.ln3 = nn.LayerNorm(planes * self.expansion)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, all_feats):
        position, feats = all_feats  # (n, 3), (n, c), (b)
        identity = self.linear_identity(feats)
        feats = self.relu(self.ln1(self.linear1(feats)))
        feats = self.transformer2([position, feats])
        if feats == None:
            return None
        feats = self.relu(self.ln2(feats))
        feats = self.ln3(self.linear3(feats))
        feats += identity
        feats = self.relu(feats)
        return [position, feats]

class Encoder(nn.Module):
    def __init__(self, in_channel, hidden_channel, layer_counts):
        super().__init__()
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.enc = self._enc_block(PointTransformerBlock,layer_counts)
    
    def _enc_block(self,PTblock,layer_counts):
        layers = []
        for _ in range(layer_counts):
            layers.append(PTblock(self.in_channel, self.hidden_channel))
        
        return nn.Sequential(*layers)
    
    def forward(self,all_feats):
        position, feats = all_feats
        latent = self.enc([position, feats])
        if latent == None:
            return None
        return latent
    
class Decoder(nn.Module):
    def __init__(self, hidden_channel, out_channel, layer_counts):
        super().__init__()
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.dec = self._dec_block(PointTransformerBlock,layer_counts)
    
    def _dec_block(self,PTblock,layer_counts):
        layers = []
        for _ in range(0,layer_counts):
            layers.append(PTblock(self.hidden_channel, self.out_channel))
        
        return nn.Sequential(*layers)
    
    def forward(self,latent):
        position, latent_feats = latent
        dec_feats = self.dec([position, latent_feats])

        return dec_feats
    
class PostProcess(nn.Module):
    def __init__(self,in_channel, hidden_channel, dc_outcn, ac_outcn):
        super().__init__()
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.dc_outcn = dc_outcn
        self.ac_outcn = ac_outcn

        self.feac_dc_dec = nn.Sequential(
            nn.Linear(self.in_channel,self.hidden_channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_channel,self.dc_outcn)
        )
        self.feac_ac_dec = nn.Sequential(
            nn.Linear(self.in_channel,self.hidden_channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_channel,self.ac_outcn)
        )
    
    def forward(self, dec_feats):

        position , feats = dec_feats

        feac_dc_dec = self.feac_dc_dec(feats)
        feac_ac_dec = self.feac_ac_dec(feats)

        return [position, feac_dc_dec, feac_ac_dec]

class Sensity_Mask(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_channel = in_dim
        self.out_channel = out_dim
        self.mask_enc = nn.Sequential(
            nn.Linear(self.in_channel,self.out_channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.out_channel,1),
            nn.Sigmoid()
        )

    def forward(self,geo):

        return self.mask_enc(geo)

class hyper_encoder_m1(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_channel = in_dim
        self.hidden_channel = hidden_dim
        self.hyper_enc = nn.Sequential(
            nn.Linear(self.in_channel,self.in_channel // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.in_channel // 2,self.in_channel // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.in_channel // 2,self.hidden_channel)
        )
    def forward(self, latent):
        
        return self.hyper_enc(latent)

class hyper_decoder_m1(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.hidden_channel = hidden_dim
        self.out_channel = out_dim
        self.hyper_dec = nn.Sequential(
            nn.Linear(self.hidden_channel,self.hidden_channel * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_channel * 2,self.hidden_channel * 4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_channel * 4,self.out_channel)
        )
    def forward(self, latent):
        
        return self.hyper_dec(latent)

class hyper_encoder_m0(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_channel = in_dim
        self.hidden_channel = hidden_dim
        self.hyper_enc = nn.Sequential(
            nn.Linear(self.in_channel,self.in_channel // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.in_channel // 2,self.in_channel // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.in_channel // 2,self.hidden_channel)
        )
    def forward(self, latent):
        
        return self.hyper_enc(latent)

class hyper_decoder_m0(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.hidden_channel = hidden_dim
        self.out_channel = out_dim
        self.hyper_dec = nn.Sequential(
            nn.Linear(self.hidden_channel,self.hidden_channel * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_channel * 2,self.hidden_channel * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_channel * 2,self.out_channel)
        )
    def forward(self, latent):
        
        return self.hyper_dec(latent)

class hyper_encoder_geo(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_channel = in_dim
        self.hidden_channel = hidden_dim
        self.hyper_enc = nn.Sequential(
            nn.Linear(self.in_channel,self.in_channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.in_channel,self.in_channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.in_channel,self.hidden_channel)
        )
    def forward(self, latent):
        
        return self.hyper_enc(latent)

class hyper_decoder_geo(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.hidden_channel = hidden_dim
        self.out_channel = out_dim
        self.hyper_dec = nn.Sequential(
            nn.Linear(self.hidden_channel,self.hidden_channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_channel,self.hidden_channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_channel,self.out_channel)
        )
    def forward(self, latent):
        
        return self.hyper_dec(latent)

class Spatial_Cxt_PostProcess(nn.Module):
    def __init__(self,input_dim,fea_dim,factor):
        super().__init__()
        self.input_dim = input_dim
        self.Cxt_Post = nn.Sequential(
            nn.Linear(self.input_dim,fea_dim*factor),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fea_dim*factor,fea_dim*factor),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fea_dim*factor,fea_dim*3)
        )
    def forward(self,fea,**kwargs):
        out = self.Cxt_Post(fea)

        return out

class Spatial_Context_Model(nn.Module):
    def __init__(self, reso_3D, off_3D, reso_2D, off_2D):
        super().__init__()
        self.reso_3D = reso_3D
        self.off_3D = off_3D
        self.reso_2D = reso_2D
        self.off_2D = off_2D
    def forward(self, xyz_for_creater, xyz_for_interp, feature, determ=False, return_all=False):
        assert xyz_for_creater.shape[0] == feature.shape[0]
        grid_3D = _grid_creater.apply(xyz_for_creater, feature, self.reso_3D, self.off_3D, determ)  # [offsets_list_3D[-1], 48]
        grid_xy = _grid_creater.apply(xyz_for_creater[:, 0:2], feature, self.reso_2D, self.off_2D, determ)  # [offsets_list[-1], 48]
        grid_xz = _grid_creater.apply(xyz_for_creater[:, 0::2], feature, self.reso_2D, self.off_2D, determ)  # [offsets_list[-1], 48]
        grid_yz = _grid_creater.apply(xyz_for_creater[:, 1:3], feature, self.reso_2D, self.off_2D, determ)  # [offsets_list[-1], 48]
        # print(grid_3D)
        context_info_3D = _grid_encoder.apply(xyz_for_interp, grid_3D, self.off_3D, self.reso_3D)  # [N_choose, 48*n_levels]
        context_info_xy = _grid_encoder.apply(xyz_for_interp[:, 0:2], grid_xy, self.off_2D, self.reso_2D)  # [N_choose, 48*n_levels]
        context_info_xz = _grid_encoder.apply(xyz_for_interp[:, 0::2], grid_xz, self.off_2D, self.reso_2D)  # [N_choose, 48*n_levels]
        context_info_yz = _grid_encoder.apply(xyz_for_interp[:, 1:3], grid_yz, self.off_2D, self.reso_2D)  # [N_choose, 48*n_levels]
        #.....
        context_info = torch.cat([context_info_3D, context_info_xy, context_info_xz, context_info_yz], dim=-1)  # [N_choose, 48*n_levels*4]
        if return_all:
            return context_info, (xyz_for_creater, xyz_for_interp, feature, grid_3D, grid_xy, grid_xz, grid_yz, context_info_3D, context_info_xy, context_info_xz, context_info_yz, self.reso_3D, self.off_3D)
        return context_info

class Channel_Context_Model_M0(nn.Module):
    def __init__(self,latent_dim):#laten_dim=48
        super().__init__()
        self.mean_channel_1 = nn.Parameter(torch.zeros(size=[1,latent_dim//3]))
        self.scale_channel_1 = nn.Parameter(torch.zeros(size=[1,latent_dim//3]))
        self.weight_channel_1  = nn.Parameter(torch.zeros(size=[1,latent_dim//3]))
        # can the three block share parameters?
        self.channel_ctx_mlp_1 = nn.Sequential(
            nn.Linear(latent_dim//3,latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim,latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim,latent_dim)
        )
        self.channel_ctx_mlp_2 = nn.Sequential(
            nn.Linear(latent_dim*2//3,latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim,latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim,latent_dim)
        )

    def forward(self,fea_lat_m1:Tensor):
        # num = fea_lat_m1.shape[0]
        # fea_d1,fea_d2,fea_d3 = torch.split(fea_lat_m1,split_size_or_sections=[16,16,16],dim=-1)
        # mean_channel_1,scale_channel_1,weight_channel_1 = self.mean_channel_1.repeat(num,1),self.scale_channel_1.repeat(num,1),self.weight_channel_1.repeat(num,1)#[16,16,16]
        # mean_channel_2,scale_channel_2,weight_channel_2 = torch.split(self.channel_ctx_mlp_1(fea_d1),split_size_or_sections=[16,16,16],dim=-1)#[16,16,16]
        # mean_channel_3,scale_channel_3,weight_channel_3 = torch.split(self.channel_ctx_mlp_2(torch.cat([fea_d1,fea_d2],dim=-1)),split_size_or_sections=[16,16,16],dim=-1)#[16,16,16]

        # channel_ctx_mean = torch.cat([mean_channel_1,mean_channel_2,mean_channel_3],dim=-1)# [N,48]
        # channel_ctx_scale = torch.cat([scale_channel_1,scale_channel_2,scale_channel_3],dim=-1)# [N,48]
        # channel_ctx_weight = torch.cat([weight_channel_1,weight_channel_2,weight_channel_3],dim=-1)# [N,48]

        # return channel_ctx_mean,channel_ctx_scale,channel_ctx_weight
        num = fea_lat_m1.shape[0]
        fea_lat_m1 = fea_lat_m1.view(num, 16, 3)
        fea_d1, fea_d2, fea_d3 = fea_lat_m1[..., 0], fea_lat_m1[..., 1], fea_lat_m1[..., 2]  # [N, 16], [N, 16], [N, 16]
        mean_channel_1,scale_channel_1,weight_channel_1 = self.mean_channel_1.repeat(num,1),self.scale_channel_1.repeat(num,1),self.weight_channel_1.repeat(num,1)#[16,16,16]
        mean_channel_2,scale_channel_2,weight_channel_2 = torch.split(self.channel_ctx_mlp_1(fea_d1),split_size_or_sections=[16,16,16],dim=-1)#[16,16,16]
        mean_channel_3,scale_channel_3,weight_channel_3 = torch.split(self.channel_ctx_mlp_2(torch.cat([fea_d1,fea_d2],dim=-1)),split_size_or_sections=[16,16,16],dim=-1)#[16,16,16]

        channel_ctx_mean = torch.stack([mean_channel_1, mean_channel_2, mean_channel_3], dim=-1).view(num, -1)  # [N, 16, 3] ->[N, 48]
        channel_ctx_scale = torch.stack([scale_channel_1, scale_channel_2, scale_channel_3], dim=-1).view(num, -1)  # [N, 16, 3] ->[N, 48]
        channel_ctx_weight = torch.stack([weight_channel_1, weight_channel_2, weight_channel_3], dim=-1).view(num, -1)  # [N, 16, 3] ->[N, 48]
        # channel_ctx_mean = torch.cat([mean_channel_1,mean_channel_1,mean_channel_1],dim=-1)# [N,48]
        # channel_ctx_scale = torch.cat([scale_channel_1,scale_channel_2,scale_channel_3],dim=-1)# [N,48]
        # channel_ctx_weight = torch.cat([weight_channel_1,weight_channel_2,weight_channel_3],dim=-1)# [N,48]

        return channel_ctx_mean,channel_ctx_scale,channel_ctx_weight

class Channel_Context_Model_M1(nn.Module):
    def __init__(self,latent_dim):# laten_dim=256
        super().__init__()
        self.mean_channel_1 = nn.Parameter(torch.zeros(size=[1,latent_dim//4]))
        self.scale_channel_1 = nn.Parameter(torch.zeros(size=[1,latent_dim//4]))
        self.weight_channel_1  = nn.Parameter(torch.zeros(size=[1,latent_dim//4]))
        
        # can the three block share parameters?
        self.channel_ctx_mlp_1 = nn.Sequential(
            nn.Linear(latent_dim//4,latent_dim*3//4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim*3//4,latent_dim*3//4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim*3//4,latent_dim*3//4)
        )
        self.channel_ctx_mlp_2 = nn.Sequential(
            nn.Linear(latent_dim*2//4,latent_dim*3//4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim*3//4,latent_dim*3//4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim*3//4,latent_dim*3//4)
        )
        self.channel_ctx_mlp_3 = nn.Sequential(
            nn.Linear(latent_dim*3//4,latent_dim*3//4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim*3//4,latent_dim*3//4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim*3//4,latent_dim*3//4)
        )

    def forward(self,fea_lat_m1:Tensor):
        num = fea_lat_m1.shape[0]
        fea_d1,fea_d2,fea_d3,fea_d4 = torch.split(fea_lat_m1,split_size_or_sections=[32,32,32,32],dim=-1)
        max, min = torch.max(fea_d1), torch.min(fea_d1)
        mean_channel_1,scale_channel_1,weight_channel_1 = self.mean_channel_1.repeat(num,1),self.scale_channel_1.repeat(num,1),self.weight_channel_1.repeat(num,1)#[64,64,64]
        mean_channel_2,scale_channel_2,weight_channel_2 = torch.split(self.channel_ctx_mlp_1(fea_d1),split_size_or_sections=[32,32,32],dim=-1)#[64,64,64]
        mean_channel_3,scale_channel_3,weight_channel_3 = torch.split(self.channel_ctx_mlp_2(torch.cat([fea_d1,fea_d2],dim=-1)),split_size_or_sections=[32,32,32],dim=-1)#[64,64,64]
        mean_channel_4,scale_channel_4,weight_channel_4 = torch.split(self.channel_ctx_mlp_3(torch.cat([fea_d1,fea_d2,fea_d3],dim=-1)),split_size_or_sections=[32,32,32],dim=-1)#[64,64,64]

        channel_ctx_mean = torch.cat([mean_channel_1,mean_channel_2,mean_channel_3,mean_channel_4],dim=-1)# [N,256]
        channel_ctx_scale = torch.cat([scale_channel_1,scale_channel_2,scale_channel_3,scale_channel_4],dim=-1)# [N,256]
        channel_ctx_weight = torch.cat([weight_channel_1,weight_channel_2,weight_channel_3,weight_channel_4],dim=-1)# [N,256]

        return channel_ctx_mean,channel_ctx_scale,channel_ctx_weight
def normalize_xyz(xyz_orig, K=3, means=None, stds=None):
    if means == None:
        xyz_orig = xyz_orig.detach()# detach calculate map
        means = torch.mean(xyz_orig, dim=0, keepdim=True)# means of xyz dimension
        stds = torch.std(xyz_orig, dim=0, keepdim=True)# std of xyz dimension

    lower_bound = means - K * stds# 3sigma principle -
    upper_bound = means + K * stds# 3sigma principle +

    norm_xyz = (xyz_orig - lower_bound) / (upper_bound - lower_bound)# min-max normalization
    norm_xyz_clamp = torch.clamp(norm_xyz, min=0, max=1)# constrain the data of norm_xyz to [0,1], because there have some points out of [miu-3sigma,miu]
    mask_xyz = torch.all((norm_xyz == norm_xyz_clamp) + 0.0, dim=1) + 0.0
    # mask_xyz: [xyz_orig.shape[0]]
    return norm_xyz, norm_xyz_clamp, mask_xyz
    