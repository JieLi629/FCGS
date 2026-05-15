import torch
import os
import numpy as np
import torch.nn as nn
from time import time
import torch.nn.functional as F
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
    def __init__(self, in_planes, planes, share_planes=8, nsample=4):
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

        