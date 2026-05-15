import os 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers.pointnetpp_utils import PointNetSetAbstractionMsg_GridPool, PointNetFeaturePropagationFast
from layers.pointnetpp_utils_mask_CTX import Encoder, Decoder, PostProcess, Sensity_Mask, hyper_decoder_geo,hyper_decoder_m0,hyper_decoder_m1,hyper_encoder_geo,hyper_encoder_m0,hyper_encoder_m1,build_static_gs_mask
from layers.pointnetpp_utils_mask_CTX import Spatial_Context_Model,Channel_Context_Model_M1,Channel_Context_Model_M0,Spatial_Cxt_PostProcess, normalize_xyz,get_resolution_list
from model.grid_utils import FreqEncoder
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_renderer import GaussianModel
# from gaussian_renderer import render
from utils.sh_utils import eval_sh
import torchvision.transforms as transforms
from layers.entropy_model import Factorized_Gaussian_Model
from compressai.entropy_models import GaussianConditional

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera['FoVx'].to('cuda') * 0.5)
    tanfovy = math.tan(viewpoint_camera['FoVy'].to('cuda') * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera['image_height']),
        image_width=int(viewpoint_camera['image_width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera['world_view_transform'].to('cuda'),
        projmatrix=viewpoint_camera['full_proj_transform'].to('cuda'),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera['camera_center'].to('cuda'),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # print('1111111')  # here
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    return rendered_image
def render_train(g_xyz,g_geo,camara_list,gaussians_feac_dec,pipline,C,H,W):
    gaussians = GaussianModel(3)
    g_opacity,g_scaling,g_rotation = torch.split(g_geo,split_size_or_sections=[1,3,4],dim=-1)
    f_dc,f_rst = torch.split(gaussians_feac_dec,split_size_or_sections=[3,45],dim=-1)
    gaussians._opacity = g_opacity.view(-1,1).to('cuda')
    gaussians._scaling = g_scaling.view(-1,3).to('cuda')
    gaussians._rotation = g_rotation.view(-1,4).to('cuda')
    gaussians._features_dc = f_dc.view(-1,1,3).to('cuda').requires_grad_(True)
    gaussians._features_rest = f_rst.view(-1,15,3).to('cuda').requires_grad_(True)
    gaussians._xyz = g_xyz[0,:,:].to('cuda')

    render_image_list = []
    for camara in camara_list:
        render_image = render(camara,gaussians,pipe=pipline,bg_color=torch.tensor([0,0,0],dtype=torch.float32,device='cuda'))
        render_image_list.append(render_image)
    render_image_train = torch.stack(render_image_list).view(-1,C,H,W)

    return render_image_train

class get_model(nn.Module):
    def __init__(self,
                 args,
                 resolution_3D_Grid=[70,80,90],
                 resolution_2D_Grid=[300,400,500],
                 split_ratio = [1/6,1/3,2/3,1]):
        super(get_model, self).__init__()
        self.training = True if args.stage == 'train' else False
        self.Q = 1
        self.norm = 3
        self.Ns = 4
        self.Nc_m1 = 4
        self.Nc_m0 = 3
        self.split_ratio = torch.Tensor(split_ratio)
        self.Q_feac_m0 = 0.001
        self.Q_op = 0.001
        self.Q_sc = 0.01
        self.Q_ro = 0.00001
        self.ad_fe = nn.Parameter(torch.tensor(data=[1.0, 0.0, 0.0]).unsqueeze(0))
        self.ad_op = nn.Parameter(torch.tensor(data=[1.0, 0.0, 0.0]).unsqueeze(0))  # mul, add, tanh # [1, 3]
        self.ad_sc = nn.Parameter(torch.tensor(data=[1.0, 0.0, 0.0]).unsqueeze(0))  # mul, add, tanh # [1, 3]
        self.ad_ro = nn.Parameter(torch.tensor(data=[1.0, 0.0, 0.0]).unsqueeze(0))
        # self.Sensity_mask = Sensity_Mask(4,4)
        self.Encoder_Transform = Encoder(56, 128, 1)
        self.Decoder_Transform = Decoder(128, 128, 1)
        self.Post_feac_dec = PostProcess(128, 128, 3, 45)
        self.Encoder_hyper_feac_m1 = hyper_encoder_m1(128, 32)
        self.Decoder_hyper_feac_m1 = hyper_decoder_m1(32, 384)
        self.Encoder_hyper_feac_m0 = hyper_encoder_m0(48, 24)
        self.Decoder_hyper_feac_m0 = hyper_decoder_m0(24, 144)
        self.Encoder_hyper_geo = hyper_encoder_geo(8, 8)
        self.Decoder_hyper_geo = hyper_decoder_geo(8, 24)
        self.Factorize_model_m1 = Factorized_Gaussian_Model(32)
        self.Factorize_model_m0 = Factorized_Gaussian_Model(24)
        self.Gaussian_Entropy_Model = GaussianConditional(None)
        self.Factorize_model_geo = Factorized_Gaussian_Model(8)
        self.resolution_3D_Grid,self.offset_3D_Grid = get_resolution_list(resolution_3D_Grid,dim=3)
        self.resolution_2D_Grid,self.offset_2D_Grid = get_resolution_list(resolution_2D_Grid,dim=2)
        # context model
        self.Inter_Context_Model = Spatial_Context_Model(self.resolution_3D_Grid,self.offset_3D_Grid,self.resolution_2D_Grid,self.offset_2D_Grid)    
        self.Intra_Context_Model_M1 = Channel_Context_Model_M1(128)
        self.Intra_Context_Model_M0 = Channel_Context_Model_M0(48)
        self.Freq_Encoder = FreqEncoder(3,4)
        self.feac_inter_outdim_m1 = 48*(len(resolution_3D_Grid)*3+len(resolution_3D_Grid)) + self.Freq_Encoder.output_dim
        self.feac_inter_outdim_m0 = 48*(len(resolution_2D_Grid)*3+len(resolution_3D_Grid)) + self.Freq_Encoder.output_dim
        self.fea_geo_inter_outdim = 8*(len(resolution_2D_Grid)*3+len(resolution_3D_Grid)) + self.Freq_Encoder.output_dim
        self.feac_SpaCxt_M1 = Spatial_Cxt_PostProcess(self.feac_inter_outdim_m1,128,1)
        self.feac_SpaCxt_M0 = Spatial_Cxt_PostProcess(self.feac_inter_outdim_m0,48,12)
        self.fea_geo_SpaCxt = Spatial_Cxt_PostProcess(self.fea_geo_inter_outdim,8,12)
        self.latdim_2_griddim = nn.Sequential(nn.Linear(128,48))

    def forward(self, N_Gaussians,g_coords_feac_fusion):

        attribute = g_coords_feac_fusion[:,:,3:]
        sen_geo = g_coords_feac_fusion[:,:,3:7]
        coords = g_coords_feac_fusion[:,:,:3]
        
        norm_xyz,norm_xyz_clam,mask_xyz = normalize_xyz(coords.squeeze(0),K=self.norm,means=None,stds=None)
        norm_xyz_clam = norm_xyz_clam.unsqueeze(0)
        g_fea_geo_freq = self.Freq_Encoder(norm_xyz_clam)
        b,_,_ = attribute.shape
        g_geo = attribute[...,:8]
        g_feac = attribute[...,8:]
        # mask_float = self.Sensity_mask(sen_geo)
        mask_bool = build_static_gs_mask(g_geo,g_feac,keep_ratio=0.9,use_feat_energy=True)
        mask_bool = mask_bool.to(g_coords_feac_fusion.device).bool().unsqueeze(0)
        mask_book_logical_not = torch.logical_not(mask_bool)
        mask_grad = mask_bool.float().view(1, -1, 1)
        # mask_grad = ((mask_float > 0.5).float() - mask_float).detach() + mask_float
        mask_ratio = mask_grad.mean().item()
        coords_m0 = coords[torch.logical_not(mask_bool)].unsqueeze(0)
        coords_m1 = coords[mask_bool].unsqueeze(0)
        fea_m0 = attribute[:,:,8:][torch.logical_not(mask_bool)].unsqueeze(0)
        fea_m1 = attribute[mask_bool].unsqueeze(0)
        Q_feac_m0 = (self.Q_feac_m0 * self.ad_fe[:, 0:1] + self.ad_fe[:, 1:2]) * (1 + torch.tanh(self.ad_fe[:, 2:3]))
        Q_op = (self.Q_op * self.ad_op[:, 0:1] + self.ad_op[:, 1:2]) * (1 + torch.tanh(self.ad_op[:, 2:3]))  # [1, 1]
        Q_sc = (self.Q_sc * self.ad_sc[:, 0:1] + self.ad_sc[:, 1:2]) * (1 + torch.tanh(self.ad_sc[:, 2:3]))  # [1, 1]
        Q_ro = (self.Q_ro * self.ad_ro[:, 0:1] + self.ad_ro[:, 1:2]) * (1 + torch.tanh(self.ad_ro[:, 2:3]))  # [1, 1]

        latent_feat = self.Encoder_Transform([coords_m1,fea_m1])
        if latent_feat == None:
            return None, None, None
        else:
            position, latent = latent_feat
        latent_scale = latent / self.Q
        quantize_mode = "noise" if self.training else "dequantize"
        latent_scale_quant = self.Gaussian_Entropy_Model.quantize(latent_scale, quantize_mode)
        choose_idx = torch.rand_like(coords.squeeze(0)[:, 0]) <= 0.01
        choose_idx_m1 = choose_idx[mask_bool.squeeze(0)]
        choose_idx_m0 = choose_idx[torch.logical_not(mask_bool.squeeze(0))]
        
        feac_hyp_m1 = self.Encoder_hyper_feac_m1(latent[:,choose_idx_m1,:])
        feac_hyp_m1_q = feac_hyp_m1 + torch.empty_like(feac_hyp_m1).uniform_(-0.5,0.5)
        likelihood_feac_hyp_m1 = self.Factorize_model_m1(feac_hyp_m1_q.squeeze(0)).unsqueeze(0)
        feac_m1_hyp_logits = -torch.sum(torch.log2(likelihood_feac_hyp_m1),dim=-1).unsqueeze(-1)
        hyper_ctx_mean_m1, hyper_ctx_scale_m1 , hyper_ctx_weight_m1= torch.split(self.Decoder_hyper_feac_m1(feac_hyp_m1_q.squeeze(0)),split_size_or_sections=[128,128,128],dim=-1)

        # inter context
        g_feac_grid_m1 = self.latdim_2_griddim(latent)
        [batch1_num,batch2_num,batch3_num,batch4_num] = torch.ceil(N_Gaussians * self.split_ratio)
        batch1_num,batch2_num,batch3_num,batch4_num = int(batch1_num),int(batch2_num),int(batch3_num),int(batch4_num)
        batch_ns = [0,batch1_num,batch2_num,batch3_num,batch4_num]
        mask_m1_batch1,mask_m1_batch2,mask_m1_batch3,mask_m1_batch4 = torch.sum(mask_bool[:,0:batch1_num]),torch.sum(mask_bool[:,0:batch2_num]),torch.sum(mask_bool[:,0:batch3_num]),torch.sum(mask_bool[:,0:batch4_num])
        mask_m1_batch = [0,mask_m1_batch1,mask_m1_batch2,mask_m1_batch3,mask_m1_batch4]

        choose_m1_batch1,choose_m1_batch2,choose_m1_batch3,choose_m1_batch4 = torch.sum(choose_idx_m1[0:mask_m1_batch1]),torch.sum(choose_idx_m1[0:mask_m1_batch2]),torch.sum(choose_idx_m1[0:mask_m1_batch3]),torch.sum(choose_idx_m1[0:mask_m1_batch4])
        choose_m1_batch = [0,choose_m1_batch1,choose_m1_batch2,choose_m1_batch3,choose_m1_batch4]

        spatial_ctx_m1_2 = self.Inter_Context_Model(norm_xyz_clam[:,0:batch1_num,:][mask_bool[:,0:batch1_num]],norm_xyz_clam[:,batch1_num:batch2_num,:][mask_bool[:,batch1_num:batch2_num]][choose_idx_m1[mask_m1_batch1:mask_m1_batch2]],g_feac_grid_m1[:,0:mask_m1_batch1,:].squeeze(0),determ=False)
        spatial_ctx_m1_3 = self.Inter_Context_Model(norm_xyz_clam[:,0:batch2_num,:][mask_bool[:,0:batch2_num]],norm_xyz_clam[:,batch2_num:batch3_num,:][mask_bool[:,batch2_num:batch3_num]][choose_idx_m1[mask_m1_batch2:mask_m1_batch3]],g_feac_grid_m1[:,0:mask_m1_batch2,:].squeeze(0),determ=False)
        spatial_ctx_m1_4 = self.Inter_Context_Model(norm_xyz_clam[:,0:batch3_num,:][mask_bool[:,0:batch3_num]],norm_xyz_clam[:,batch3_num:batch4_num,:][mask_bool[:,batch3_num:batch4_num]][choose_idx_m1[mask_m1_batch3:mask_m1_batch4]],g_feac_grid_m1[:,0:mask_m1_batch3,:].squeeze(0),determ=False)
        spatial_ctx_m1_1 = torch.zeros(size=[choose_idx_m1[0:mask_m1_batch1].sum(),spatial_ctx_m1_2.shape[-1]],device=spatial_ctx_m1_2.device,dtype=spatial_ctx_m1_2.dtype)

        spatial_ctx_m1= torch.cat([torch.cat([spatial_ctx_m1_1,spatial_ctx_m1_2,spatial_ctx_m1_3,spatial_ctx_m1_4],dim=0),g_fea_geo_freq[mask_bool][choose_idx_m1]],dim=-1)

        spatial_ctx_mean_m1_1,spatial_ctx_scale_m1_1,spatial_ctx_weight_m1_1 = torch.split(self.feac_SpaCxt_M1(spatial_ctx_m1[0:choose_idx_m1[0:mask_m1_batch1].sum()]),split_size_or_sections=[128,128,128],dim=-1)
        spatial_ctx_mean_m1_2,spatial_ctx_scale_m1_2,spatial_ctx_weight_m1_2 = torch.split(self.feac_SpaCxt_M1(spatial_ctx_m1[choose_idx_m1[0:mask_m1_batch1].sum():choose_idx_m1[0:mask_m1_batch2].sum()]),split_size_or_sections=[128,128,128],dim=-1)
        spatial_ctx_mean_m1_3,spatial_ctx_scale_m1_3,spatial_ctx_weight_m1_3 = torch.split(self.feac_SpaCxt_M1(spatial_ctx_m1[choose_idx_m1[0:mask_m1_batch2].sum():choose_idx_m1[0:mask_m1_batch3].sum()]),split_size_or_sections=[128,128,128],dim=-1)
        spatial_ctx_mean_m1_4,spatial_ctx_scale_m1_4,spatial_ctx_weight_m1_4 = torch.split(self.feac_SpaCxt_M1(spatial_ctx_m1[choose_idx_m1[0:mask_m1_batch3].sum():choose_idx_m1[0:mask_m1_batch4].sum()]),split_size_or_sections=[128,128,128],dim=-1)
        spatial_ctx_mean_m1 = [spatial_ctx_mean_m1_1,spatial_ctx_mean_m1_2,spatial_ctx_mean_m1_3,spatial_ctx_mean_m1_4]
        spatial_ctx_scale_m1 = [spatial_ctx_scale_m1_1,spatial_ctx_scale_m1_2,spatial_ctx_scale_m1_3,spatial_ctx_scale_m1_4]
        spatial_ctx_weight_m1 = [spatial_ctx_weight_m1_1,spatial_ctx_weight_m1_2,spatial_ctx_weight_m1_3,spatial_ctx_weight_m1_4]
        
        channel_ctx_mean_m1,channel_ctx_scale_m1,channel_ctx_weight_m1 = self.Intra_Context_Model_M1(latent.squeeze(0)[choose_idx_m1])
        likelihood_feac_lat_m1_sp = []
        for ns in range(self.Ns):
            sp_mean_list_m1 = spatial_ctx_mean_m1[ns]# [batch_num,256]
            sp_scale_list_m1 = spatial_ctx_scale_m1[ns]# [batch_num,256]
            sp_weight_list_m1 = spatial_ctx_weight_m1[ns]# [batch_num,256]
            likelihood_feac_lat_m1_ch = []
            for nc in range(self.Nc_m1):
                sp_mean_m1, sp_scale_m1, sp_weight_m1 = sp_mean_list_m1[:,nc*32:(nc+1)*32], sp_scale_list_m1[:,nc*32:(nc+1)*32], sp_weight_list_m1[:,nc*32:(nc+1)*32]
                ch_mean_m1, ch_scale_m1, ch_weight_m1 = channel_ctx_mean_m1[choose_m1_batch[ns]:choose_m1_batch[ns+1],nc*32:(nc+1)*32],channel_ctx_scale_m1[choose_m1_batch[ns]:choose_m1_batch[ns+1],nc*32:(nc+1)*32],channel_ctx_weight_m1[choose_m1_batch[ns]:choose_m1_batch[ns+1],nc*32:(nc+1)*32]
                hper_mean_m1, hyper_scale_m1, hper_weight_m1 = hyper_ctx_mean_m1[choose_m1_batch[ns]:choose_m1_batch[ns+1],nc*32:(nc+1)*32],hyper_ctx_scale_m1[choose_m1_batch[ns]:choose_m1_batch[ns+1],nc*32:(nc+1)*32],hyper_ctx_weight_m1[choose_m1_batch[ns]:choose_m1_batch[ns+1],nc*32:(nc+1)*32]
                weights = torch.stack([sp_weight_m1,ch_weight_m1,hper_weight_m1],dim=-1)
                weights = torch.softmax(weights,dim=-1)
                sp_weight_m1, ch_weight_m1, hper_weight_m1 = weights[...,0], weights[...,1], weights[...,2]
                scale = [sp_scale_m1,ch_scale_m1,hyper_scale_m1]
                mean = [sp_mean_m1,ch_mean_m1,hper_mean_m1]
                weight = [sp_weight_m1,ch_weight_m1,hper_weight_m1]
                x = latent_scale_quant[:,mask_m1_batch[ns]:mask_m1_batch[ns+1],nc*32:(nc+1)*32].squeeze(0)[choose_idx_m1[mask_m1_batch[ns]:mask_m1_batch[ns+1]]]
                likelihood_feac_lat_m1_ctx = 0
                for i in range(3):
                    likelihood_feac_lat_m1_com = self.Gaussian_Entropy_Model._likelihood(x,scale[i]/self.Q,mean[i]/self.Q)
                    likelihood_feac_lat_m1_ctx += likelihood_feac_lat_m1_com * weight[i]
                likelihood_feac_lat_m1_ch.append(likelihood_feac_lat_m1_ctx)
            likelihood_feac_lat_m1_sp.append(torch.cat(likelihood_feac_lat_m1_ch,dim=-1))
        likelihood_feac_m1 = torch.cat(likelihood_feac_lat_m1_sp,dim=0).unsqueeze(0)
        feac_m1_logits = -torch.sum(torch.log2(likelihood_feac_m1),dim=-1).unsqueeze(-1)

        latent_dec = self.Decoder_Transform([position,latent_scale_quant])
        # fea_m0_quant = fea_m0 + torch.empty_like(fea_m0).uniform_(-0.5,0.5) * Q
        position, feac_dc, feac_ac = self.Post_feac_dec(latent_dec)

        feac_m0_scale = fea_m0 / Q_feac_m0
        quantize_mode = "noise" if self.training else "dequantize"
        feac_m0_scale_quant = self.Gaussian_Entropy_Model.quantize(feac_m0_scale, quantize_mode)
        feac_m0_Q = feac_m0_scale_quant * Q_feac_m0
        feac_m1_full = torch.zeros_like(attribute[:,:,8:],device=attribute.device,dtype=attribute.dtype)
        feac_m0_full = torch.zeros_like(attribute[:,:,8:],device=attribute.device,dtype=attribute.dtype)
        feac_m0_full[torch.logical_not(mask_bool)] = feac_m0_Q.to(attribute.dtype)
        feac_m1_full[mask_bool] = torch.cat((feac_dc,feac_ac),dim=-1)
        feac_dec_dc,feac_dec_ac = torch.split(feac_m0_full * (1-mask_grad) + feac_m1_full * mask_grad,[3,45],dim=-1)
        feac_dec = torch.cat((feac_dec_dc,feac_dec_ac),dim=-1)
        # _,likelihood_feac_m1 = self.Gaussian_Entropy_Model(latent_quant.squeeze(0),hyper_ctx_scale_m1,hyper_ctx_mean_m1)
        # likelihood_feac_m1 = likelihood_feac_m1.unsqueeze(0)
        # bits_feac_ctx_m1 = (-torch.sum(torch.log2(likelihood_feac_m1),dim=-1).unsqueeze(-1)).sum()
        # feac_m1_logits = -torch.sum(torch.log2(likelihood_feac_m1),dim=-1).unsqueeze(-1)

        # feac_m0_Q = fea_m0 + torch.empty_like(fea_m0).uniform_(-0.5,0.5) * Q_feac_m0

        feac_hyp_m0 = self.Encoder_hyper_feac_m0(fea_m0[:,choose_idx_m0,:])
        feac_hyp_m0_q = feac_hyp_m0 + torch.empty_like(feac_hyp_m0).uniform_(-0.5,0.5)
        likelihood_feac_hyp_m0 = self.Factorize_model_m0(feac_hyp_m0_q.squeeze(0)).unsqueeze(0)
        feac_m0_hyp_logits = -torch.sum(torch.log2(likelihood_feac_hyp_m0),dim=-1).unsqueeze(-1)
        hyper_ctx_mean_m0, hyper_ctx_scale_m0, hyper_ctx_weight_m0 = torch.split(self.Decoder_hyper_feac_m0(feac_hyp_m0_q.squeeze(0)),split_size_or_sections=[48,48,48],dim=-1)


        # context
        mask_m0_batch1,mask_m0_batch2,mask_m0_batch3,mask_m0_batch4 = torch.sum(mask_book_logical_not[:,0:batch1_num]),torch.sum(mask_book_logical_not[:,0:batch2_num]),torch.sum(mask_book_logical_not[:,0:batch3_num]),torch.sum(mask_book_logical_not[:,0:batch4_num])
        mask_m0_batch = [0,mask_m0_batch1,mask_m0_batch2,mask_m0_batch3,mask_m0_batch4]

        choose_m0_batch1,choose_m0_batch2,choose_m0_batch3,choose_m0_batch4 = torch.sum(choose_idx_m0[0:mask_m0_batch1]),torch.sum(choose_idx_m0[0:mask_m0_batch2]),torch.sum(choose_idx_m0[0:mask_m0_batch3]),torch.sum(choose_idx_m0[0:mask_m0_batch4])
        choose_m0_batch = [0,choose_m0_batch1,choose_m0_batch2,choose_m0_batch3,choose_m0_batch4]

        spatial_ctx_m0_2 = self.Inter_Context_Model(norm_xyz_clam[:,0:batch1_num,:].squeeze(0),norm_xyz_clam[:,batch1_num:batch2_num,:][mask_book_logical_not[:,batch1_num:batch2_num]][choose_idx_m0[mask_m0_batch1:mask_m0_batch2]],feac_dec[:,0:batch1_num,:].squeeze(0),determ=False)
        spatial_ctx_m0_3 = self.Inter_Context_Model(norm_xyz_clam[:,0:batch2_num,:].squeeze(0),norm_xyz_clam[:,batch2_num:batch3_num,:][mask_book_logical_not[:,batch2_num:batch3_num]][choose_idx_m0[mask_m0_batch2:mask_m0_batch3]],feac_dec[:,0:batch2_num,:].squeeze(0),determ=False)
        spatial_ctx_m0_4 = self.Inter_Context_Model(norm_xyz_clam[:,0:batch3_num,:].squeeze(0),norm_xyz_clam[:,batch3_num:batch4_num,:][mask_book_logical_not[:,batch3_num:batch4_num]][choose_idx_m0[mask_m0_batch3:mask_m0_batch4]],feac_dec[:,0:batch3_num,:].squeeze(0),determ=False)
        spatial_ctx_m0_1 = torch.zeros(size=[choose_idx_m0[0:mask_m0_batch1].sum(),spatial_ctx_m0_2.shape[-1]],device=spatial_ctx_m0_2.device,dtype=spatial_ctx_m0_2.dtype)
        
        spatial_ctx_m0 = torch.cat([torch.cat([spatial_ctx_m0_1,spatial_ctx_m0_2,spatial_ctx_m0_3,spatial_ctx_m0_4],dim=0),g_fea_geo_freq[mask_book_logical_not][choose_idx_m0]],dim=-1)

        spatial_ctx_mean_m0_1,spatial_ctx_scale_m0_1,spatial_ctx_weight_m0_1 = torch.split(self.feac_SpaCxt_M0(spatial_ctx_m0[0:choose_idx_m0[0:mask_m0_batch1].sum()]),split_size_or_sections=[48,48,48],dim=-1)
        spatial_ctx_mean_m0_2,spatial_ctx_scale_m0_2,spatial_ctx_weight_m0_2 = torch.split(self.feac_SpaCxt_M0(spatial_ctx_m0[choose_idx_m0[0:mask_m0_batch1].sum():choose_idx_m0[0:mask_m0_batch2].sum()]),split_size_or_sections=[48,48,48],dim=-1)
        spatial_ctx_mean_m0_3,spatial_ctx_scale_m0_3,spatial_ctx_weight_m0_3 = torch.split(self.feac_SpaCxt_M0(spatial_ctx_m0[choose_idx_m0[0:mask_m0_batch2].sum():choose_idx_m0[0:mask_m0_batch3].sum()]),split_size_or_sections=[48,48,48],dim=-1)
        spatial_ctx_mean_m0_4,spatial_ctx_scale_m0_4,spatial_ctx_weight_m0_4 = torch.split(self.feac_SpaCxt_M0(spatial_ctx_m0[choose_idx_m0[0:mask_m0_batch3].sum():choose_idx_m0[0:mask_m0_batch4].sum()]),split_size_or_sections=[48,48,48],dim=-1)

        spatial_ctx_mean_m0 = [spatial_ctx_mean_m0_1,spatial_ctx_mean_m0_2,spatial_ctx_mean_m0_3,spatial_ctx_mean_m0_4]
        spatial_ctx_scale_m0 = [spatial_ctx_scale_m0_1,spatial_ctx_scale_m0_2,spatial_ctx_scale_m0_3,spatial_ctx_scale_m0_4]
        spatial_ctx_weight_m0 = [spatial_ctx_weight_m0_1,spatial_ctx_weight_m0_2,spatial_ctx_weight_m0_3,spatial_ctx_weight_m0_4]

        channel_ctx_mean_m0,channel_ctx_scale_m0,channel_ctx_weight_m0 = self.Intra_Context_Model_M0(fea_m0.squeeze(0)[choose_idx_m0])

        likelihood_feac_lat_m0_sp = []
        for ns in range(self.Ns):
            sp_mean_list_m0 = spatial_ctx_mean_m0[ns]# [batch_num,48]
            sp_scale_list_m0 = spatial_ctx_scale_m0[ns]# [batch_num,48]
            sp_weight_list_m0 = spatial_ctx_weight_m0[ns]# [batch_num,48]
            likelihood_feac_lat_m0_ch = []
            for nc in range(self.Nc_m0):
                sp_mean_m0, sp_scale_m0, sp_weight_m0 = sp_mean_list_m0[:,nc::3], sp_scale_list_m0[:,nc::3], sp_weight_list_m0[:,nc::3]
                ch_mean_m0, ch_scale_m0, ch_weight_m0 = channel_ctx_mean_m0[choose_m0_batch[ns]:choose_m0_batch[ns+1],nc::3],channel_ctx_scale_m0[choose_m0_batch[ns]:choose_m0_batch[ns+1],nc::3],channel_ctx_weight_m0[choose_m0_batch[ns]:choose_m0_batch[ns+1],nc::3]
                hper_mean_m0, hyper_scale_m0, hper_weight_m0 = hyper_ctx_mean_m0[choose_m0_batch[ns]:choose_m0_batch[ns+1],nc::3],hyper_ctx_scale_m0[choose_m0_batch[ns]:choose_m0_batch[ns+1],nc::3],hyper_ctx_weight_m0[choose_m0_batch[ns]:choose_m0_batch[ns+1],nc::3]
                weights = torch.stack([sp_weight_m0,ch_weight_m0,hper_weight_m0],dim=-1)
                weights = torch.softmax(weights,dim=-1)
                sp_weight_m0, ch_weight_m0, hper_weight_m0 = weights[...,0], weights[...,1], weights[...,2]
                scale = [sp_scale_m0,ch_scale_m0,hyper_scale_m0]
                mean = [sp_mean_m0,ch_mean_m0,hper_mean_m0]
                weight = [sp_weight_m0,ch_weight_m0,hper_weight_m0]
                x = feac_m0_scale_quant[:,mask_m0_batch[ns]:mask_m0_batch[ns+1],nc::3].squeeze(0)[choose_idx_m0[mask_m0_batch[ns]:mask_m0_batch[ns+1]]]
                likelihood_feac_lat_m0_ctx = 0
                # Q_feac = Q_feac_0[mask_fea_m0_bool][mask_m0_batch[ns]:mask_m0_batch[ns+1],nc::3]
                for i in range(3):
                    likelihood_feac_lat_m0_com = self.Gaussian_Entropy_Model._likelihood(x,scale[i]/Q_feac_m0,mean[i]/Q_feac_m0)
                    likelihood_feac_lat_m0_ctx += likelihood_feac_lat_m0_com * weight[i]
                likelihood_feac_lat_m0_ch.append(likelihood_feac_lat_m0_ctx)
            likelihood_feac_lat_m0_sp.append(torch.cat(likelihood_feac_lat_m0_ch,dim=-1))
        likelihood_feac_m0 = torch.cat(likelihood_feac_lat_m0_sp,dim=0).unsqueeze(0)
        feac_m0_logits = -torch.sum(torch.log2(likelihood_feac_m0),dim=-1).unsqueeze(-1)

        # feac_m0_res = (fea_m0 - hyper_ctx_mean_m0) / Q_feac_m0
        # feac_m0_res_Q = feac_m0_res + torch.empty_like(feac_m0_res).uniform_(-0.5,0.5)
        # _,likelihood_feac_m0 = self.Gaussian_Entropy_Model(feac_m0_res_Q.squeeze(0),hyper_ctx_scale_m0 / Q_feac_m0,None)
        # likelihood_feac_m0 = likelihood_feac_m0.unsqueeze(0)
        # feac_m0_logits = -torch.sum(torch.log2(likelihood_feac_m0),dim=-1).unsqueeze(-1)
        # feac_m0_Q = feac_m0_res_Q * Q_feac_m0 + hyper_ctx_mean_m0

        Q_op = Q_op.repeat(N_Gaussians,1)
        Q_sc = Q_sc.repeat(N_Gaussians,3)
        Q_ro = Q_ro.repeat(N_Gaussians,4)
        Q_geo = torch.cat((Q_op,Q_sc,Q_ro),dim=-1).unsqueeze(0)
        # geo_Q = geo + torch.empty_like(geo).uniform_(-0.5,0.5) * Q_geo
        geo_scale = g_geo / Q_geo
        quantize_mode = "noise" if self.training else "dequantize"
        geo_scale_quant = self.Gaussian_Entropy_Model.quantize(geo_scale, quantize_mode)
        geo_Q = geo_scale_quant * Q_geo
        geo_hyper = self.Encoder_hyper_geo(g_geo[:,choose_idx,:])
        geo_hyper_q = geo_hyper + torch.empty_like(geo_hyper).uniform_(-0.5,0.5)
        likelihood_geo_hyp = self.Factorize_model_geo(geo_hyper_q.squeeze(0)).unsqueeze(0)
        logits_geo_hyp = -torch.sum(torch.log2(likelihood_geo_hyp),dim=-1).unsqueeze(-1)
        hyper_ctx_mean_geo, hyper_ctx_scale_geo, hyper_ctx_weight_geo = torch.split(self.Decoder_hyper_geo(geo_hyper_q.squeeze(0)),split_size_or_sections=[8,8,8],dim=-1)

        choose_batch1,choose_batch2,choose_batch3,choose_batch4 = torch.sum(choose_idx[0:batch1_num]),torch.sum(choose_idx[0:batch2_num]),torch.sum(choose_idx[0:batch3_num]),torch.sum(choose_idx[0:batch4_num])
        choose_batch = [0,choose_batch1,choose_batch2,choose_batch3,choose_batch4]

        spatial_ctx_geo_2 = self.Inter_Context_Model(norm_xyz_clam[:,0:batch1_num,:].squeeze(0),norm_xyz_clam[:,batch1_num:batch2_num,:].squeeze(0)[choose_idx[batch1_num:batch2_num]],geo_Q[:,0:batch1_num,:].squeeze(0),determ=False)
        spatial_ctx_geo_3 = self.Inter_Context_Model(norm_xyz_clam[:,0:batch2_num,:].squeeze(0),norm_xyz_clam[:,batch2_num:batch3_num,:].squeeze(0)[choose_idx[batch2_num:batch3_num]],geo_Q[:,0:batch2_num,:].squeeze(0),determ=False)
        spatial_ctx_geo_4 = self.Inter_Context_Model(norm_xyz_clam[:,0:batch3_num,:].squeeze(0),norm_xyz_clam[:,batch3_num:batch4_num,:].squeeze(0)[choose_idx[batch3_num:batch4_num]],geo_Q[:,0:batch3_num,:].squeeze(0),determ=False)
        spatial_ctx_geo_1 = torch.zeros(size=[choose_idx[0:batch1_num].sum(),spatial_ctx_geo_2.shape[-1]],device=spatial_ctx_geo_2.device,dtype=spatial_ctx_geo_2.dtype)

        spatial_ctx_geo = torch.cat([torch.cat([spatial_ctx_geo_1,spatial_ctx_geo_2,spatial_ctx_geo_3,spatial_ctx_geo_4],dim=0),g_fea_geo_freq.squeeze(0)[choose_idx]],dim=-1)

        spatial_ctx_mean_geo_1,spatial_ctx_scale_geo_1,spatial_ctx_weight_geo_1 = torch.split(self.fea_geo_SpaCxt(spatial_ctx_geo[0:choose_idx[0:batch1_num].sum()]),split_size_or_sections=[8,8,8],dim=-1)
        spatial_ctx_mean_geo_2,spatial_ctx_scale_geo_2,spatial_ctx_weight_geo_2 = torch.split(self.fea_geo_SpaCxt(spatial_ctx_geo[choose_idx[0:batch1_num].sum():choose_idx[0:batch2_num].sum()]),split_size_or_sections=[8,8,8],dim=-1)
        spatial_ctx_mean_geo_3,spatial_ctx_scale_geo_3,spatial_ctx_weight_geo_3 = torch.split(self.fea_geo_SpaCxt(spatial_ctx_geo[choose_idx[0:batch2_num].sum():choose_idx[0:batch3_num].sum()]),split_size_or_sections=[8,8,8],dim=-1)
        spatial_ctx_mean_geo_4,spatial_ctx_scale_geo_4,spatial_ctx_weight_geo_4 = torch.split(self.fea_geo_SpaCxt(spatial_ctx_geo[choose_idx[0:batch3_num].sum():choose_idx[0:batch4_num].sum()]),split_size_or_sections=[8,8,8],dim=-1)

        spatial_ctx_mean_geo = [spatial_ctx_mean_geo_1,spatial_ctx_mean_geo_2,spatial_ctx_mean_geo_3,spatial_ctx_mean_geo_4]
        spatial_ctx_scale_geo = [spatial_ctx_scale_geo_1,spatial_ctx_scale_geo_2,spatial_ctx_scale_geo_3,spatial_ctx_scale_geo_4]
        spatial_ctx_weight_geo = [spatial_ctx_weight_geo_1,spatial_ctx_weight_geo_2,spatial_ctx_weight_geo_3,spatial_ctx_weight_geo_4]

        likelihood_geo_lat_sp = []
        for ns in range(self.Ns):
            sp_mean_geo = spatial_ctx_mean_geo[ns]# [batch_num,48]
            sp_scale_geo = spatial_ctx_scale_geo[ns]# [batch_num,48]
            sp_weight_geo = spatial_ctx_weight_geo[ns]# [batch_num,48]
                # ch_mean_m0, ch_scale_m0, ch_weight_m0 = channel_ctx_mean_m0[mask_batch[ns]:mask_batch[ns+1],nc::3],channel_ctx_scale_m0[mask_batch[ns]:mask_batch[ns+1],nc::3],channel_ctx_weight_m0[mask_batch[ns]:mask_batch[ns+1],nc::3]
            hper_mean_geo, hyper_scale_geo, hper_weight_geo = hyper_ctx_mean_geo[choose_batch[ns]:choose_batch[ns+1],:],hyper_ctx_scale_geo[choose_batch[ns]:choose_batch[ns+1],:],hyper_ctx_weight_geo[choose_batch[ns]:choose_batch[ns+1],:]
            weights = torch.stack([sp_mean_geo,hper_weight_geo],dim=-1)
            weights = torch.softmax(weights,dim=-1)
            sp_weight_geo, hper_weight_geo = weights[...,0], weights[...,1]
            scale = [sp_scale_geo,hyper_scale_geo]
            mean = [sp_mean_geo,hper_mean_geo]
            weight = [sp_weight_geo,hper_weight_geo]
            x = geo_scale_quant[:,batch_ns[ns]:batch_ns[ns+1],:].squeeze(0)[choose_idx[batch_ns[ns]:batch_ns[ns+1]]]
            Q_geo_sp = Q_geo[:,batch_ns[ns]:batch_ns[ns+1],:].squeeze(0)[choose_idx[batch_ns[ns]:batch_ns[ns+1]]]
            likelihood_geo_lat_ctx = 0
            for i in range(2):
                likelihood_geo_lat_com = self.Gaussian_Entropy_Model._likelihood(x,scale[i]/Q_geo_sp,mean[i]/Q_geo_sp)
                likelihood_geo_lat_ctx += likelihood_geo_lat_com * weight[i]
            likelihood_geo_lat_sp.append(likelihood_geo_lat_ctx)
        likelihood_geo = torch.cat(likelihood_geo_lat_sp,dim=0).unsqueeze(0)
        logits_geo = -torch.sum(torch.log2(likelihood_geo),dim=-1).unsqueeze(-1)
        
        # geo_res = (g_geo - hyper_ctx_mean_geo) / Q_geo
        # geo_res_Q = geo_res + torch.empty_like(geo_res).uniform_(-0.5,0.5)
        # _,likelihood_geo = self.Gaussian_Entropy_Model(geo_res_Q.squeeze(0),hyper_ctx_scale_geo / Q_geo,None)
        # likelihood_geo = likelihood_geo.unsqueeze(0)
        # logits_geo = -torch.sum(torch.log2(likelihood_geo),dim=-1).unsqueeze(-1)
        # geo_Q = geo_res_Q * Q_geo + hyper_ctx_mean_geo

        # latent_dec = self.Decoder_Transform([position,latent_quant])
        # # fea_m0_quant = fea_m0 + torch.empty_like(fea_m0).uniform_(-0.5,0.5) * Q
        # position, feac_dc, feac_ac = self.Post_feac_dec(latent_dec)

        # feac_m1_full = torch.zeros_like(attribute[:,:,8:],device=feac_dc.device,dtype=feac_dc.dtype)
        # feac_m0_full = torch.zeros_like(attribute[:,:,8:],device=feac_dc.device,dtype=feac_dc.dtype)

        logits_m1_full = torch.zeros(size=(b,choose_idx.sum(),1),device=feac_dc.device,dtype=feac_dc.dtype)
        logits_m0_full = torch.zeros(size=(b,choose_idx.sum(),1),device=feac_dc.device,dtype=feac_dc.dtype)
        
        logits_hyp_m1_full = torch.zeros(size=(b,choose_idx.sum(),1),device=feac_dc.device,dtype=feac_dc.dtype)
        logits_hyp_m0_full = torch.zeros(size=(b,choose_idx.sum(),1),device=feac_dc.device,dtype=feac_dc.dtype)

        # feac_m0_full[torch.logical_not(mask_bool)] = feac_m0_Q.to(feac_dc.dtype)
        # feac_m1_full[mask_bool] = torch.cat((feac_dc,feac_ac),dim=-1)
        mask_choose = mask_grad.squeeze(0)[choose_idx].unsqueeze(0)
        mask_bool_choose = mask_choose.to(g_coords_feac_fusion.device).bool().squeeze(-1)
        logits_m0_full[torch.logical_not(mask_bool_choose)] = feac_m0_logits
        logits_m1_full[mask_bool_choose] = feac_m1_logits

        logits_hyp_m0_full[torch.logical_not(mask_bool_choose)] = feac_m0_hyp_logits
        logits_hyp_m1_full[mask_bool_choose] = feac_m1_hyp_logits

        mask_choos_grad = mask_choose.float().view(1, -1, 1)
        # feac_dec_dc,feac_dec_ac = torch.split(feac_m0_full * mask_grad + feac_m1_full * (1-mask_grad),[3,45],dim=-1)
        bits_total = logits_m0_full * (1-mask_choos_grad) + logits_m1_full * mask_choos_grad + logits_geo + logits_hyp_m0_full * (1-mask_choos_grad) + logits_hyp_m1_full * mask_choos_grad + logits_geo_hyp
        bpp = torch.sum(bits_total) / mask_choose.shape[1]
        return [coords, geo_Q, feac_dec_dc, feac_dec_ac], mask_grad, bpp