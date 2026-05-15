import os 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers.pointnetpp_utils import PointNetSetAbstractionMsg_GridPool, PointNetFeaturePropagationFast
from layers.pointnetpp_utils_mask import Encoder, Decoder, PostProcess, Sensity_Mask
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_renderer import GaussianModel
# from gaussian_renderer import render
from utils.sh_utils import eval_sh
import torchvision.transforms as transforms

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
    def __init__(self):
        super(get_model, self).__init__()

        self.Q_feat = 0.001
        self.Q_latent = nn.Parameter(torch.tensor(data=1.0))
        self.Sensity_mask = Sensity_Mask(4,4)
        self.Encoder_Transform = Encoder(56, 128, 1)
        self.Decoder_Transform = Decoder(128, 128, 1)
        self.Post_feac_dec = PostProcess(128, 128, 3, 45)

    def forward(self, N_Gaussians,g_coords_feac_fusion):

        attribute = g_coords_feac_fusion[:,:,3:]
        sen_geo = g_coords_feac_fusion[:,:,3:7]
        coords = g_coords_feac_fusion[:,:,:3]

        mask_float = self.Sensity_mask(sen_geo)
        mask_bool = (mask_float > 0.5).to(torch.bool).squeeze(-1)
        mask_grad = ((mask_float > 0.5).float() - mask_float).detach() + mask_float
        mask_ratio = mask_grad.mean().item()
        coords_m0 = coords[mask_bool].unsqueeze(0)
        coords_m1 = coords[torch.logical_not(mask_bool)].unsqueeze(0)
        fea_m0 = attribute[:,:,8:][mask_bool].unsqueeze(0)
        fea_m1 = attribute[torch.logical_not(mask_bool)].unsqueeze(0)
        # Q = self.Q_feat * (1 + torch.tanh(self.Q_latent))
        latent_feat = self.Encoder_Transform([coords_m1,fea_m1])
        if latent_feat == None:
            return None, None
        else:
            position, latent = latent_feat
        latent_quant = latent + torch.empty_like(latent).uniform_(-0.5,0.5)
        latent_dec = self.Decoder_Transform([position,latent_quant])
        # fea_m0_quant = fea_m0 + torch.empty_like(fea_m0).uniform_(-0.5,0.5) * Q
        position, feac_dc, feac_ac = self.Post_feac_dec(latent_dec)

        feac_m1_full = torch.zeros_like(attribute[:,:,8:],device=feac_dc.device,dtype=feac_dc.dtype)
        feac_m0_full = torch.zeros_like(attribute[:,:,8:],device=feac_dc.device,dtype=feac_dc.dtype)
        feac_m0_full[mask_bool] = fea_m0.to(feac_dc.dtype)
        feac_m1_full[torch.logical_not(mask_bool)] = torch.cat((feac_dc,feac_ac),dim=-1)

        feac_dec_dc,feac_dec_ac = torch.split(feac_m0_full * mask_grad + feac_m1_full * (1-mask_grad),[3,45],dim=-1)
        return [coords, feac_dec_dc, feac_dec_ac], mask_grad