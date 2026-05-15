import os 
import math
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.functional as F
from layers.pointnetpp_utils_disjoint import PointNetSetAbstractionMsg_GridPool, PointNetFeaturePropagationFast
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

        self.sa1 = PointNetSetAbstractionMsg_GridPool(1024, [0.05], [3], 64, [[128, 128, 128]])
        self.sa2 = PointNetSetAbstractionMsg_GridPool(256, [0.1], [3], 128, [[128, 192, 256]])
        self.sa3 = PointNetSetAbstractionMsg_GridPool(64, [0.2], [3], 256, [[256, 384, 512]])
        # self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [4, 8], 256+256, [[256, 256, 512], [256, 384, 512]])
        # self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagationFast(512+256, [512, 256])
        self.fp2 = PointNetFeaturePropagationFast(256+128, [256, 128])
        self.fp1 = PointNetFeaturePropagationFast(128+64, [128, 128])
        # self.fp3 = PointNetFeaturePropagationFast(512, [512, 256])
        # self.fp2 = PointNetFeaturePropagationFast(256, [256, 128])
        # self.fp1 = PointNetFeaturePropagationFast(128, [128, 128])
        self.fea_enc= nn.Sequential(
            nn.Conv1d(56,64,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64,64,1)
        )
        # self.conv1 = nn.Conv1d(128, 64, 1)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(64, 3, 1)
        # self.conv3 = nn.Conv1d(64, 45, 1)
        self.feac_dc_dec = nn.Sequential(
            nn.Conv1d(128,64,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64,3,1)
        )
        self.feac_ac_dec = nn.Sequential(
            nn.Conv1d(128,64,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64,45,1)
        )
        self.geo_op_dec = nn.Sequential(
            nn.Conv1d(128,64,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64,1,1)
        )
        self.geo_sc_dec = nn.Sequential(
            nn.Conv1d(128,64,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64,3,1)
        )
        self.geo_ro_dec = nn.Sequential(
            nn.Conv1d(128,64,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64,4,1)
        )

    def forward(self, N_Gaussians,xyz):
        l0_points = xyz[:,3:,:]
        l0_xyz = xyz[:,:3,:]

        raw_fea = self.fea_enc(l0_points)
        # l1_xyz, l1_points = self.sa1(l0_xyz, l0_points,N_Gaussians//4)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points,N_Gaussians//16)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points,N_Gaussians//64)
        l1_xyz, l1_points = self.sa1(l0_xyz, raw_fea,N_Gaussians//3)
        if l1_xyz == None and l1_points == None:
            return None
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points,N_Gaussians//9)
        if l2_xyz == None and l2_points == None:
            return None
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points,N_Gaussians//27)
        if l3_xyz == None and l3_points == None:
            return None
        # l1_xyz, l1_points = self.sa1(l0_xyz, l0_points,N_Gaussians//3)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points,N_Gaussians//9)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points,N_Gaussians//27)
        # l4_xyz, l4_points = self.sa4(l3_xyz, l3_points,N_Gaussians//16)
        # l1_points = l1_points + torch.empty_like(l1_points).uniform_(-0.5,0.5)
        # l2_points = l2_points + torch.empty_like(l2_points).uniform_(-0.5,0.5)
        # l3_points = l3_points + torch.empty_like(l3_points).uniform_(-0.5,0.5)
        l1_points = (torch.round(l1_points)-l1_points).detach() + l1_points
        l2_points = (torch.round(l2_points)-l2_points).detach() + l2_points
        l3_points = (torch.round(l3_points)-l3_points).detach() + l3_points
        raw_fea = (torch.round(raw_fea)-raw_fea).detach() + raw_fea
        # l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, raw_fea, l1_points)

        # x = F.leaky_relu(self.bn1(self.conv1(l0_points)))
        x_dc = self.feac_dc_dec(l0_points).permute(0, 2, 1)
        x_ac = self.feac_ac_dec(l0_points).permute(0, 2, 1)
        # x_op = self.geo_op_dec(l0_points).permute(0, 2, 1)
        # x_sc = self.geo_sc_dec(l0_points).permute(0, 2, 1)
        # x_ro = self.geo_ro_dec(l0_points).permute(0, 2, 1)
        # x = F.log_softmax(x, dim=1)
        # x = l0_points[:,3:,:]
        x = torch.cat((x_dc,x_ac),dim=-1)
        return x