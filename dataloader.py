# import os
# import numpy 
# from torch.utils.data import Dataset
# from typing import NamedTuple
# from scene import Scene
# from gaussian_renderer import GaussianModel
# import torch
# import random
# class D1(NamedTuple):
#     data_device: str
#     eval:bool
#     images:str
#     lod:int
#     model_path:str
#     resolution:int
#     sh_degree:int
#     source_path:str
#     white_background:bool

# class DL3DVDataset(Dataset):
#     def __init__(self,files_path,stage,N_train,Batch_size):
#         super(DL3DVDataset,self).__init__()
#         self.train_file_list_path = os.path.join(files_path,f'../../hash_name_{stage}.txt')
#         self.stage = stage
#         self.N_train = N_train
#         self.Batch_size = Batch_size
#         with open(self.train_file_list_path) as f:
#             lines = f.readlines()
#             lines = [line.strip() for line in lines]
#         self.gaussian_list = []
#         self.image_list = []
#         for line in lines:
#             gaussian_path = os.path.join(files_path,line,'./point_cloud/iteration_30000/point_cloud.ply')
#             image_path = os.path.join(files_path,'../imgs_undist',line)
#             if os.path.exists(gaussian_path) and os.path.exists(image_path):
#                 self.gaussian_list.append(gaussian_path)
#                 self.image_list.append(image_path)
#         self.gaussian = GaussianModel(3)
#         gaussian_file = self.gaussian_list[3]
#         self.gaussian.load_ply(gaussian_file)
#         image_file = self.image_list[3]
#         image_args = D1(data_device = 'cpu',eval = True,images = 'images',lod = 0,model_path = "",resolution = -1,sh_degree = 3,source_path = image_file,white_background = False)
#         scene = Scene(image_args,shuffle=False)
#         # self.viewpointstack = random.sample(scene.getTrainCameras().copy(),self.N_train)
#         self.viewpointstack = scene.getTrainCameras().copy()
    
#     def __len__(self):
#         # assert len(self.gaussian_list) == len(self.image_list), f"Error! Gaussians list length must be equal to image list"
#         return len(self.viewpointstack)
    
#     def __getitem__(self,idx):
#         # from gaussian_renderer import GaussianModel
#         view = self.viewpointstack[idx]
#         C,H,W = view.original_image.shape
#         camaras_list = []
#         # for view in viewpointstack:
#         camaras_list.append(
#             {
#                 'FoVx':view.FoVx,
#                 'FoVy':view.FoVy,
#                 'image_height':view.image_height,
#                 'image_width':view.image_width,
#                 'world_view_transform':view.world_view_transform,
#                 'full_proj_transform':view.full_proj_transform,
#                 'camera_center':view.camera_center
#             }
#         )
#         # train_viewpoints_image = torch.stack([view.original_image for view in viewpointstack]).view(-1,C,H,W).detach()
#         train_viewpoints_image = view.original_image.view(-1,C,H,W).detach()
#         g_xyz = self.gaussian._xyz.detach()
#         N_gaussian = g_xyz.shape[0]
#         g_feac_dc = self.gaussian._features_dc.view(N_gaussian,-1).detach()
#         g_feac_rst = self.gaussian._features_rest.view(N_gaussian,-1).detach()
#         g_geo_op = self.gaussian._opacity.view(N_gaussian,-1).detach()
#         g_geo_sc = self.gaussian._scaling.view(N_gaussian,-1).detach()
#         g_geo_ro = self.gaussian._rotation.view(N_gaussian,-1).detach()
        
#         gaussians_feac = torch.cat((g_feac_dc,g_feac_rst),dim=-1).detach()
#         gaussians_geo = torch.cat((g_geo_op,g_geo_sc,g_geo_ro),dim=-1).detach()
#         gaussians_info = [g_xyz,gaussians_feac,gaussians_geo]

#         return gaussians_info,N_gaussian,camaras_list,train_viewpoints_image

import os
import numpy 
from torch.utils.data import Dataset
from typing import NamedTuple
from scene import Scene
from gaussian_renderer import GaussianModel
import torch
import random
class D1(NamedTuple):
    data_device: str
    eval:bool
    images:str
    lod:int
    model_path:str
    resolution:int
    sh_degree:int
    source_path:str
    white_background:bool

class DL3DVDataset(Dataset):
    def __init__(self,files_path,stage,N_train,Batch_size):
        super(DL3DVDataset,self).__init__()
        self.train_file_list_path = os.path.join(files_path,f'../../hash_name_{stage}.txt')
        self.stage = stage
        self.N_train = N_train
        self.Batch_size = Batch_size
        with open(self.train_file_list_path) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        self.gaussian_list = []
        self.image_list = []
        for line in lines:
            gaussian_path = os.path.join(files_path,line,'./point_cloud/iteration_30000/point_cloud.ply')
            image_path = os.path.join(files_path,'../imgs_undist',line)
            if os.path.exists(gaussian_path) and os.path.exists(image_path):
                self.gaussian_list.append(gaussian_path)
                self.image_list.append(image_path)
    
    def __len__(self):
        assert len(self.gaussian_list) == len(self.image_list), f"Error! Gaussians list length must be equal to image list"
        return len(self.gaussian_list)
    
    def __getitem__(self,idx):
        # from gaussian_renderer import GaussianModel
        gaussian = GaussianModel(3)
        gaussian_file = self.gaussian_list[idx]
        image_file = self.image_list[idx]
        image_args = D1(data_device = 'cpu',eval = True,images = 'images',lod = 0,model_path = "",resolution = -1,sh_degree = 3,source_path = image_file,white_background = False)
        scene = Scene(image_args,shuffle=False)
        # random.seed(1)
        if len(scene.getTrainCameras().copy()) < self.N_train:
            self.N_train = len(scene.getTrainCameras().copy())
        viewpointstack = random.sample(scene.getTrainCameras().copy(),self.N_train)
        C,H,W = viewpointstack[0].original_image.shape
        camaras_list = []
        for view in viewpointstack:
            camaras_list.append(
                {
                    'FoVx':view.FoVx,
                    'FoVy':view.FoVy,
                    'image_height':view.image_height,
                    'image_width':view.image_width,
                    'world_view_transform':view.world_view_transform,
                    'full_proj_transform':view.full_proj_transform,
                    'camera_center':view.camera_center
                }
            )
        train_viewpoints_image = torch.stack([view.original_image for view in viewpointstack]).view(-1,C,H,W).detach()
        gaussian.load_ply(gaussian_file)
        g_xyz = gaussian._xyz.detach()
        N_gaussian = g_xyz.shape[0]
        g_feac_dc = gaussian._features_dc.view(N_gaussian,-1).detach()
        g_feac_rst = gaussian._features_rest.view(N_gaussian,-1).detach()
        g_geo_op = gaussian._opacity.view(N_gaussian,-1).detach()
        g_geo_sc = gaussian._scaling.view(N_gaussian,-1).detach()
        g_geo_ro = gaussian._rotation.view(N_gaussian,-1).detach()
        
        gaussians_feac = torch.cat((g_feac_dc,g_feac_rst),dim=-1).detach()
        gaussians_geo = torch.cat((g_geo_op,g_geo_sc,g_geo_ro),dim=-1).detach()
        gaussians_info = [g_xyz,gaussians_feac,gaussians_geo]

        return gaussians_info,N_gaussian,camaras_list,train_viewpoints_image
