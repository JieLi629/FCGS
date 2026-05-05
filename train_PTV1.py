import os
import gc
import sys
import torch.multiprocessing as mp
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print("Using GPU:",os.environ["CUDA_VISIBLE_DEVICES"])
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import numpy as np
print(f"NumPy version: {np.__version__}")
print(f"NumPy path: {np.__file__}")
from torch.utils.tensorboard import SummaryWriter  
import torch.nn as nn
from typing import NamedTuple
from argparse import ArgumentParser
from utils.loss_utils import l1_loss,l2_loss,ssim
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from scene import Scene
# from model.FCGS_model import FCGS
from Modle_ALL_CTX_NEWGT_ME import FCGS
from layers.utils_scalable import render_train
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from random import randint
import random
import lpips
# from model.encodings_cuda import STE_multistep
from utils.image_utils import psnr
from model.encodings_cuda import encoder
lpips_model = lpips.LPIPS(net="alex").to("cuda")
import numpy as np
from train_base_PTV1 import Config_Set,load_state_dict
from dataloader.dataloader import DL3DVDataset
root_path = os.path.dirname(__file__)
writer = SummaryWriter('./log5')
from layers.utils_scalable import get_model
b2M = 8*1024*1024
B2M = 1024*1024
train_views = 16
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

class D2(NamedTuple):
    convert_SHs_python:bool
    compute_cov3D_python:bool
    debug:bool

def train(args):
    dataset = D1(
        data_device = 'cuda',
        eval = True,
        images = 'images',
        lod = 0,
        model_path = "",
        resolution = -1,
        sh_degree = 3,
        source_path = args.source_path,
        white_background = False
    )
    pipline = D2(
        convert_SHs_python = False,
        compute_cov3D_python = False,
        debug = False
    )
    PTV1 = get_model().cuda()
    DL3DV = DL3DVDataset(args.source_path,stage='train',N_train=train_views,Batch_size=1)
    TrainDataloader = DataLoader(DL3DV,batch_size=1,shuffle=True,num_workers=8)
    if args.stage == 'train':
        testing = False
    checkpoint = torch.load(f'./checkpoint/basePTV1_lr_0.25eminus4_samplek4_base_epoch1_step3400_Noise_Q_L1loss_with_split_feac_rec_loss/total/checkpoint_epoch1_step600.pth')#load an object file saved by torch.save function
    # checkpoint = None')#load an object file saved by torch.save function
    # checkpoint1 = torch.load('/data1/lij/FCGS_TEST/checkpoints/checkpoint_0.0008.pkl')
    PTV1 = load_state_dict(PTV1,checkpoint)
    optimizer, schedular = Config_Set(PTV1,1)
    optimizer.zero_grad()
    Scene_view_ID = 0
    PTV1.train()
    for epoch in range(1,args.total_epoch+1):
        for step,(gaussians_info,N_gaussian,camaras_list,train_viewpoints_image) in enumerate(TrainDataloader):
            Scene_view_ID += 1
            optimizer.zero_grad()
            if N_gaussian > 1000000 or N_gaussian < 100000:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                continue
            g_xyz = gaussians_info[0].cuda().detach()
            _,C,H,W = train_viewpoints_image.shape[1:]
            # g_xyz = (g_xyz - g_xyz.min(axis=0)[0])/(g_xyz.max(axis=0)[0]-g_xyz.min(axis=0)[0]+1e-9)
            g_feac = gaussians_info[1].cuda().detach()
            g_geo = gaussians_info[2].cuda().detach()
            g_coords_feac_fusion = torch.cat((g_xyz,g_geo,g_feac),dim=2)
            if not g_coords_feac_fusion.is_contiguous():
                g_coords_feac_fusion = g_coords_feac_fusion.contiguous()
            # g_coords_feac_fusion = g_coords_feac_fusion.permute(0,2,1)
            gt_image = train_viewpoints_image[0,...].clone().detach().cuda()
            render_gt = render_train(g_xyz,g_geo,camaras_list,g_feac,pipline,C,H,W).detach()
            psnr_gt_total = 0
            for i in range(len(camaras_list)):
                psnr_gt_total += psnr(render_gt[i,:,:,:],gt_image[i,:,:,:]).mean().double().item()
            psnr_gt_total = psnr_gt_total / len(camaras_list)
            # del render_gt
            fea_dec = PTV1(N_gaussian,g_coords_feac_fusion)
            
            if fea_dec == None:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                continue
            else:
                position, feac_dc, feac_ac = fea_dec
            gaussians_fea_dec = torch.cat((feac_dc,feac_ac),dim=-1)
            render = render_train(position,g_geo,camaras_list,gaussians_fea_dec,pipline,C,H,W)

            L1_loss = l1_loss(render,render_gt) 
            g_feac_dc, g_feac_ac = torch.split(g_feac,split_size_or_sections=[3,45], dim=-1)
            feac_rec_loss = 0.1 * l1_loss(g_feac_dc,feac_dc) + 0.05 * l1_loss(g_feac_ac,feac_ac) 
            ssim_train = ssim(render,render_gt)
            psnr_train_total = 0
            psnr_train_total_against_GT = 0
            for i in range(len(camaras_list)):
                psnr_train_total += psnr(render[i,:,:,:],render_gt[i,:,:,:]).mean().double().item()
                psnr_train_total_against_GT += psnr(render[i,:,:,:],gt_image[i,:,:,:]).mean().double().item()
            psnr_train_ave = psnr_train_total / len(camaras_list)
            psnr_train_against_GT_ave = psnr_train_total_against_GT / len(camaras_list)
            loss = 0.05 * (1-ssim_train) + 0.95 * L1_loss + feac_rec_loss
            
            loss.backward()

            # print(list(param.grad for _,param in PTV1.Encoder_Transform.named_parameters()))

            print(f"Epoch:{epoch} \t Step:{step} \t Gaussians:{N_gaussian.item()} \t learning_rate:{optimizer.param_groups[0]['lr']:.8f} \t loss_train:{loss.item():.15f} \t psnr:{psnr_train_ave:.4f} \t psnr_render_against_gt: {psnr_train_against_GT_ave:.4f} \t psnr_gt:{psnr_gt_total:.4f}")

            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            if step % 50 == 0:
                if not os.path.exists(os.path.join(root_path,args.model_save_path,'./total')):
                    os.makedirs(os.path.join(root_path,args.model_save_path,'./total'),exist_ok=True)
                torch.save({'model':PTV1.state_dict()},os.path.join(args.model_save_path,'./total',f'./checkpoint_epoch{epoch}_step{step}.pth'))
        schedular.step()
    print("training_finished")
if __name__ == '__main__':
    # mp.set_start_method('spawn',force=True)
    parser = ArgumentParser(description='Training script for FCGS')
    parser.add_argument("--lmd",default=8e-4)
    parser.add_argument("--nr",type=int,default=3)
    parser.add_argument("--source_path",type=str,default='./dataset/DL3DV-GS-960P/3DGS/')
    parser.add_argument("--total_epoch",nargs="+",type=int,default=50)
    parser.add_argument("--start_checkpoint",type=str,default=None)
    parser.add_argument("--stage",type=str,default='train')
    parser.add_argument("--lmd_mask",type=float,default='3e-3')
    parser.add_argument("--lamda_ssim",type=float,default=0.1)
    parser.add_argument("--determ",type=float,default=0)
    parser.add_argument("--model_save_path",type=str,default='./checkpoint/basePTV1_lr_0.5eminus4_samplek4_base_epoch1_step600_Noise_Q_L1loss_with_split_feac_rec_loss_renderGT')
    parser.add_argument("--rm_mask_from_loss_iter",type=int,default=800)
    parser.add_argument("--saved_ckpt_iter",type=list,default=[0,2000,4000,8000,10000])
    parser.add_argument("--training_phase",type=int,default=[1,2,3],help='the phase of training process')
    parser.add_argument("--phase_choices",type=list,default=[1,2,3],help='the phase of training process')
    args = parser.parse_args(sys.argv[1:])
    train(args)
    
