import os
import gc
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Using GPU:",os.environ["CUDA_VISIBLE_DEVICES"])
import torch
import numpy as np
print(f"NumPy version: {np.__version__}")
print(f"NumPy path: {np.__file__}")
# from torch.utils.tensorboard import SummaryWriter  
import torch.nn as nn
from typing import NamedTuple
from argparse import ArgumentParser
from utils.loss_utils import l1_loss,l2_loss,ssim
# from model.FCGS_model import FCGS
from layers.utils_mask_CTX_fcgs_m_modify_quant import render_train
from torch.utils.data import DataLoader
import lpips
# from model.encodings_cuda import STE_multistep
from utils.image_utils import psnr
lpips_model = lpips.LPIPS(net="alex").to("cuda")
import numpy as np
from train_base_PTV1_mask_CTX_fcgs_m_modify_quant_frozen_quant import Config_Set,load_state_dict
from dataloader.dataloader import DL3DVDataset
root_path = os.path.dirname(__file__)
# writer = SummaryWriter('./log5')
from layers.utils_mask_CTX_fcgs_m_modify_quant import get_model
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
    PTV1 = get_model(args=args).cuda()
    DL3DV = DL3DVDataset(args.source_path,stage='train',N_train=train_views,Batch_size=1)
    TrainDataloader = DataLoader(DL3DV,batch_size=1,shuffle=True,num_workers=4,pin_memory=False)
    if args.stage == 'train':
        testing = False
    checkpoint = torch.load(f'/home/lilimaogroup/lijie/FCGS/checkpoint/basePTV1_RD_lr_1eminus4_split_degree_reac_rec_loss_FCGS_mask_with_FCGS_CTX_finetune_v2/total/checkpoint_epoch76_step200.pth')#load an object file saved by torch.save function
    # checkpoint = None')#load an object file saved by torch.save function
    checkpoint1 = torch.load('/home/lilimaogroup/lijie/FCGS/checkpoints/checkpoint_0.0016.pkl')
    PTV1 = load_state_dict(PTV1,checkpoint,checkpoint1)
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
            # g_coords_feac_fusion = g_coords_feac_fusion.permute(0,2,1)
            gt_image = train_viewpoints_image[0,...].clone().detach().cuda()
            render_gt = render_train(g_xyz,g_geo,camaras_list,g_feac,pipline,C,H,W).detach()
            psnr_gt_total = 0
            for i in range(len(camaras_list)):
                psnr_gt_total += psnr(render_gt[i,:,:,:],gt_image[i,:,:,:]).mean().double().item()
            psnr_gt_total = psnr_gt_total / len(camaras_list)
            del render_gt
            fea_dec, mask, bpp = PTV1(N_gaussian,g_coords_feac_fusion)
            
            if fea_dec == None:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                continue
            else:
                position, geo, feac_dc, feac_ac = fea_dec
            gaussians_fea_dec = torch.cat((feac_dc,feac_ac),dim=-1)
            render = render_train(position,geo,camaras_list,gaussians_fea_dec,pipline,C,H,W)

            L1_loss = l1_loss(render,gt_image) 
            # g_feac_dc, g_feac_ac = torch.split(g_feac,split_size_or_sections=[3,45], dim=-1)
            # feac_rec_loss = 0.2 * l1_loss(g_feac_dc,feac_dc) + 0.25 * l1_loss(g_feac_ac[...,:9],feac_ac[...,:9]) + 0.4 * l1_loss(g_feac_ac[...,9:24],feac_ac[...,9:24]) + 0.25 * l1_loss(g_feac_ac[...,24:],feac_ac[...,24:])
            ssim_train = ssim(render,gt_image)
            psnr_train_total = 0
            for i in range(len(camaras_list)):
                psnr_train_total += psnr(render[i,:,:,:],gt_image[i,:,:,:]).mean().double().item()
            psnr_train_ave = psnr_train_total / len(camaras_list)
            loss = 0.05 * (1-ssim_train) + 0.95 * L1_loss + args.lmd * bpp
            # loss = 0.1 * mask.mean() 
            loss.backward()

            # print(list(param.grad for _,param in PTV1.Encoder_Transform.named_parameters()))
            print(f"Epoch:{epoch} \t Step:{step} \t Gaussians:{N_gaussian.item()} \t learning_rate:{optimizer.param_groups[0]['lr']:.8f} \t loss_train:{loss.item():.15f} \t psnr:{psnr_train_ave:.15f} \t psnr_gt:{psnr_gt_total:.15f} \t mask_ratio:{mask.mean().item():.4f} \t bpp:{bpp.mean().item():.4f}",flush=True)
            if step % 100 == 0:
                print(f"Allocated: {torch.cuda.memory_allocated()/1e9:0.5f} GB")
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            if step % 10 == 0:
                if not os.path.exists(os.path.join(root_path,args.model_save_path,'./total')):
                    os.makedirs(os.path.join(root_path,args.model_save_path,'./total'),exist_ok=True)
                torch.save({'model':PTV1.state_dict()},os.path.join(args.model_save_path,'./total',f'./checkpoint_epoch{epoch}_step{step}.pth'))
        schedular.step()
    print("training_finished")
if __name__ == '__main__':
    # mp.set_start_method('spawn',force=True)
    parser = ArgumentParser(description='Training script for FCGS')
    parser.add_argument("--lmd",default=16e-4)
    parser.add_argument("--nr",type=int,default=3)
    parser.add_argument("--source_path",type=str,default='/home/lilimaogroup/lijie/FCGS/dataset')
    parser.add_argument("--total_epoch",nargs="+",type=int,default=100)
    parser.add_argument("--start_checkpoint",type=str,default=None)
    parser.add_argument("--stage",type=str,default='train')
    parser.add_argument("--lmd_mask",type=float,default='3e-3')
    parser.add_argument("--lamda_ssim",type=float,default=0.1)
    parser.add_argument("--determ",type=float,default=0)
    parser.add_argument("--model_save_path",type=str,default='./checkpoint/basePTV1_RD_lr_1eminus4_split_degree_reac_rec_loss_FCGS_mask_with_FCGS_CTX_finetune_v2_modify_quant_frozen_quant_L2loss_limit_hyp1_rate')
    parser.add_argument("--rm_mask_from_loss_iter",type=int,default=800)
    parser.add_argument("--saved_ckpt_iter",type=list,default=[0,2000,4000,8000,10000])
    parser.add_argument("--training_phase",type=int,default=[1,2,3],help='the phase of training process')
    parser.add_argument("--phase_choices",type=list,default=[1,2,3],help='the phase of training process')
    args = parser.parse_args(sys.argv[1:])
    train(args)
    
