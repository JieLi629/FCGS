import os
import torch.optim as optim
import torch

def Config_Set(model,phase):
    print("Start Setting Training Configuration...")
    if phase == 1:
        parameters_group = [
        # {"params":model.linear_q.parameters(),"lr":1e-4},
        # {"params":model.linear_k.parameters(),"lr":1e-4},
        # {"params":model.linear_v.parameters(),"lr":1e-4},
        # {"params":model.linear_p.parameters(),"lr":1e-4},
        # {"params":model.linear_w.parameters(),"lr":1e-4},
        # {"params":model.deembeding.parameters(),"lr":1e-4},
        {"params":model.ad_fe,"lr":0.5e-5},
        {"params":model.ad_op,"lr":0.5e-5},
        {"params":model.ad_sc,"lr":0.5e-5},
        {"params":model.ad_ro,"lr":0.5e-5},
        {"params":model.Encoder_Transform.parameters(),"lr":1e-4},
        {"params":model.Encoder_hyper_feac_m1.parameters(),"lr":1e-4},
        {"params":model.Decoder_hyper_feac_m1.parameters(),"lr":1e-4},
        {"params":model.Encoder_hyper_feac_m0.parameters(),"lr":1e-4},
        {"params":model.Decoder_hyper_feac_m0.parameters(),"lr":1e-4},
        {"params":model.Encoder_hyper_geo.parameters(),"lr":1e-4},
        {"params":model.Decoder_hyper_geo.parameters(),"lr":1e-4},
        {"params":model.Factorize_model_m1.parameters(),"lr":1e-4},
        {"params":model.Factorize_model_m0.parameters(),"lr":1e-4},
        {"params":model.Factorize_model_geo.parameters(),"lr":1e-4},
        {"params":model.Intra_Context_Model_M1.parameters(),"lr":1e-4},
        {"params":model.Intra_Context_Model_M0.parameters(),"lr":1e-4},

        {"params":model.feac_SpaCxt_M1.parameters(),"lr":1e-4},
        {"params":model.feac_SpaCxt_M0.parameters(),"lr":1e-4},
        {"params":model.fea_geo_SpaCxt.parameters(),"lr":1e-4},
        {"params":model.latdim_2_griddim.parameters(),"lr":1e-4},
        # {"params":model.sa3.parameters(),"lr":1e-4},
        # {"params":model.fp3.parameters(),"lr":1e-4},
        {"params":model.Decoder_Transform.parameters(),"lr":1e-4},
        {"params":model.Post_feac_dec.parameters(),"lr":1e-4},
        # {"params":model.fea_enc.parameters(),"lr":1e-4},
        # {"params":model.Q_l0,"lr":1e-4},
        # {"params":model.Q_l1,"lr":1e-4},
        # {"params":model.Q_l2,"lr":1e-4},
        # {"params":model.Q_l3,"lr":1e-4},
        # {"params":model.conv1.parameters(),"lr":1e-4},
        # {"params":model.bn1.parameters(),"lr":1e-4},
        # {"params":model.feac_dc_dec.parameters(),"lr":1e-4},
        # {"params":model.feac_ac_dec.parameters(),"lr":1e-4},
        # {"params":model.pre_enc.parameters(),"lr":1e-4},
        # {"params":model.post_dec.parameters(),"lr":1e-4},
        # {"params":model.Decoder_fea_dc_ME.parameters(),"lr":1e-4},
        # {"params":model.Decoder_fea_ac_ME.parameters(),"lr":1e-4},

        ]
        optimizer = optim.Adam(parameters_group,betas=(0.9,0.999),weight_decay=0)
        schedular = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=10,gamma=0.5)
        print(f"Optimizer:Adam \t lr:{parameters_group[0]['lr']}")
    elif phase == 2:
        parameters_group = [
        {"params":model.Encoder_mask.parameters(),"lr":2e-5},
        {"params":model.Encoder_ga.parameters(),"lr":2e-5},
        {"params":model.Encoder_hyper_feac_m1.parameters(),"lr":2e-5},
        {"params":model.Entropy_model_fea_m1.parameters(),"lr":2e-5},
        {"params":model.Decoder_hyper_feac_m1.parameters(),"lr":2e-5},
        {"params":model.Entropy_lat_feac_m1.parameters(),"lr":2e-5},
        # {"params":model.latdim_2_griddim.parameters(),"lr":2e-5},
        # # {"params":model.Freq_Encoder.parameters(),"lr":1e-5},
        # {"params":model.feac_SpaCxt_M1.parameters(),"lr":2e-5},
        # {"params":model.Intra_Context_Model_M1.parameters(),"lr":2e-5},
        {"params":model.Decoder_gs.parameters(),"lr":2e-5},
        {"params":model.Decoder_fea_dc.parameters(),"lr":2e-5},
        {"params":model.Decoder_fea_ac.parameters(),"lr":2e-5}
        ]
        optimizer = optim.Adam(parameters_group,betas=(0.9,0.999),weight_decay=0)
        schedular = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=300,gamma=2)
    elif phase == 3:
        # parameters_group = [
        # {"params":model.Encoder_mask.parameters(),"lr":1.22e-5},
        # {"params":model.Encoder_ga.parameters(),"lr":2.2e-5},
        # {"params":model.Encoder_hyper_feac_m1.parameters(),"lr":2.2e-5},
        # {"params":model.Entropy_model_fea_m1.parameters(),"lr":2.2e-5},
        # {"params":model.Decoder_hyper_feac_m1.parameters(),"lr":2.2e-5},
        # {"params":model.Entropy_lat_m1.parameters(),"lr":2.2e-5},
        # {"params":model.Entropy_lat_m0.parameters(),"lr":2.2e-5},
        # {"params":model.Entropy_lat_geo.parameters(),"lr":2.2e-5},
        # {"params":model.Encoder_hyper_feac_m0.parameters(),"lr":2.2e-5},
        # {"params":model.Decoder_hyper_feac_m0.parameters(),"lr":2.2e-5},
        # {"params":model.Encoder_hyper_geo.parameters(),"lr":2.2e-5},
        # {"params":model.Decoder_hyper_geo.parameters(),"lr":2.2e-5},
        # {"params":model.Entropy_model_fea_m0.parameters(),"lr":2.2e-5},
        # {"params":model.Entropy_model_geo.parameters(),"lr":2.2e-5},
        # {"params":model.Decoder_gs.parameters(),"lr":2.2e-5},
        # {"params":model.Decoder_fea_dc.parameters(),"lr":2.2e-5},
        # {"params":model.Decoder_fea_ac.parameters(),"lr":2.2e-5},
        # {"params":model.latdim_2_griddim.parameters(),"lr":2.2e-5},
        # {"params":model.Freq_Encoder.parameters(),"lr":2.2e-5},
        # {"params":model.feac_SpaCxt_M1.parameters(),"lr":2.2e-5},
        # {"params":model.feac_SpaCxt_M0.parameters(),"lr":2.2e-5},
        # {"params":model.fea_geo_SpaCxt.parameters(),"lr":2.2e-5},
        # {"params":model.Intra_Context_Model_M1.parameters(),"lr":2.2e-5},
        # {"params":model.Intra_Context_Model_M0.parameters(),"lr":2.2e-5},
        # {"params":model.ad_fe,"lr":2.2e-5},
        # {"params":model.ad_op,"lr":2.2e-5},
        # {"params":model.ad_sc,"lr":2.2e-5},
        # {"params":model.ad_ro,"lr":2.2e-5}
        # ]
        parameters_group = [
        {"params":model.Encoder_mask.parameters(),"lr":2e-5},
        {"params":model.Encoder_ga.parameters(),"lr":2e-5},
        {"params":model.Encoder_hyper_feac_m1.parameters(),"lr":2e-5},
        {"params":model.Entropy_model_fea_m1.parameters(),"lr":2e-5},
        {"params":model.Decoder_hyper_feac_m1.parameters(),"lr":2e-5},
        {"params":model.Entropy_lat_m1.parameters(),"lr":2e-5},
        {"params":model.Entropy_lat_m0.parameters(),"lr":2e-5},
        {"params":model.Entropy_lat_geo.parameters(),"lr":2e-5},
        {"params":model.Encoder_hyper_feac_m0.parameters(),"lr":1e-5},
        {"params":model.Decoder_hyper_feac_m0.parameters(),"lr":1e-5},
        {"params":model.Encoder_hyper_geo.parameters(),"lr":1e-5},
        {"params":model.Decoder_hyper_geo.parameters(),"lr":1e-5},
        {"params":model.Entropy_model_fea_m0.parameters(),"lr":1e-5},
        {"params":model.Entropy_model_geo.parameters(),"lr":1e-5},
        {"params":model.Decoder_gs.parameters(),"lr":2e-5},
        {"params":model.Decoder_fea_dc.parameters(),"lr":2e-5},
        {"params":model.Decoder_fea_ac.parameters(),"lr":2e-5},
        {"params":model.latdim_2_griddim.parameters(),"lr":2e-5},
        {"params":model.Freq_Encoder.parameters(),"lr":2e-5},
        {"params":model.feac_SpaCxt_M1.parameters(),"lr":2e-5},
        {"params":model.feac_SpaCxt_M0.parameters(),"lr":1e-5},
        {"params":model.fea_geo_SpaCxt.parameters(),"lr":1e-5},
        {"params":model.Intra_Context_Model_M1.parameters(),"lr":2e-5},
        {"params":model.Intra_Context_Model_M0.parameters(),"lr":1e-5},
        {"params":model.ad_fe,"lr":1e-5},
        {"params":model.ad_op,"lr":1e-5},
        {"params":model.ad_sc,"lr":1e-5},
        {"params":model.ad_ro,"lr":1e-5}
        ]
        optimizer = optim.Adam(parameters_group,betas=(0.9,0.999),weight_decay=0)
        schedular = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=500,gamma=2)
    return optimizer,schedular

def load_state_dict(model,ckpt=None,ckpt1=None):

    model_dict = model.state_dict()
    # dict_skip = {'latdim_2_griddim','Freq_Encoder','feac_SpaCxt_M1','Intra_Context_Model_M1','Encoder_hyper_feac_m1','Decoder_hyper_feac_m1','Entropy_model_fea_m1','Encoder_hyper_feac_m0','Decoder_hyper_feac_m0','Entropy_model_fea_m0','Entropy_lat_m1','Encoder_hyper_geo','Decoder_hyper_geo','Entropy_model_geo','fea_geo_SpaCxt','ad_op','ad_sc','ad_ro'}
    dict_frozen = {'ad_op','ad_sc','ad_ro','ad_fe','Encoder_mask'}
    dict = {'ad_op','ad_sc','ad_ro','ad_fe','Encoder_mask'}
    # dict_untrain = {'Encoder_ga', 'Decoder_gs', 'Decoder_fea_dc', 'Decoder_fea_ac','ad_sc','ad_ro','ad_op','fea_geo_SpaCxt','Encoder_hyper_geo','Decoder_hyper_geo','Entropy_model_geo'}
    # dict_extern = {'Encoder_hyper_feac_m1','Decoder_hyper_feac_m1','Entropy_model_fea_m1','ad_fe','ad_sc','ad_ro','ad_op','fea_geo_SpaCxt','Encoder_hyper_feac_m0','Decoder_hyper_feac_m0','Entropy_model_fea_m0','Intra_Context_Model_M0','Intra_Context_Model_M1','feac_SpaCxt_M0','latdim_2_griddim','Freq_Encoder','feac_SpaCxt_M1','Encoder_hyper_feac_m1','Decoder_hyper_feac_m1','Entropy_model_fea_m1','Encoder_hyper_geo','Decoder_hyper_geo','Entropy_model_geo','Encoder_hyper_feac_m0','Decoder_hyper_feac_m0','Entropy_model_fea_m0','Intra_Context_Model_M0','Intra_Context_Model_M1'}
    if ckpt:
        # pretrained_dict = {k:v for k,v in ckpt.items() if k.split('.')[0] in dict}
        pretrained_dict = {k:v for k,v in ckpt['model'].items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
    if ckpt1:
        pretrained_dict_release = {k:v for k,v in ckpt1.items() if k.split('.')[0] in dict}
        model_dict.update(pretrained_dict_release)
    # if ckpt1:
    #     pretrained_dict1 = {k:v for k,v in ckpt1.items() if k.split('.')[0] in dict}
    # model_dict.update(pretrained_dict1)# dict update method
    model.load_state_dict(model_dict)
    if dict_frozen:
            for name, para in model.named_parameters():
                if name.split('.')[0] in dict_frozen:
                    para.requires_grad = False
    return model
# def load_state_dict(model, ckpt, ckpt1=None):
#     model_dict = model.state_dict()
#     untrainable_prefixes = {
#         'Encoder_mask', 
#         # 'Encoder_ga', 'Decoder_gs', 
#         # 'Decoder_fea_dc', 'Decoder_fea_ac', 
#         # # 'Encoder_mask',
#         # 'ad_fe', 'ad_op', 'ad_sc', 'ad_ro',
#         # 'fea_geo_SpaCxt',
#         # 'Entropy_model_geo',
#         # 'Encoder_hyper_geo',
#         # 'Decoder_hyper_geo'
#     }
#     skip_dict = {
#         # 'Entropy_model_fea_m1','Entropy_model_fea_m0','Entropy_model_geo'
#     }
#     skip_dict1 = {
#         # 'ad_fe', 'ad_op', 'ad_sc', 'ad_ro'
#     }
#     ckpt1_to_model_map = {
#         'Encoder_fea': 'Encoder_ga',      
#         'Decoder_fea': 'Decoder_gs',     
#         'head_f_dc': 'Decoder_fea_dc',  
#         'head_f_rst': 'Decoder_fea_ac',
#         'latdim_2_griddim_fea':'latdim_2_griddim',
#         'Encoder_fea_hyp':'Encoder_hyper_feac_m1',
#         'Decoder_fea_hyp':'Decoder_hyper_feac_m1',
#         'Encoder_feq_hyp':'Encoder_hyper_feac_m0',
#         'Decoder_feq_hyp':'Decoder_hyper_feac_m0',
#         'Encoder_geo_hyp':'Encoder_hyper_geo',
#         'Decoder_geo_hyp':'Decoder_hyper_geo',
#         'fea_channel_ctx.mean_d0':'Intra_Context_Model_M1.mean_channel_1',
#         'fea_channel_ctx.scale_d0':'Intra_Context_Model_M1.scale_channel_1',
#         'fea_channel_ctx.prob_d0':'Intra_Context_Model_M1.weight_channel_1',
#         'fea_channel_ctx.MLP_d0':'Intra_Context_Model_M1.channel_ctx_mlp_1',
#         'fea_channel_ctx.MLP_d1':'Intra_Context_Model_M1.channel_ctx_mlp_2',
#         'fea_channel_ctx.MLP_d2':'Intra_Context_Model_M1.channel_ctx_mlp_3',
#         'feq_channel_ctx.mean_d0':'Intra_Context_Model_M0.mean_channel_1',
#         'feq_channel_ctx.scale_d0':'Intra_Context_Model_M0.scale_channel_1',
#         'feq_channel_ctx.prob_d0':'Intra_Context_Model_M0.weight_channel_1',
#         'feq_channel_ctx.MLP_d0':'Intra_Context_Model_M0.channel_ctx_mlp_1',
#         'feq_channel_ctx.MLP_d1':'Intra_Context_Model_M0.channel_ctx_mlp_2',
#         'context_analyzer_geo':'fea_geo_SpaCxt.Cxt_Post',
#         'context_analyzer_fea':'feac_SpaCxt_M1.Cxt_Post',
#         'context_analyzer_feq':'feac_SpaCxt_M0.Cxt_Post',
#         'EF_fea':'Entropy_model_fea_m1',
#         'EF_feq':'Entropy_model_fea_m0',
#         'EF_geo':'Entropy_model_geo'
#     }

#     def get_state_dict(ckpt_data):
#         if ckpt_data is None:
#             return {}
#         if isinstance(ckpt_data, dict) and 'model' in ckpt_data:
#             return ckpt_data['model']
#         return ckpt_data

#     if ckpt:
#         pretrained_dict = get_state_dict(ckpt)
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k.split('.')[0] not in skip_dict}
#         model_dict.update(pretrained_dict)
#         print(f"[Load] Loaded {len(pretrained_dict)} layers from ckpt (Base).")
#     if ckpt1:
#         pretrained_dict_1 = get_state_dict(ckpt1)
#         count_override = 0
        
#         for k, v in pretrained_dict_1.items():
#             target_k = k

#             for old_prefix, new_prefix in ckpt1_to_model_map.items():
#                 if k.startswith(old_prefix + '.'):
#                     target_k = k.replace(old_prefix, new_prefix, 1)
#                     break
#                 elif k == old_prefix:
#                     target_k = new_prefix
#                     break
#             if target_k in model_dict and target_k not in skip_dict1:
#                 if model_dict[target_k].shape == v.shape:
#                     model_dict[target_k] = v
#                     count_override += 1
#                 else:
#                     print(f"[Warning] Shape mismatch for {target_k}. Model: {model_dict[target_k].shape}, Ckpt1: {v.shape}")
#             elif k in model_dict and k not in ckpt1_to_model_map and k not in skip_dict1:
#                  if model_dict[k].shape == v.shape:
#                     model_dict[k] = v
#                     count_override += 1
                    
#         print(f"[Load] Loaded/Mapped {count_override} layers from ckpt1 (Override).")

#     model.load_state_dict(model_dict, strict=True)

#     for name, para in model.named_parameters():
#         prefix = name.split('.')[0]
#         if prefix in untrainable_prefixes:
#             para.requires_grad = False
#         else:
#             para.requires_grad = True
            
#     return model