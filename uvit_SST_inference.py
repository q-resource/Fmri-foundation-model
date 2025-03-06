import random

import numpy as np
import sys
import torch
import nibabel as nib
# sys.path.append('/users/win-fmrib-analysis/eov316/wgong/ukbiobank/ukb_braindiffusion/src/U-ViT/')
# sys.path.append('/gpfs3/well/win-biobank/users/eov316/ukbiobank/ukb_braindiffusion/src/')
sys.path.append('/home/jiaty/U_VIT/')
from gongcode.models import UViT
# from dataloaders import load_Nifti_data_one_mod
from torch.utils.data import Dataset, DataLoader
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import os
import sde
import accelerate
from torch import multiprocessing as mp
from fmridataset import *
import einops
# from scipy import linalg, stats
import random
import torch
import pandas as pd
# from glmnet import ElasticNet
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
def g_impute_nan_as_mean(x):

    x_mean = np.nanmean(x,axis = 0)
    for i in range(0,x.shape[1]):
        x[np.isnan(x[:,i]),i] = x_mean[i]

    return x



import argparse


def cli_parser():
    parser = argparse.ArgumentParser(description="UVIT_fmri")
    parser.add_argument('-lr', default=0.0001, type=int,help='Learning rate')
    parser.add_argument('-batch_size', default=689, type=int,help='Batch size')
    parser.add_argument('-nepoch', default=10, type=int,help='Number of epochs')
    parser.add_argument('-depth', default=10, type=int,help='Number of epochs')
    parser.add_argument('-embed_dim', default=512, type=int,help='Number of epochs')
    parser.add_argument('-patch_size', default=8,type=int, help='Number of epochs')
    parser.add_argument('-pred_mode', default='noise_pred',type=str, help='Number of epochs')
    parser.add_argument('-num_heads', default=8,type=int, help='Number of epochs')
    parser.add_argument('-use_checkpoint', default=True, type=bool,help='Number of epochs')
    parser.add_argument('-resumeEP', default=399,type=int, help='Number of epochs')
    parser.add_argument('-resumeITER', default=0,type=int, help='Number of epochs')
    parser.add_argument('-select_mod', default=0, type=int,help='Number of epochs')
    parser.add_argument('-cfg_scale', default=12, type=int,help='Number of epochs')
    parser.add_argument('-nstep', default=50,type=int, help='Number of epochs')
    parser.add_argument('--using_FC', default=False, type=bool)
    parser.add_argument('--condition_num', default=1, type=int)
    parser.add_argument('-inference_mode', default='noise_pred', type=str, help='noise_pred or train_mean')
    parser.add_argument('--using_mask', default=False, type=bool)
    parser.add_argument('--using_vbm', default=False, type=bool)
    parser.add_argument('-self_mask_rate', default=None, type=int, help='self_mask_rate')
    parser.add_argument('-mask_rate', default=0, type=float, help='mask_rate')
    parser.add_argument('--using_dti', default=False, type=bool)
    parser.add_argument('--using_seedfc',default=False,type=bool,help='using seedfc or not')
    parser.add_argument('--seedfc_id', default=0, type=int, help='the seedfc of region id list(range(1, 101))')
    parser.add_argument('--using_norm', default=False, type=bool, help='using normalized or not')
    parser.add_argument('--region_mask_id', default=None, type=int, help='the id of region in template')
    return parser


def load_data(args,region_mask_id):
    global finnal_test_FC_data,finnal_test_vbm_data,finnal_test_seedfc_data,finnal_test_dti_data,mask_list
    finnal_test_FC_data, finnal_test_vbm_data, finnal_test_seedfc_data, finnal_test_dti_data, mask_list=0,0,0,0,0
    SSTBL = np.load('/home/jiaty/SST_dataset/SSTBLcommon_post.npy')
    index_SSTBL = pd.read_csv("/home/jiaty/SST_dataset/SSTBLtraintestindex.csv")
    index_SSTBL['index'] = index_SSTBL['index'].astype(int)
    SSTBL_train_BL = index_SSTBL[index_SSTBL['index'] == 1].index
    SSTBL_test_BL = index_SSTBL[index_SSTBL['index'] == 0].index
    SSTBL_train_data_BL = SSTBL[SSTBL_train_BL]
    SSTBL_test_data_BL = SSTBL[SSTBL_test_BL]


    SSTBL_fu2 = np.load('/home/jiaty/SST_dataset/SSTFU2common_post.npy')
    index_SSTBL_fu2 = pd.read_csv("/home/jiaty/SST_dataset/SSTFU2traintestindex.csv")
    index_SSTBL_fu2['index'] = index_SSTBL_fu2['index'].astype(int)
    SSTBL_train_fu2 = index_SSTBL_fu2[index_SSTBL_fu2['index'] == 1].index
    SSTBL_test_fu2 = index_SSTBL_fu2[index_SSTBL_fu2['index'] == 0].index
    SSTBL_train_data_fu2 = SSTBL_fu2[SSTBL_train_fu2]
    SSTBL_test_data_fu2 = SSTBL_fu2[SSTBL_test_fu2]

    SSTBL_fu3 = np.load('/home/jiaty/SST_dataset/SSTFU3common_post.npy')
    index_SSTBL_fu3 = pd.read_csv("/home/jiaty/SST_dataset/SSTFU3traintestindex.csv")
    index_SSTBL_fu3['index'] = index_SSTBL_fu3['index'].astype(int)
    SSTBL_train_fu3 = index_SSTBL_fu3[index_SSTBL_fu3['index'] == 1].index
    SSTBL_test_fu3 = index_SSTBL_fu3[index_SSTBL_fu3['index'] == 0].index
    SSTBL_train_data_fu3 = SSTBL_fu3[SSTBL_train_fu3]
    SSTBL_test_data_fu3 = SSTBL_fu3[SSTBL_test_fu3]

    final_train_data = np.concatenate(( SSTBL_train_data_fu2, SSTBL_train_data_fu3,SSTBL_train_data_BL), axis=0)
    finnal_test_data = np.concatenate(( SSTBL_test_data_fu2, SSTBL_test_data_fu3,SSTBL_test_data_BL), axis=0)

    num_list = []
    num_list.append(SSTBL_train_data_fu2.shape[0])
    num_list.append(SSTBL_train_data_fu3.shape[0])
    num_list.append(SSTBL_train_data_BL.shape[0])
    print("train number", num_list)  # [1034, 892, 754]

    num_list_test = []
    num_list_test.append(SSTBL_test_data_BL.shape[0])
    num_list_test.append(SSTBL_test_data_fu2.shape[0])
    num_list_test.append(SSTBL_test_data_fu3.shape[0])
    print("test_num", num_list_test)  # [262, 232, 195]
    print("final_train_data", final_train_data.shape)
    print("finnal_test_data", finnal_test_data.shape)
    if args.using_FC:
        SSTBL_FC = np.load('/home/jiaty/SST_dataset/SSTBLcommon_FC.npy')
        SSTBL_train_FC_BL = SSTBL_FC[SSTBL_train_BL]
        SSTBL_test_FC_BL = SSTBL_FC[SSTBL_test_BL]

        SSTBL_FC_fu2 = np.load('/home/jiaty/SST_dataset/SSTFU2common_FC.npy')
        SSTBL_train_FC_fu2 = SSTBL_FC_fu2[SSTBL_train_fu2]
        SSTBL_test_FC_fu2 = SSTBL_FC_fu2[SSTBL_test_fu2]

        SSTBL_FC_fu3 = np.load('/home/jiaty/SST_dataset/SSTFU3common_FC.npy')
        SSTBL_train_FC_fu3 = SSTBL_FC_fu3[SSTBL_train_fu3]
        SSTBL_test_FC_fu3 = SSTBL_FC_fu3[SSTBL_test_fu3]

        finnal_train_FC_data = np.concatenate(( SSTBL_train_FC_fu2, SSTBL_train_FC_fu3,SSTBL_train_FC_BL), axis=0)
        finnal_test_FC_data = np.concatenate((SSTBL_test_FC_fu2, SSTBL_test_FC_fu3,SSTBL_test_FC_BL), axis=0)

        print("finnal_test_FC_data", finnal_test_FC_data.shape)
        print("finnal_train_FC_data", finnal_train_FC_data.shape)

    if args.using_vbm:
        SSTBL_vbm = np.load('/home/jiaty/SST_dataset/SSTBLVBM.npy')
        SSTBL_vbm_train = SSTBL_vbm[SSTBL_train_BL]
        SSTBL_vbm_test = SSTBL_vbm[SSTBL_test_BL]

        SST_fu2_vbm = np.load('/home/jiaty/SST_dataset/SSTFU2VBM.npy')
        SST_fu2_vbm_train=SST_fu2_vbm[SSTBL_train_fu2]
        SST_fu2_vbm_test=SST_fu2_vbm[SSTBL_test_fu2]

        SST_fu3_vbm = np.load('/home/jiaty/SST_dataset/SSTFU3VBM.npy')
        SST_fu3_vbm_train = SST_fu3_vbm[SSTBL_train_fu3]
        SST_fu3_vbm_test = SST_fu3_vbm[SSTBL_test_fu3]


        finnal_train_vbm_data=np.concatenate((SST_fu2_vbm_train,SST_fu3_vbm_train,SSTBL_vbm_train),axis=0)
        finnal_test_vbm_data=np.concatenate((SST_fu2_vbm_test,SST_fu3_vbm_test,SSTBL_vbm_test),axis=0)


    if args.using_seedfc:
        SSTBL_seedfc=np.load("/home/jiaty/SST_dataset/BL_SSTcommon_seedFC_64_4.npy")
        SSTBL_seedfc=SSTBL_seedfc[:,args.seedfc_id,:,:,:]
        SSTBL_seedfc_train=SSTBL_seedfc[SSTBL_train_BL]
        SSTBL_seedfc_test=SSTBL_seedfc[SSTBL_test_BL]

        SST_fu2_seedfc = np.load('/home/jiaty/SST_dataset/FU2_SSTcommon_seedFC_64_4.npy')
        SST_fu2_seedfc=SST_fu2_seedfc[:,args.seedfc_id,:,:,:]
        SST_fu2_seedfc_train=SST_fu2_seedfc[SSTBL_train_fu2]
        SST_fu2_seedfc_test=SST_fu2_seedfc[SSTBL_test_fu2]

        SST_fu3_seedfc = np.load('/home/jiaty/SST_dataset/FU3_SSTcommon_seedFC_64_4.npy')
        SST_fu3_seedfc=SST_fu3_seedfc[:,args.seedfc_id,:,:,:]
        SST_fu3_seedfc_train=SST_fu3_seedfc[SSTBL_train_fu3]
        SST_fu3_seedfc_test=SST_fu3_seedfc[SSTBL_test_fu3]

        finnal_train_seedfc_data = np.concatenate(( SST_fu2_seedfc_train, SST_fu3_seedfc_train,SSTBL_seedfc_train,), axis=0)
        finnal_test_seedfc_data = np.concatenate(( SST_fu2_seedfc_test, SST_fu3_seedfc_test,SSTBL_seedfc_test,), axis=0)

    if args.using_dti:
        SSTBL_dti = np.load('/home/jiaty/SSTDTI/DTI_BL.npy')
        SSTBL_dti_train=SSTBL_dti[SSTBL_train_BL]
        SSTBL_dti_test=SSTBL_dti[SSTBL_test_BL]

        SSTFU2_dti = np.load('/home/jiaty/SSTDTI/DTI_FU2.npy')
        SST_fu2_dti_train=SSTFU2_dti[SSTBL_train_fu2]
        SST_fu2_dti_test=SSTFU2_dti[SSTBL_test_fu2]

        SSTFU3_dti = np.load('/home/jiaty/SSTDTI/DTIFU3.npy')
        SST_fu3_dti_train=SSTFU3_dti[SSTBL_train_fu3]
        SST_fu3_dti_test=SSTFU3_dti[SSTBL_test_fu3]

        finnal_train_dti_data = np.concatenate(( SST_fu2_dti_train, SST_fu3_dti_train,SSTBL_dti_train), axis=0)
        finnal_test_dti_data = np.concatenate((SST_fu2_dti_test, SST_fu3_dti_test,SSTBL_dti_test), axis=0)

    # final_train_data = np.concatenate(( SSTBL_train_data_fu2, SSTBL_train_data_fu3,SSTBL_train_data_BL), axis=0)
    # finnal_test_data = np.concatenate(( SSTBL_test_data_fu2, SSTBL_test_data_fu3,SSTBL_test_data_BL), axis=0)
    # finnal_train_FC_data = np.concatenate(( SSTBL_train_FC_fu2, SSTBL_train_FC_fu3,SSTBL_train_FC_BL), axis=0)
    # finnal_test_FC_data = np.concatenate(( SSTBL_test_FC_fu2, SSTBL_test_FC_fu3,SSTBL_test_FC_BL,), axis=0)
    # finnal_train_vbm_data = np.concatenate(( SST_fu2_vbm_train, SST_fu3_vbm_train,SSTBL_vbm_train), axis=0)
    # finnal_test_vbm_data = np.concatenate((SST_fu2_vbm_test, SST_fu3_vbm_test,SSTBL_vbm_test), axis=0)






    # print("finnal_test_FC_data", finnal_test_FC_data.shape)
    # print("finnal_train_FC_data", finnal_train_FC_data.shape)
    if args.using_mask:
        # SSTBLgs_mask = nib.load('/home/jiaty/SST_dataset/SSTBLgsmask.nii').get_fdata()
        # SSTBLss_mask = nib.load('/home/jiaty/SST_dataset/SSTBLssmask.nii').get_fdata()
        # SSTfu2gs_mask = nib.load('/home/jiaty/SST_dataset/SSTFU2gsmask.nii').get_fdata()
        # SSTfu2ss_mask = nib.load('/home/jiaty/SST_dataset/SSTFU2ssmask.nii').get_fdata()
        # SSTfu3gs_mask = nib.load('/home/jiaty/SST_dataset/SSTFU3gsmask.nii').get_fdata()
        # SSTfu3ss_mask = nib.load('/home/jiaty/SST_dataset/SSTFU3ssmask.nii').get_fdata()
        # mask_list = [SSTfu2ss_mask, SSTfu3ss_mask,SSTBLss_mask]
        SSTBLmask=nib.load('/home/jiaty/SST_dataset/motor_control_uniformity-test_z_FDR_0.01_3mm.nii.gz').get_fdata()
        SSTBLmask=SSTBLmask==0
        SSTFU2mask=nib.load('/home/jiaty/SST_dataset/motor_control_uniformity-test_z_FDR_0.01_3mm.nii.gz').get_fdata()
        SSTFU2mask=SSTFU2mask==0
        SSTFU3mask = nib.load('/home/jiaty/SST_dataset/motor_control_uniformity-test_z_FDR_0.01_3mm.nii.gz').get_fdata()
        SSTFU3mask = SSTFU3mask == 0
        mask_list = [SSTBLmask, SSTFU2mask, SSTFU3mask]
    frimdata_test=My3DDataset_inferece(final_train_data,finnal_test_data,finnal_test_FC_data,finnal_test_vbm_data,finnal_test_seedfc_data,finnal_test_dti_data,
                                       1,[0],0.1,mask_list,args.self_mask_rate,using_fc=args.using_FC,number_list=num_list,
                                       using_mask=args.using_mask,mask_rate=args.mask_rate,using_seedfc=args.using_seedfc
                                       ,using_dti=args.using_dti,norm=args.using_norm,region_mask_id=args.region_mask_id)
    frim_test_dataloader=DataLoader(frimdata_test,batch_size=args.batch_size,shuffle=False)

    return frim_test_dataloader,#fu2_mean,fu3_mean,num_list
    #, train_data,test_data #frim_train_dataloader,

def load_model(args,checkpoint_path,device):
    ckpt_path =checkpoint_path
    contextinpue_chans = args.condition_num
    if args.using_FC:
        contextinpue_chans = args.condition_num + 1
    if args.using_vbm:
        contextinpue_chans=contextinpue_chans+1
    if args.using_dti:
        contextinpue_chans = contextinpue_chans + 1
    if args.using_seedfc:
        contextinpue_chans = contextinpue_chans + 1

    nnet = UViT(img_size=64, patch_size=args.patch_size, in_chans=1, contextinpue_chans=contextinpue_chans,embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
                mlp_ratio=4.,
                qkv_bias=False, qk_scale=None, norm_layer=torch.nn.LayerNorm, mlp_time_embed=False,num_clip_token=512,
                use_checkpoint=args.use_checkpoint, conv=True, skip=True)
    nnet.to(device)
    state_dict = torch.load(os.path.join(ckpt_path, 'model_ep'+str(args.resumeEP)+'_iter'+str(args.resumeITER)+'.pth'),'cpu')
    load_weights_dict = {k: v for k, v in state_dict.items()
                         if "fc" not in k and nnet.state_dict()[k].numel() == v.numel()}
    nnet.load_state_dict(load_weights_dict, strict=False)

    nnet.eval()
    score_model_test = sde.ScoreModel(nnet, pred=args.pred_mode, sde=sde.VPSDE())


    nparam = 0
    for p in nnet.parameters():
        nparam = nparam + np.prod(p.shape)
    print('Number of param = ' + str(nparam/1000000))
    return score_model_test



def cfg_nnet(x, t, **kwargs):

    _cond = score_model_test.noise_pred(x, t, **kwargs)
    if cfg_scale!=1.0:
        _empty_concat = torch.zeros_like(kwargs['context'], device=device)
        kwargs['context'] = _empty_concat
        _uncond = score_model_test.noise_pred(x, t, **kwargs)
        return _cond + cfg_scale * (_cond - _uncond)
    else:
        return _cond

def sample_fn(sample_steps=20, _x_init=None, context = None,FC=None):
    # _x_init = torch.randn(_x_init.shape, device=device)
    kwargs = dict(context=context,FC=FC)
    noise_schedule = NoiseScheduleVP(schedule='linear')
    model_fn = model_wrapper(
        cfg_nnet,
        noise_schedule,
        time_input_type='0',
        model_kwargs=kwargs
    )
    dpm_solver = DPM_Solver(model_fn, noise_schedule)
    return dpm_solver.sample(
        _x_init,
        steps=sample_steps,
        eps=1e-4,
        adaptive_step_size=False,
        fast_version=True)


def inference_pretrained_model(args,frim_test_dataloader,save_path,device,region_mask_id):
    naverge = 1
    samples_all = []
    for ii in range(0, naverge):
        for input_ in frim_test_dataloader:
            for _batch, y, fc in input_:
                real_data = _batch.to(device)  # [64,1,64,64,64]
                con_all = y.to(device)  # [128,3,64,64]
                print("real data",real_data.shape)
                print("con_all",con_all.shape)#([64, 3, 64, 64, 64])
                # y = _batch[1][:,:,0].to(device) + 1
                if args.inference_mode=="train_mean":
                    inference_input=real_data
                else:
                    inference_input = torch.randn((_batch.shape[0], 1, _batch.shape[2], _batch.shape[3], _batch.shape[4]), device=device)
                samples = sample_fn(sample_steps=args.nstep, _x_init=inference_input,context=con_all,FC=fc.to(device)) #cfg_scale=args.cfg_scale,,
                print("sample",samples.shape) #sample torch.Size([157, 1, 64, 64, 64])
                samples=samples.detach().cpu()
                samples=samples[:,0,:,:,:] #[64, 64, 64]
                samples_all.append(samples)
        print("sample all",np.array(samples_all).shape)
        samples_all=np.array(samples_all)
        # samples_all[:num_list[0],:,:,:]=samples_all[:num_list[0]]+fu2_mean
        # samples_all[num_list[0]:,:,:,:]=samples_all[num_list[0]:,:,:,:]+fu3_mean
        if args.region_mask_id is not None:
            np.save(os.path.join(save_path, "test_prediction_SST_non_motor_control_naverge_{}_{}.npy".format(naverge,region_mask_id)), samples_all)
        else:
            np.save(os.path.join(save_path,"test_prediction_SST_non_motor_control_naverge_{}.npy".format(naverge)),samples_all)

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    setup_seed(34007)
    main_path="/home/jiaty/SST_task"
    main_path_checkpoint = "/home/jiaty/SST_task/checkpoint/BLFU23_predict_stopsuccess_by_gosuccess_seedfc_0/"
    file_name='2024-05-21-17-02-27'
    checkpoint_path=os.path.join(main_path_checkpoint,file_name)
    print(file_name)
    parser = cli_parser()
    args = parser.parse_args('')
    main_path_inference=main_path_checkpoint.replace("checkpoint",'inference')
    if os.path.exists(main_path_inference) is False:
        os.makedirs(main_path_inference)
    save_path=os.path.join(main_path_inference,file_name)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    save_path_current=os.path.join(save_path,"cfg_"+str(args.cfg_scale)+"_nstep_"+str(args.nstep)+"_inference_mode:"+str(args.inference_mode))
    inference_log='prem_'+ args.pred_mode +  'bs_' + str(args.batch_size)+'ps_' + str(args.patch_size) + 'dep_' + str(
        args.depth) + 'embd_' + str(args.embed_dim) + 'nh_' + str(args.num_heads) + 'epo_' + str(args.nepoch) + 'lr_' +str(args.lr)\
              +'_usingFC_'+ str(args.using_FC)+'_using_mask_'+str(args.using_mask)+"_using_self_mask"+str(args.self_mask_rate)\
              +"_using_vbm"+str(args.using_vbm)+"_using_dti"+str(args.using_dti)+"_using_seedfc"+str(args.using_seedfc)+"_seedfc_id"+str(args.seedfc_id)\
              +"_using_norm"+str(args.using_norm)+"cfg_"+str(args.cfg_scale)+"_nstep_"+str(args.nstep)+"_inference_mode:"+str(args.inference_mode)+"_region_mask_id"+str(args.region_mask_id)
    if os.path.exists(save_path_current) is False:
        os.makedirs(save_path_current)
    # save_path=os.path.join(save_path,)
    # if os.path.exists(save_path) is False:
    #     os.makedirs(save_path)
    import torch

    # if torch.cuda.is_available():
    #     print(f"有 {torch.cuda.device_count()} 个GPU可用")
    #     for i in range(torch.cuda.device_count()):
    #         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    # else:
    #     print("没有可用的GPU")

    device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    # image_mask, image_mask1=load_mask()
    for i in range(1): #1:51
        print("region mask id",i)
        frim_test_dataloader=load_data(args,region_mask_id=i)
        score_model_test=load_model(args,checkpoint_path,device)

        global cfg_scale
        cfg_scale = args.cfg_scale
        inference_pretrained_model(args, frim_test_dataloader,save_path_current, device,region_mask_id=i)
        print("done!!!")




