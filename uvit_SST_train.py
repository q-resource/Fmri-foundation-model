import random
# import h5py
import numpy as np
import sys

import pandas as pd
import torch
import nibabel as nib
#sys.path.append('/users/win-fmrib-analysis/eov316/wgong/ukbiobank/ukb_braindiffusion/src/U-ViT/')
sys.path.append('/home/jiaty/U_VIT/')
from gongcode.models import UViT
# from dataloaders import load_Nifti_data
from torch.utils.data import Dataset, DataLoader
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import os
import sde
import accelerate
from torch import multiprocessing as mpi
from datetime import datetime
from fmridataset import *
import signal
import urllib

import argparse
def cli_parser():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-lr', default=0.0001,type=int, help='Learning rate')
    parser.add_argument('-batch_size', default=64,type=int, help='Batch size')
    parser.add_argument('-nepoch', default=400,type=int,help='Number of epochs')
    parser.add_argument('-depth', default=10, type=int,help='Number of epochs')
    parser.add_argument('-embed_dim', default=512,type=int, help='Number of epochs')
    parser.add_argument('-patch_size', default=8,type=int, help='Number of epochs')
    parser.add_argument('-pred_mode', default='noise_pred',type=str, help='Number of epochs')
    parser.add_argument('-num_heads', default=8, type=int,help='Number of epochs')
    parser.add_argument('-use_checkpoint', default=True,type=bool, help='Number of epochs')
    parser.add_argument('-resumeEP', default=0,type=int, help='Number of epochs')
    parser.add_argument('-resumeITER', default=0,type=int, help='Number of epochs')
    parser.add_argument('--num_workers',default=6,type=int)
    parser.add_argument('--using_FC', default=True, type=bool)
    parser.add_argument('--condition_num', default=1, type=int)
    parser.add_argument('--p_uncond', default=0.05, type=int)
    parser.add_argument('--using_mask', default=False, type=bool)
    parser.add_argument('--using_vbm', default=False, type=bool)
    parser.add_argument('--using_dti', default=False, type=bool)
    parser.add_argument('-self_mask_rate', default=None, type=int, help='if using all set None')
    parser.add_argument('--using_seedfc',default=None,type=bool,help='using seedfc or not')
    parser.add_argument('--seedfc_id', default=0, type=int, help='the seedfc of region id list(range(1, 101))')
    parser.add_argument('--using_norm', default=False, type=bool, help='using normalized or not')
    parser.add_argument('-mask_rate', default=0.01, type=float, help='mask_rate')
    return parser




def load_data(args):

    global finnal_train_FC_data,finnal_train_vbm_data,finnal_train_seedfc_data,finnal_train_dti_data,mask_list
    finnal_train_FC_data, finnal_train_vbm_data, finnal_train_seedfc_data, finnal_train_dti_data, mask_list=0,0,0,0,0
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

    # Combine all data first
    all_data = np.concatenate((SSTBL_train_data_BL, SSTBL_train_data_fu2, SSTBL_train_data_fu3,
                              SSTBL_test_data_BL, SSTBL_test_data_fu2, SSTBL_test_data_fu3), axis=0)
    
    # Calculate split sizes
    total_size = len(all_data)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Create random splits
    train_data, val_data, test_data = torch.utils.data.random_split(
        all_data, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Store indices for consistent splitting of other data types
    train_indices = train_data.indices
    val_indices = val_data.indices
    test_indices = test_data.indices
    
    # Convert back to numpy arrays
    final_train_data = all_data[train_indices]
    final_val_data = all_data[val_indices]
    final_test_data = all_data[test_indices]

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

        # Combine all FC data
        all_FC_data = np.concatenate((SSTBL_train_FC_BL, SSTBL_train_FC_fu2, SSTBL_train_FC_fu3,
                                     SSTBL_test_FC_BL, SSTBL_test_FC_fu2, SSTBL_test_FC_fu3), axis=0)
        
        # Split FC data using same indices
        finnal_train_FC_data = all_FC_data[train_indices]
        finnal_val_FC_data = all_FC_data[val_indices]
        finnal_test_FC_data = all_FC_data[test_indices]

        print("finnal_train_FC_data", finnal_train_FC_data.shape)
        print("finnal_val_FC_data", finnal_val_FC_data.shape)
        print("finnal_test_FC_data", finnal_test_FC_data.shape)

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

        # Combine all VBM data
        all_vbm_data = np.concatenate((SSTBL_vbm_train, SST_fu2_vbm_train, SST_fu3_vbm_train,
                                      SSTBL_vbm_test, SST_fu2_vbm_test, SST_fu3_vbm_test), axis=0)
        
        # Split VBM data using same indices
        finnal_train_vbm_data = all_vbm_data[train_indices]
        finnal_val_vbm_data = all_vbm_data[val_indices]
        finnal_test_vbm_data = all_vbm_data[test_indices]

    if args.using_seedfc:
        SSTBL_seedfc=np.load("/home/jiaty/SST_dataset/BL_SSTcommon_seedFC_64_4.npy")
        print("SSTBL_seedfc",SSTBL_seedfc.shape)
        SSTBL_seedfc=SSTBL_seedfc[:,args.seedfc_id,:,:,:]
        SSTBL_seedfc_train=SSTBL_seedfc[SSTBL_train_BL]
        SSTBL_seedfc_test=SSTBL_seedfc[SSTBL_test_BL]

        SST_fu2_seedfc = np.load('/home/jiaty/SST_dataset/FU2_SSTcommon_seedFC_64_4.npy')
        print("SST_fu2_seedfc",SST_fu2_seedfc.shape)
        SST_fu2_seedfc=SST_fu2_seedfc[:,args.seedfc_id,:,:,:]
        SST_fu2_seedfc_train=SST_fu2_seedfc[SSTBL_train_fu2]
        SST_fu2_seedfc_test=SST_fu2_seedfc[SSTBL_test_fu2]

        SST_fu3_seedfc = np.load('/home/jiaty/SST_dataset/FU3_SSTcommon_seedFC_64_4.npy')
        print("SST_fu3_seedfc", SST_fu3_seedfc.shape)
        SST_fu3_seedfc=SST_fu3_seedfc[:,args.seedfc_id,:,:,:]
        SST_fu3_seedfc_train=SST_fu3_seedfc[SSTBL_train_fu3]
        SST_fu3_seedfc_test=SST_fu3_seedfc[SSTBL_test_fu3]

        # Combine all seedfc data
        all_seedfc_data = np.concatenate((SSTBL_seedfc_train, SST_fu2_seedfc_train, SST_fu3_seedfc_train,
                                         SSTBL_seedfc_test, SST_fu2_seedfc_test, SST_fu3_seedfc_test), axis=0)
        
        # Split seedfc data using same indices
        finnal_train_seedfc_data = all_seedfc_data[train_indices]
        finnal_val_seedfc_data = all_seedfc_data[val_indices]
        finnal_test_seedfc_data = all_seedfc_data[test_indices]

        print("finnal_train_seedfc_data", finnal_train_seedfc_data.shape)
        print("finnal_val_seedfc_data", finnal_val_seedfc_data.shape)
        print("finnal_test_seedfc_data", finnal_test_seedfc_data.shape)

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

        # Combine all DTI data
        all_dti_data = np.concatenate((SSTBL_dti_train, SST_fu2_dti_train, SST_fu3_dti_train,
                                      SSTBL_dti_test, SST_fu2_dti_test, SST_fu3_dti_test), axis=0)
        
        # Split DTI data using same indices
        finnal_train_dti_data = all_dti_data[train_indices]
        finnal_val_dti_data = all_dti_data[val_indices]
        finnal_test_dti_data = all_dti_data[test_indices]



    num_list = []
    num_list.append(SSTBL_train_data_BL.shape[0])
    num_list.append(SSTBL_train_data_fu2.shape[0])
    num_list.append(SSTBL_train_data_fu3.shape[0])
    print("train number", num_list)  # [1034, 892, 754]
    print("final_train_data",final_train_data.shape)
    print("finnal_test_data",finnal_test_data.shape)

    if args.using_mask:
        SSTBLgs_mask = nib.load('/home/jiaty/SST_dataset/SSTBLgsmask.nii').get_fdata()
        SSTBLss_mask = nib.load('/home/jiaty/SST_dataset/SSTBLssmask.nii').get_fdata()
        SSTfu2gs_mask = nib.load('/home/jiaty/SST_dataset/SSTFU2gsmask.nii').get_fdata()
        SSTfu2ss_mask = nib.load('/home/jiaty/SST_dataset/SSTFU2ssmask.nii').get_fdata()
        SSTfu3gs_mask = nib.load('/home/jiaty/SST_dataset/SSTFU3gsmask.nii').get_fdata()
        SSTfu3ss_mask = nib.load('/home/jiaty/SST_dataset/SSTFU3ssmask.nii').get_fdata()

        mask_list = [SSTBLgs_mask, SSTfu2gs_mask, SSTfu3gs_mask]


    frimdata_train=My3DDataset_target(final_train_data,finnal_train_FC_data,finnal_train_vbm_data,finnal_train_seedfc_data,finnal_train_dti_data,1,[0],
                                      args.p_uncond,mask_list,num_list,args.self_mask_rate,using_FC=args.using_FC,
                                      using_mask=args.using_mask,using_VBM=args.using_vbm,using_seedfc=args.using_seedfc,using_dti=args.using_dti,norm=args.using_norm)
    # frimdata_test=My3DDataset(test_data,test_FC,2,0.1,)
    #ukb_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    frim_train_dataloader=DataLoader(frimdata_train,batch_size=args.batch_size,shuffle=True)
    # frim_test_dataloader=DataLoader(frimdata_test,batch_size=args.batch_size,shuffle=False)
    print("finish loading data")
    return frim_train_dataloader#,frim_test_dataloader


import copy
from utils import ema
#num_class=3
def main(args,frim_train_dataloader,ckpt_path):
    contextinpue_chans=args.condition_num
    if args.using_FC:
        contextinpue_chans=contextinpue_chans+1
    if args.using_vbm:
        contextinpue_chans=contextinpue_chans+1
    if args.using_dti:
        contextinpue_chans = contextinpue_chans + 1
    if args.using_seedfc:
        contextinpue_chans = contextinpue_chans + 1
    nnet = UViT(img_size=64, patch_size=args.patch_size, in_chans=1, contextinpue_chans=contextinpue_chans,embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
                mlp_ratio=4.,
                qkv_bias=False, qk_scale=None, norm_layer=torch.nn.LayerNorm, mlp_time_embed=False,
                num_clip_token=512,

                use_checkpoint=False, conv=True, skip=True)


    nnet.train()
    nnet.to(device)
    nnet_ema = copy.deepcopy(nnet).to(device)
    nnet_ema.eval()
    nnet_ema.requires_grad_(False)

    # from ddpm_code.optimizer import SophiaG
    # optimizer = torch.optim.AdamW(nnet.parameters(),lr = lr,weight_decay = 0.01)
    # optimizer = SophiaG(nnet.parameters(), lr=lr, weight_decay=0.01, bs=batch_size, betas=(0.965, 0.99), rho=0.01)
    optimizer = torch.optim.AdamW(nnet.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.99, 0.99))

    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,first_cycle_steps=args.nepoch * 10000,
                                                 cycle_mult=1.0,
                                                 max_lr=args.lr,
                                                 min_lr=0.000001,
                                                 warmup_steps=2000,
                                                 gamma=1.0)
    ##accelerate
    #nnet, optimizer, ukb_loader, lr_scheduler = accelerator.prepare(
    #        nnet, optimizer, ukb_loader, lr_scheduler)
    score_model = sde.ScoreModel(nnet, pred=args.pred_mode, sde=sde.VPSDE())
    score_model_ema = sde.ScoreModel(nnet_ema, pred=args.pred_mode, sde=sde.VPSDE())

    nparam = 0
    for p in nnet.parameters():

        nparam = nparam + np.prod(p.shape)
    print('Number of param = ' + str(nparam / 1000000))



    train_loss = []
    for ep in range(0, args.nepoch):
        for batch_idx, (_batch,y,FC) in enumerate(frim_train_dataloader):

            optimizer.zero_grad()

            loss = sde.LSimple(score_model, _batch.to(device), pred=args.pred_mode,context=y.to(device),FC=FC.to(device)).mean()#y=null
            #print("loss",loss.squeeze().item())#loss 1.093326449394226
            # print(loss)
            #accelerator.backward(loss)
            loss.backward()
            #accelerator.clip_grad_norm_(nnet.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(nnet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()


            if torch.isnan(loss.detach()):
                break

            ema(nnet_ema, nnet, 0.9999)

            if batch_idx % 2000 == 0:
                torch.save(nnet.state_dict(),
                           os.path.join(ckpt_path, 'model_ep' + str(ep) + '_iter' + str(batch_idx) + '.pth'))
                torch.save(nnet_ema.state_dict(),
                           os.path.join(ckpt_path, 'EMAmodel_ep' + str(ep) + '_iter' + str(batch_idx) + '.pth'))

            if batch_idx % 5 == 0:
                loss_print = loss.detach().item()

                train_loss.append(loss_print)
                print('epoch:' + str(ep) + ',Iter: ' + str(batch_idx) + ', Loss:' + str(loss_print))
                np.save(os.path.join(ckpt_path, 'train_loss.npy'), np.array(train_loss))

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass
if __name__ == '__main__':
    parser = cli_parser()
    args = parser.parse_args()
    time_now = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
    print("running time is",time_now)

    # device = torch.device("cuda:6,7" if torch.cuda.is_available() else "cpu")
    main_path="/home/jiaty/SST_task/checkpoint"
    if os.path.exists(main_path) is False:
        os.makedirs(main_path)
    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    # device = torch.device("cuda:6,7" if torch.cuda.is_available() else "cpu")
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    file_info=str(time_now) + 'prem_'+ args.pred_mode +  'bs_' + str(args.batch_size)+'ps_' + str(args.patch_size) + 'dep_' + str(
        args.depth) + 'embd_' + str(args.embed_dim) + 'nh_' + str(args.num_heads) + 'epo_' + str(args.nepoch) + 'lr_' +str(args.lr)\
              + 'p_' +str(args.p_uncond)+'_usingFC_'+ str(args.using_FC)+'_using_mask_'+str(args.using_mask)\
              +"_using_self_mask_"+str(args.self_mask_rate)+'uisng_seedfc'+str(args.using_seedfc)+'uisng_dti'+str(args.using_dti)+'random_mask'+str(args.mask_rate)

    print("model training config: ",file_info)
    save_file_name = "BLFU23_predict_stopsuccess_by_gosuccess"
    training_info = "train_test8:2_" + save_file_name

    if args.using_seedfc:
        training_info=training_info+"_seedfc_"+str(args.seedfc_id)
        save_file_name=save_file_name+"_seedfc_"+str(args.seedfc_id)
    if args.using_dti:
        training_info = training_info + "_dti"
        save_file_name = save_file_name + "_dti"
    if args.using_FC:
        training_info = training_info + "_FC"
        save_file_name = save_file_name + "_FC"
    if args.using_vbm:
        training_info = training_info + "_vbm"
        save_file_name = save_file_name + "_vbm"
    if args.self_mask_rate:
        training_info = training_info + "_indipercent"
        save_file_name = save_file_name + "_indipercent"
    if args.mask_rate:
        training_info = training_info + "_randommask"
        save_file_name = save_file_name + "_randommask"


    print("training_info:",training_info)
    save_model_path=os.path.join(main_path,save_file_name)
    if os.path.exists(save_model_path) is False:
        os.makedirs(save_model_path)
    ckpt_path = os.path.join(save_model_path,str(time_now))
    os.system('mkdir ' + ckpt_path)
    with open(os.path.join(ckpt_path,'train_log.txt'), 'w') as file:
        file.write(file_info+training_info)

    frim_train_dataloader=load_data(args)
    main(args,frim_train_dataloader,ckpt_path)

    # torch.maiultiprocessing.spawn(main, (args,), args.ngpus_per_node)
    print("done!!!")
