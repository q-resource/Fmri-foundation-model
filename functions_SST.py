import numpy as np
import pandas as pd
import os
import nibabel as nib
from sklearn.preprocessing import StandardScaler
import matplotlib
from matplotlib import pyplot as plt
# from gongcode.inference_data import load_mask
matplotlib.use('Agg')
import torch
from scipy import stats as stats
from scipy.io import savemat
from einops import rearrange, repeat
import random
import sys
import seaborn as sns
from scipy.io import loadmat
import h5py
from scipy.stats import pearsonr
def load_mask():
    image_mask_name = '/home/jiaty/U_VIT/Dataset/MNI152_T1_3mm_brain_mask.nii'  # 61，73，61
    # image_mask1 = nib.load(image_mask_name).get_fdata()
    # image_mask = torch.FloatTensor(nib.load(image_mask_name).get_fdata().flatten()) > 0
    # #
    # image_mask_name = '/home/jiaty/U_VIT/Dataset/MNI152_T1_3mm_brain_mask.nii'  # 61，73，61
    # image_mask1 = nib.load(image_mask_name).get_fdata()
    # image_mask1 = torch.FloatTensor(np.expand_dims(np.expand_dims(image_mask1, 0), 0))
    #
    #
    # image_mask1 = torch.nn.functional.interpolate(image_mask1, (64, 64, 64), mode='nearest')[0, 0, :, :,
    #               :]  # 【batchsize，chnnal,64,64,64】
    # print("mask shape", image_mask1.shape)
    image_mask=nib.load(image_mask_name).get_fdata()
    sub_contrast = image_mask[:, 5:69, :]
    data_zero = np.zeros((64 - 61, 64, 61))
    sub_contrast = np.concatenate((data_zero, sub_contrast), axis=0)
    data_expanded_second = np.zeros((64, 64, 64))
    data_expanded_second[:, :, :61] = sub_contrast
    image_mask64 = data_expanded_second
    print("image_mask",image_mask.shape)
    print("image_mask64",image_mask64.shape)
    return image_mask,image_mask64
def preprocess_data_add_zero_reshape_to_64():

    save_path = '/home/jiaty/SST_dataset'
    data=np.load('/home/jiaty/SST_dataset/FU3_SSTcommon_seedFC.npy')
    # index_data_fu2=pd.read_csv("/home/jiaty/UVIT_dataset/FU3afterQCtrainindex.csv")
    # # data_fu2 = all_data_fu2 [:, 0:4, :, :, :]
    # index_data_fu2['Index'] = index_data_fu2['Index'].astype(int)
    # # print(index_data["Index"].unique())
    # train_index_fu2 = index_data_fu2[index_data_fu2['Index'] == 1].index
    print(data.shape)
    all_sub=[]
    for sub in range(data.shape[0]):#data.shape[0]
        sub_i=data[sub,:,:,:,:]
        sub_i=np.nan_to_num(sub_i,nan=0)
        sub_i_list=[]
        print(sub)
        for contrast in range(sub_i.shape[0]):#sub_i.shape[0]
            # print("sub",sub,"contrast",contrast)
            sub_contrast=sub_i[contrast,:,:,:]
            sub_contrast = sub_contrast[ :, 5:69, :]
            data_zero = np.zeros((64 - 61, 64, 61))
            sub_contrast = np.concatenate((data_zero, sub_contrast), axis=0)
            data_expanded_second = np.zeros((64, 64, 64))
            data_expanded_second[:, :, :61] =sub_contrast
            sub_contrast=data_expanded_second
            # sub_contrast = ((sub_contrast - np.min(sub_contrast)) / (np.max(sub_contrast) - np.min(sub_contrast)))
            sub_i_list.append(sub_contrast)
        all_sub.append(sub_i_list)
    print("all sub",np.array(all_sub).shape)#all sub (1157, 56, 64, 64, 64)
    all_sub=np.array(all_sub)
    np.save(os.path.join(save_path,"FU3_SSTcommon_seedFC_64.npy"),all_sub)

def plot_lossCurve(save_path,inference_path):
    # set figsize
    #path="/home/jiaty/UVIT_ckeackpoint_{}/{}/train_loss.npy".format(save_path.split("/")[-1],file_name)
    path = inference_path.replace('inference', "checkpoint")
    path = os.path.dirname(path)
    path = os.path.join(path, 'train_loss.npy')
    print(path)
    # path="/home/jiaty/UVIT_ckeackpoint_N_BL23/2024-04-09-14-43-24_noise_pred_64_8_10_512_8_400_0.0001/train_loss.npy"
    train_loss_list = np.load(path)
    fig = plt.figure(figsize=(5, 4), dpi=150)
    # fig = plt.figure(figsize=figsize)
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, 'r', lw=1)  # lw为曲线宽度
    # val_loss_lines = plt.plot(x, val_loss_list, 'b', lw=1)
    plt.title("loss")
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.legend(["train_loss",
                ])
    plt.savefig(os.path.join(save_path,"train_loss.jpg"),dpi=300)
    plt.show()
    plt.close()


def mask_predict_data_add_zero(path,save_path):
    # path="/home/jiaty/UVIT_dataset/test_prediction_157_naverage_3_FC.npy" #3 157 64 64 64 FC
    naverge = int(path.split(".npy")[0].split("_")[-1])
    # print("naverge",naverge)
    # save_path="/home/jiaty/UVIT_dataset"
    predict_data=np.load(path)
    print("predict data", predict_data.shape)#predict data (1, 585, 64, 64, 64)
    predict_data_mean=predict_data.mean(axis=0)
    np.save(path.replace(".npy","_mean.npy"),np.array(predict_data_mean))
    # print("predict_data_mean",predict_data_mean.shape)
    # predict_data=predict_data[0,:,:,:,:]

    # image_mask, image_mask1=load_mask()
    image_mask_name = '/home/jiaty/U_VIT/Dataset/MNI152_T1_3mm_brain_mask.nii'  # 61，73，61
    image_mask1 = nib.load(image_mask_name).get_fdata()
    # print("image_mask1.shape",image_mask1.shape)

    predicted_masked_all=[]
    for i in range(predict_data_mean.shape[0]):
        predict_i=predict_data_mean[i,:,:,:]  #[64,64,64]
        predict_i=predict_i[:,:,:61]
        predict_i=predict_i[3:,:,:]
        data_expanded_second = np.zeros((61, 73, 61))
        data_expanded_second[:,5:69,:]=predict_i
        predict_i=data_expanded_second

        # sub_contrast = torch.FloatTensor(np.expand_dims(np.expand_dims(predict_i, 0), 0))
        # # 计算每个维度的插值倍数
        # target_shape = (61, 73, 61)
        # resized_data = torch.nn.functional.interpolate(sub_contrast, size=target_shape, mode='nearest')[0, 0, :, :, :] #bilinear bicubic linear area
        # predict_i=resized_data.numpy()#[61,73,61]
        index=image_mask1<1
        predict_i[index]=np.NAN
        predicted_masked_all.append(predict_i)
    # print("predicted_masked_all",np.array(predicted_masked_all).shape) # (157, 61, 73, 61)
    save_path_masked=os.path.join(save_path,"predicted_masked_all_naverage_{}_mean.npy".format(naverge))
    np.save(save_path_masked,np.array(predicted_masked_all))
    naverge_list=[]
    for n in range(naverge):
        predict_n=predict_data[n,:,:,:,:]
        predicted_masked_all = []
        for i in range(predict_n.shape[0]):
            predict_i = predict_n[i, :, :, :]
            predict_i = predict_i[:, :, :61]
            predict_i = predict_i[3:, :, :]
            data_expanded_second = np.zeros((61, 73, 61))
            data_expanded_second[:, 5:69, :] = predict_i
            predict_i = data_expanded_second
            # print(predict_i.shape)

            # sub_contrast = torch.FloatTensor(np.expand_dims(np.expand_dims(predict_i, 0), 0))
            # # 计算每个维度的插值倍数
            # target_shape = (61, 73, 61)
            # resized_data = torch.nn.functional.interpolate(sub_contrast, size=target_shape, mode='nearest')[0, 0, :, :,
            #                :]
            # predict_i = resized_data.numpy()  # [61,73,61]
            index = image_mask1 < 1
            predict_i[index] = np.NAN
            predicted_masked_all.append(predict_i)
        naverge_list.append(predicted_masked_all)
        # print("predicted_masked_all", np.array(predicted_masked_all).shape)  # (157, 61, 73, 61)
    print("predicted_masked_all each", np.array(naverge_list).shape)  # (3,157, 61, 73, 61)

    save_path_masked_each = os.path.join(save_path, "predicted_masked_all_naverage_{}_each.npy".format(naverge))
    # print(save_path_masked_each)
    np.save(save_path_masked_each, np.array(naverge_list))
    return save_path_masked
def show_example(path,save_path,index_sub_all):
    each_path=path.replace("mean.npy","each.npy")
    test_index=index_sub_all[index_sub_all['index']==0]
    test_index_list=test_index['Sub_ID'].values.tolist()
    # print("each_path",each_path)
    data_mean=np.load(path)
    name=path.split("/")[-1].split("_mean.npy")[0]
    # savemat('/home/jiaty/UVIT_dataset/predicted_masked_all_157.mat', {'data': data})
    # print(data_mean.shape)
    save_path_nii=os.path.join(save_path,"predicted_condition_masked")
    if os.path.exists(save_path_nii) is False:
        os.makedirs(save_path_nii)
    nii_save = os.path.join(save_path_nii, name)
    if os.path.exists(nii_save) is False:
        os.makedirs(nii_save)
    for i in range(data_mean.shape[0]):
        data_i=data_mean[i,:,:,:]
        # nifti_img = nib.Nifti1Image(data_i, )
        home_jiaty = '/home/jiaty/UVIT_dataset'
        img = nib.load(os.path.join(home_jiaty, 'con_happy.nii'))
        # nii1 = nib.Nifti1Image(data_i, new_affine)#
        # new_spacing = [3, 3, 3]
        # img.header.set_zooms(new_spacing)
        img = nib.Nifti1Image(data_i, img.affine)

        # non_nan_count = np.count_nonzero(~np.isnan(data_i))
        # print(non_nan_count)

        nib.save(img, os.path.join(nii_save,"output_{}_mean.nii".format(test_index_list[i])))
    naverge = int(path.split("_mean.npy")[0].split("_")[-1])
    naverge_list=[]
    data_each=np.load(each_path)
    print("data_each",data_each.shape) #data_each (585, 61, 73, 61)
    for n in range(naverge):
        data_n=data_each[n,:,:,:,:]
        for i in range(data_n.shape[0]):
            data_i = data_n[i, :, :, :]
            # nifti_img = nib.Nifti1Image(data_i, )
            home_jiaty = '/home/jiaty/UVIT_dataset'
            img = nib.load(os.path.join(home_jiaty, 'con_happy.nii'))
            # nii1 = nib.Nifti1Image(data_i, new_affine)#
            # new_spacing = [3, 3, 3]
            # img.header.set_zooms(new_spacing)
            img = nib.Nifti1Image(data_i, img.affine)
            nib.save(img, os.path.join(nii_save, "output_{}_naverage_{}.nii".format(test_index_list[i],n)))

def load_data_fu23():
    #[203, 184, 198]
    all_data_fu2 = np.load("/home/jiaty/UVIT_dataset/FU2_sub_1157_56_post.npy")
    # FC_fu2=np.load("/home/jiaty/UVIT_dataset/FU2_1157_rest_FC.npy")
    index_data_fu2=pd.read_csv("/home/jiaty/UVIT_dataset/FU2afterQCtrainindex.csv")
    data_fu2 = all_data_fu2 [:, 2, :, :, :]
    index_data_fu2['Index'] = index_data_fu2['Index'].astype(int)
    # print(index_data["Index"].unique())
    train_index_fu2 = index_data_fu2[index_data_fu2['Index'] == 1].index
    # print(len(train_index)) #932
    test_index_fu2 = index_data_fu2[index_data_fu2['Index'] == 0].index
    # train_FC_fu2 = FC_fu2[train_index_fu2]
    # test_FC_fu2 = FC_fu2[test_index_fu2]
    train_data_fu2 = data_fu2[train_index_fu2]
    test_data_fu2 = data_fu2[test_index_fu2]

    all_data_fu3 = np.load("/home/jiaty/UVIT_dataset/FU3_sub_1038_56_post.npy")
    # FC_fu3=np.load("/home/jiaty/UVIT_dataset/FU3_1038_rest_FC.npy")
    index_data_fu3=pd.read_csv("/home/jiaty/UVIT_dataset/FU3afterQCtrainindex.csv")
    data_fu3 = all_data_fu3[:, 2, :, :, :]
    # print(all_data.shape)#(1157, 4, 64, 64, 64)
    index_data_fu3['Index']=index_data_fu3['Index'].astype(int)
    # print(index_data["Index"].unique())
    train_index_fu3=index_data_fu3[index_data_fu3['Index']==1].index
    # print(len(train_index)) #932
    test_index_fu3=index_data_fu3[index_data_fu3['Index']==0].index
    # print(len(test_index)) #225
    train_data_fu3=data_fu3[train_index_fu3]
    test_data_fu3=data_fu3[test_index_fu3]
    # train_FC_fu3=FC_fu3[train_index_fu3]
    # test_FC_fu3=FC_fu3[test_index_fu3]

    final_train_data=np.concatenate((train_data_fu2, train_data_fu3), axis=0)
    finnal_test_data=np.concatenate((test_data_fu2,test_data_fu3),axis=0)
    # finnal_train_FC_data=np.concatenate((train_FC_fu2,train_FC_fu3),axis=0)
    # finnal_test_FC_data=np.concatenate((test_FC_fu2,test_FC_fu3),axis=0)
    print("final_train_data",final_train_data.shape)
    print("finnal_test_data",finnal_test_data.shape)
    # print("finnal_test_FC_data",finnal_test_FC_data.shape)
    # print("finnal_train_FC_data",finnal_train_FC_data.shape)
    return final_train_data,finnal_test_data

def load_data_fu23baseline(target):
    index_SSTBL = pd.read_csv("/home/jiaty/SST_dataset/SSTBLtraintestindex.csv")
    sub_list_BL = pd.read_csv("/home/jiaty/SST_dataset/SSTBLcommonfiles.csv")
    sub_list_BL['Sub_ID'] = sub_list_BL['Sub_ID'].apply(lambda x: x[2:-2])
    print(sub_list_BL['Sub_ID'].head())
    index_SSTBL_fu2 = pd.read_csv("/home/jiaty/SST_dataset/SSTFU2traintestindex.csv")
    sub_list_fu2 = pd.read_csv("/home/jiaty/SST_dataset/SSTFU2commonfiles.csv")
    sub_list_fu2["Sub_ID"] = sub_list_fu2["Sub_ID"].apply(lambda x: x.split("'")[0])
    index_SST_fu3 = pd.read_csv("/home/jiaty/SST_dataset/SSTFU3traintestindex.csv")
    sub_list_fu3 = pd.read_csv("/home/jiaty/SST_dataset/SSTFU3commonfiles.csv")
    sub_list_fu3["Sub_ID"] = sub_list_fu3["Sub_ID"].apply(lambda x: x.split("'")[0])
    index_all = pd.concat([index_SSTBL_fu2, index_SST_fu3], axis=0)
    index_all = pd.concat([index_all, index_SSTBL], axis=0)
    sub_all = pd.concat([sub_list_fu2, sub_list_fu3], axis=0)
    sub_all = pd.concat([sub_all, sub_list_BL], axis=0)
    index_sub_all = pd.concat([index_all, sub_all], axis=1)

    SSTBL = np.load('/home/jiaty/SST_dataset/SSTBLcommon_post.npy')
    SSTBL=SSTBL[:,target,:,:,:]
    SSTBL_FC = np.load('/home/jiaty/SST_dataset/SSTBLcommon_FC.npy')
    index_SSTBL = pd.read_csv("/home/jiaty/SST_dataset/SSTBLtraintestindex.csv")
    index_SSTBL['index'] = index_SSTBL['index'].astype(int)
    SSTBL_train_BL = index_SSTBL[index_SSTBL['index'] == 1].index
    SSTBL_test_BL = index_SSTBL[index_SSTBL['index'] == 0].index
    SSTBL_train_data_BL = SSTBL[SSTBL_train_BL]
    SSTBL_test_data_BL = SSTBL[SSTBL_test_BL]
    SSTBL_train_FC_BL = SSTBL_FC[SSTBL_train_BL]
    SSTBL_test_FC_BL = SSTBL_FC[SSTBL_test_BL]

    SSTBL_fu2 = np.load('/home/jiaty/SST_dataset/SSTFU2common_post.npy')
    SSTBL_fu2=SSTBL_fu2[:,target,:,:,:]
    SSTBL_FC_fu2 = np.load('/home/jiaty/SST_dataset/SSTFU2common_FC.npy')
    index_SSTBL_fu2 = pd.read_csv("/home/jiaty/SST_dataset/SSTFU2traintestindex.csv")
    index_SSTBL_fu2['index'] = index_SSTBL_fu2['index'].astype(int)
    SSTBL_train_fu2 = index_SSTBL_fu2[index_SSTBL_fu2['index'] == 1].index
    SSTBL_test_fu2 = index_SSTBL_fu2[index_SSTBL_fu2['index'] == 0].index
    SSTBL_train_data_fu2 = SSTBL_fu2[SSTBL_train_fu2]
    SSTBL_test_data_fu2 = SSTBL_fu2[SSTBL_test_fu2]
    SSTBL_train_FC_fu2 = SSTBL_FC_fu2[SSTBL_train_fu2]
    SSTBL_test_FC_fu2 = SSTBL_FC_fu2[SSTBL_test_fu2]

    SSTBL_fu3 = np.load('/home/jiaty/SST_dataset/SSTFU3common_post.npy')
    SSTBL_fu3=SSTBL_fu3[:,target,:,:,:]
    SSTBL_FC_fu3 = np.load('/home/jiaty/SST_dataset/SSTFU3common_FC.npy')
    index_SSTBL_fu3 = pd.read_csv("/home/jiaty/SST_dataset/SSTFU3traintestindex.csv")
    index_SSTBL_fu3['index'] = index_SSTBL_fu3['index'].astype(int)
    SSTBL_train_fu3 = index_SSTBL_fu3[index_SSTBL_fu3['index'] == 1].index
    SSTBL_test_fu3 = index_SSTBL_fu3[index_SSTBL_fu3['index'] == 0].index
    SSTBL_train_data_fu3 = SSTBL_fu3[SSTBL_train_fu3]
    SSTBL_test_data_fu3 = SSTBL_fu3[SSTBL_test_fu3]
    SSTBL_train_FC_fu3 = SSTBL_FC_fu3[SSTBL_train_fu3]
    SSTBL_test_FC_fu3 = SSTBL_FC_fu3[SSTBL_test_fu3]

    test_num_list=[]
    test_num_list.append(SSTBL_test_data_fu2.shape[0])
    test_num_list.append(SSTBL_test_data_fu3.shape[0])

    train_num_list=[]
    train_num_list.append(SSTBL_train_data_fu2.shape[0])
    train_num_list.append(SSTBL_train_data_fu3.shape[0])
    final_train_data=np.concatenate((SSTBL_train_data_fu2,SSTBL_train_data_fu3,SSTBL_train_data_BL),axis=0)
    finnal_test_data=np.concatenate((SSTBL_test_data_fu2,SSTBL_test_data_fu3,SSTBL_test_data_BL),axis=0)
    # finnal_train_FC_data=np.concatenate((train_FC_fu2,train_FC_fu3),axis=0)
    # finnal_test_FC_data=np.concatenate((test_FC_fu2,test_FC_fu3),axis=0)
    print("final_train_data", final_train_data.shape) #final_train_data (3022, 64, 64, 64)
    print("finnal_test_data", finnal_test_data.shape) #finnal_test_data (729, 64, 64, 6

    return final_train_data,finnal_test_data,test_num_list,train_num_list,index_sub_all
# def computed_corr_inference_test(path,save_path,test_data,type,datasets_name='fu23baseline',split=[225,424]):
#     print("-----------computed 646464 inference corr test-----------")
#     inference = np.load(path)
#     print("inference",inference.shape)
#     print("test_data",test_data.shape)
#     inference = inference[0, :, :, :, :]
#     if type=="all":
#         inference=inference
#         test_data=test_data
#     if type=='fu2':
#         inference=inference[:split[0],:,:,:] #[:108,:,:,:] [:225,:,:,:]
#         test_data=test_data[:split[0],:,:,:]  #[:108,:,:,:]
#     if type=='fu3':
#         if datasets_name=='fu23':
#             inference-inference[split[0]:,:,:,:] #[108:,:,:,:][225:,:,:,:]
#             test_data=test_data[split[0]:,:,:,:] #[108:,:,:,:][225:,:,:,:]
#         else:
#             inference = inference[split[0]:split[1],:,:,:]#[108:205,:,:,:][225:424, :, :, :]
#             test_data = test_data[split[0]:split[1],:,:,:]#[108:205,:,:,:][225:424, :, :, :]
#     if type == 'baseline' and datasets_name=='fu23baseline':
#         inference = inference[split[1]:,:,:,:] #[205,:,:,;,:][424:, :, :, :]
#         test_data = test_data[split[1]:,:,:,:] #[424:, :, :, :] #[205,:,:,;,:]
#     image_mask, image_mask1 = load_mask()
#     # image_mask1 = image_mask1.numpy()
#     index = image_mask1 > 0
#     inference_all = []
#     test_data_all = []
#     for i in range(inference.shape[0]):
#         #print("predict[i, :, :, :]",predict[i, :, :, :].shape)
#         inference_i = inference[i, :, :, :]
#         inference_i=inference_i[index]
#         inference_i=inference_i.flatten()
#         inference_all.append(inference_i)
#         #print("test_data[i, :, :, :]",test_data[i, :, :, :].shape)
#         test_data_i=test_data[i, :, :, :]
#         test_data_i=test_data_i[index]
#         test_data_i=test_data_i.flatten()
#         test_data_all.append(test_data_i)
#     rr_list=[]
#     for i in range(test_data.shape[0]):
#         #print(len(test_data_all[i]))
#         #print(len(predict_all[i]))
#         rr=np.corrcoef(test_data_all[i],inference_all[i])[0, 1]
#         rr_list.append(rr)
#
#     rr_pd=pd.DataFrame(np.array(rr_list))
#     print(rr_pd.describe())
#     max_index = np.argmax(rr_list)
#     print("the most corr",rr_list[max_index])
#     print("the most corr index",max_index)
#     np.save(os.path.join(save_path,"preidict_corr_real_byself_{}.npy".format(type)),np.array(rr_list))
#     plt.hist(np.array(rr_list), color='skyblue', edgecolor='black') #bins=30,
#     plt.title('Histogram of preidict_corr_real_byself_{}'.format(type))
#     plt.xlabel('Value')
#     plt.ylabel('Corr')
#     plt.grid(True)
#     plt.savefig(os.path.join(save_path,"preidict_corr_real_byself_noise_{}.jpg".format(type)),dpi=300)
#     plt.show()
#     plt.close()
def compute_dice_iou(test_data, inference):
    # Initialize arrays to store dice and iou values
    dice = np.zeros(21)
    iou = np.zeros(21)

    # Loop over different percentiles
    for idx, p in enumerate(np.arange(5, 100, 5)):
        # Calculate the threshold values based on non-symmetric percentiles
        lower_percentile_test = np.percentile(np.abs(test_data), p / 2)
        upper_percentile_test = np.percentile(np.abs(test_data), 100 - p / 2)
        lower_percentile_inference = np.percentile(np.abs(inference), p / 2)
        upper_percentile_inference = np.percentile(np.abs(inference), 100 - p / 2)

        # Thresholding
        test_data_thresh = np.where((test_data > lower_percentile_test) & (test_data < upper_percentile_test), 0, 1)

        # Calculate True Positives for each threshold
        TP_A = np.sum(test_data_thresh == 1)

        # Same thresholds for inference
        inference_thresh = np.where((inference > lower_percentile_inference) & (inference < upper_percentile_inference),
                                    0, 1)
        TP_B = np.sum(inference_thresh == 1)

        # Calculate True Positives for the intersection of test_data and inference
        TP = np.sum(test_data_thresh * inference_thresh)

        # Calculate Dice coefficient
        dice[idx] = 2 * TP / (TP_A + TP_B)

        # Calculate Intersection over Union (IoU)
        iou[idx] = TP / np.sum((test_data_thresh | inference_thresh) == 1)


    return dice, iou


def computed_corr_inference_test(path, save_path, test_data, type, datasets_name='fu23baseline', split=[225, 424]):
    print("-----------computed 646464 inference corr test-----------")
    #print("inference", inference.shape)
    print("test_data", test_data.shape)
    inference = np.load(path)
    print("inference", inference.shape)
    print("test_data", test_data.shape)
    inference = inference[0, :, :, :, :]

    if type == "all":
        pass
    elif type == 'fu2':
        inference = inference[:split[0], :, :, :]
        test_data = test_data[:split[0], :, :, :]
    elif type == 'fu3':
        if datasets_name == 'fu23':
            inference = inference[split[0]:, :, :, :]
            test_data = test_data[split[0]:, :, :, :]
        else:
            inference = inference[split[0]:split[1], :, :, :]
            test_data = test_data[split[0]:split[1], :, :, :]
    elif type == 'baseline' and datasets_name == 'fu23baseline':
        inference = inference[split[1]:, :, :, :]
        test_data = test_data[split[1]:, :, :, :]

    image_mask, image_mask1 = load_mask()
    index = image_mask1 > 0

    inference_all = []
    test_data_all = []

    dice_list = []
    iou_list = []
    rr_list = []

    for i in range(inference.shape[0]):
        inference_i = inference[i, :, :, :]
        inference_i = inference_i[index]
        inference_i = inference_i.flatten()
        inference_all.append(inference_i)

        test_data_i = test_data[i, :, :, :]
        test_data_i = test_data_i[index]
        test_data_i = test_data_i.flatten()
        test_data_all.append(test_data_i)

        # Calculate Dice coefficients and IoU values
        dice, iou = compute_dice_iou(test_data_i, inference_i)
        dice_list.append(dice)
        iou_list.append(iou)

        # Calculate correlation coefficient
        rr = np.corrcoef(test_data_i, inference_i)[0, 1]
        rr_list.append(rr)

    dice_array = np.array(dice_list)
    iou_array = np.array(iou_list)
    rr_array = np.array(rr_list)

    # Saving Dice coefficients, IoU values, and correlation coefficients
    np.save(os.path.join(save_path, "dice_coefficients_{}.npy".format(type)), dice_array)
    np.save(os.path.join(save_path, "iou_values_{}.npy".format(type)), iou_array)
    np.save(os.path.join(save_path, "correlation_coefficients_{}.npy".format(type)), rr_array)

    rr_pd = pd.DataFrame(rr_array)
    print(rr_pd.describe())
    max_index = np.argmax(rr_list)
    print("the most corr", rr_list[max_index])
    print("the most corr index", max_index)

    plt.hist(rr_array, color='skyblue', edgecolor='black')
    plt.title('Histogram of correlation coefficients {}'.format(type))
    plt.xlabel('Value')
    plt.ylabel('Corr')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "correlation_histogram_{}.jpg".format(type)), dpi=300)
    plt.show()
    plt.close()

    # 画出每个人的21个Dice系数和IoU值并保存
    for i in range(dice_array.shape[1]):
        # 绘制Dice系数的直方图
        plt.hist(dice_array[:, i], bins=10, color='skyblue', edgecolor='black')
        plt.title('Histogram of Dice coefficients {}'.format(type))
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, "dice_histogram_person{}_{}.jpg".format(i, type)), dpi=300)
        plt.show()
        plt.close()

        # 绘制IoU值的直方图
        plt.hist(iou_array[:, i], bins=10, color='skyblue', edgecolor='black')
        plt.title('Histogram of IoU values {}'.format(type))
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, "iou_histogram_person{}_{}.jpg".format(i, type)), dpi=300)
        plt.show()
        plt.close()

    corr_matrix = np.zeros((test_data.shape[0], test_data.shape[0]))

    # 计算相关性并填充相关性矩阵
    for i in range(test_data.shape[0]):
        for j in range(test_data.shape[0]):
            corr = np.corrcoef(test_data_all[i], inference_all[j])[0, 1]
            corr_matrix[i, j] = corr

    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='viridis', origin='lower')
    plt.colorbar(label='Correlation')
    plt.title('Correlation Matrix of each_predict_corr_each_test')
    plt.xlabel('Column Index')
    plt.ylabel('Column Index')
    plt.savefig(os.path.join(save_path, "each_predict_corr_each_test_{}.jpg".format(type)), dpi=300)
    # plt.show()
    plt.close()


def computed_corr_inference_train_mean(path,save_path,train_data,type,datasets_name='fu23baseline'):
    inference = np.load(path)
    inference = inference[0, :, :, :, :]
    if type == "all":
        inference = inference
        train_data=train_data
    if type == 'fu2':
        inference = inference[:225, :, :, :] #[:108,:,:,:]
        train_data=train_data[:932,:,:,:]   #[1049,:,:,:]
    if type == 'fu3':
        if datasets_name=='fu23':
            inference = inference[225:, :, :, :]#[108:,:,:,:]
            train_data=train_data[932,:,:,:]    #[1049:,:,:,:]
        else:
            inference = inference[225:424, :, :, :] #[108:205,:,:,:]
            train_data = train_data[932:1771 :, :, :]#[1049:1990,:,:,:]
    if type=='baseline' and datasets_name=='fu23baseline':
        inference=inference[424:,:,:,:]        #[205,:,:,:]
        train_data=train_data[1771:,:,:,:]     #[1990,:,:,:]
    image_mask, image_mask1 = load_mask()
    image_mask1 = image_mask1.numpy()
    index = image_mask1 > 0
    train_data_flatten_all = []

    for i in range(train_data.shape[0]):
        train_data_i = train_data[i, :, :, :]
        # test_data_i=(test_data_i-np.mean(test_data_i))/np.std(test_data_i)
        train_data_i = train_data_i[index]
        train_data_i = train_data_i.flatten()
        train_data_flatten_all.append(train_data_i)
    # print("test_data_flatten_all",np.array(train_data_flatten_all).shape)
    train_data_mean = np.array(train_data_flatten_all).mean(axis=0)
    # print("test_data_mean",len(train_data_mean))

    rr_list = []
    for i in range(inference.shape[0]):
        inference_i = inference[i, :, :, :]
        # predict_i=(predict_i-np.mean(predict_i))/np.std(predict_i)
        inference_i = inference_i[index]
        inference_i = inference_i.flatten()

        rr = np.corrcoef(train_data_mean, inference_i)[0, 1]
        rr_list.append(rr)
    rr_pd = pd.DataFrame(np.array(rr_list))
    print(rr_pd.describe())
    max_index = np.argmax(rr_list)
    print("the most corr", rr_list[max_index])
    print("the most corr index", max_index)
    np.save(os.path.join(save_path, "inference_corr_train_Mean_{}.npy".format(type)), np.array(rr_pd))

    plt.hist(np.array(rr_list), color='skyblue', edgecolor='black')  # bins=30,
    plt.title('Histogram of inference_corr_train_Mean_{}'.format(type))
    plt.xlabel('Value')
    plt.ylabel('Corr')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "inference_corr_train_Mean_{}.jpg".format(type)), dpi=300)
    plt.show()
    plt.close()
def computed_corr_train_mean_test(save_path,train_data,test_data,type,datasets_name='fu23baseline',split=[225,424],train_split=[932,1771]):
    print("")
    if type == "all":
        test_data = test_data
        train_data=train_data
    if type == 'fu2':
        test_data = test_data[:split[0], :, :, :]
        train_data=train_data[:train_split[0],:,:,:]
    if type == 'fu3':
        if datasets_name=='fu23':
            test_data = test_data[split[0]:, :, :, :]
            train_data = train_data[train_split[0]:, :, :, :]
        else:
            test_data = test_data[split[0]:split[1], :, :, :]
            train_data = train_data[train_split[0]:train_split[1], :, :, :]
    if type=='baseline':
        if datasets_name=='fu23baseline':
            test_data=test_data[split[1]:,:,:,:]
            train_data=train_data[train_split[1]:,:,:,:]
    print("test_data",test_data.shape)
    print("train_data",train_data.shape)

    image_mask, image_mask1 = load_mask()
    # image_mask1 = image_mask1.numpy()
    index = image_mask1 > 0
    train_data_flatten_all = []
    for i in range(train_data.shape[0]):
        train_data_i = train_data[i, :, :, :]
        # test_data_i=(test_data_i-np.mean(test_data_i))/np.std(test_data_i)
        train_data_i = train_data_i[index]
        train_data_i = train_data_i.flatten()
        train_data_flatten_all.append(train_data_i)
    # print("test_data_flatten_all", np.array(train_data_flatten_all).shape)
    train_data_mean = np.array(train_data_flatten_all).mean(axis=0)
    rr_list = []
    for i in range(test_data.shape[0]):
        test_data_i = test_data[i, :, :, :]
        # predict_i=(predict_i-np.mean(predict_i))/np.std(predict_i)
        test_data_i = test_data_i[index]
        test_data_i = test_data_i.flatten()

        rr = np.corrcoef(train_data_mean, test_data_i)[0, 1]
        rr_list.append(rr)
    rr_pd = pd.DataFrame(np.array(rr_list))
    print(rr_pd.describe())
    max_index = np.argmax(rr_list)
    print("the most corr", rr_list[max_index])
    print("the most corr index", max_index)
    np.save(os.path.join(save_path, "all_real_mean_corr_each_real_{}.npy".format(type)), np.array(rr_pd))
    plt.hist(np.array(rr_list), color='skyblue', edgecolor='black')  # bins=30,

    # 添加标题和标签
    plt.title('Histogram of all_real_mean_corr_each_real_{}'.format(type))
    plt.xlabel('Value')
    plt.ylabel('Corr')
    # 显示网格线
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "all_real_mean_corr_each_real_{}.jpg".format(type)), dpi=300)
    # 显示图形
    plt.show()
    plt.close()
def computed_corr_inference_inference(path,save_path,type='fu2',datasets_name='fu23',split=[225,424]):
    print("-----------computed 646464 inference corr inference-----------")
    inference = np.load(path)
    print("inference", inference.shape)
    inference = inference[0, :, :, :, :]
    if type == "all":
        inference = inference
        # test_data = test_data
    if type == 'fu2':
        inference = inference[:split[0], :, :, :]  # [:108,:,:,:] [:225,:,:,:]
        # test_data = test_data[:108, :, :, :]  # [:108,:,:,:]
    if type == 'fu3':
        if datasets_name == 'fu23':
            inference - inference[split[1]:, :, :, :]  # [108:,:,:,:][225:,:,:,:]
            # test_data = test_data[:108, :, :, :]  # [108:,:,:,:][225:,:,:,:]
        else:
            inference = inference[split[0]:split[1], :, :, :]  # [108:205,:,:,:][225:424, :, :, :]
            # test_data = test_data[108:205, :, :, :]  # [108:205,:,:,:][225:424, :, :, :]
    if type == 'baseline' and datasets_name == 'fu23baseline':
        inference = inference[split[1]:, :, :, :, :]  # [205,:,:,;,:][424:, :, :, :]
        # test_data = test_data[205, :, :, :, :]  # [424:, :, :, :] #[205,:,:,;,:]
    image_mask, image_mask1 = load_mask()
    # image_mask1 = image_mask1.numpy()
    index = image_mask1 > 0
    inference_all = []
    # test_data_all = []
    for i in range(inference.shape[0]):
        # print("predict[i, :, :, :]",predict[i, :, :, :].shape)
        inference_i = inference[i, :, :, :]
        inference_i = inference_i[index]
        inference_i = inference_i.flatten()
        inference_all.append(inference_i)
        # print("test_data[i, :, :, :]",test_data[i, :, :, :].shape)
        # test_data_i = test_data[i, :, :, :]
        # test_data_i = test_data_i[index]
        # test_data_i = test_data_i.flatten()
        # test_data_all.append(test_data_i)
    rr_list = []
    print("inference_all",np.array(inference_all).shape)#[225 67683]
    inference_all=np.array(inference_all)
    correlation_matrix=np.corrcoef(inference_all)

    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='viridis', origin='lower')
    plt.colorbar(label='Correlation')
    plt.title('Correlation Matrix of predict_corr_predict')
    plt.xlabel('Column Index')
    plt.ylabel('Column Index')
    plt.savefig(os.path.join(save_path, "predict_corr_predict_{}.jpg".format(type)), dpi=300)
    plt.show()

def show_rr_rr(save_path):
    # rr=np.load(os.path.join(save_path,"prediction_corr_real_byself.npy"))
    rr=np.load("/home/jiaty/UVIT_result_updata/2024-03-25-23-22-30_noise_pred_8_10_512_8_300_0.0005/prediction_corr_real_byself_Mean1000.npy")
    rr_aa=np.load("/home/jiaty/UVIT_result_updata/2024-03-14-18-47-23noise_pred_8_10_512_8_1000.001/corr_64_post.npy",)
    # rr_aa=np.load(os.path.join(save_path,"all_real_mean_corr_each_predict.npy"))
    print(rr.shape)
    # print("-------------------------------------")
    print(rr_aa.shape)
    # print("-------------------------------------")
    rr=rr.flatten()
    print(rr.shape)
    rr2=rr_aa-rr
    print(rr2)
    print(np.array(rr2).max())
    #color_list=

    plt.hist(rr2, bins=157, color='skyblue', edgecolor='black')#

    # 添加标题和标签
    plt.title('Histogram of Corr')
    plt.xlabel('Value')
    plt.ylabel('Corr')

    # 显示网格线
    plt.grid(True)
    plt.savefig(os.path.join(save_path,"all_real_1000_mean_corr_each_real_157-corr_64_post.jpg"),dpi=300)
    # 显示图形
    plt.show()
    plt.close()


def computed_FC():
    import os
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from einops import rearrange, reduce
    import nibabel as nib

    # 假设文件夹名称存储在csv文件中
    folder_names = pd.read_csv('folder_names.csv')['folder_name'].tolist()
    # folder_names=['Preprocessed_sub-000000112288_ses-followup2_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz']
    all_subject=[]
    for folder_name in folder_names:
        # 构造文件路径
        file_path = f'/public/mig_old_storage/home1/ISTBI_data/IMAGEN_New_Preprocessed/Prepressed/fmriprep/{folder_name}/ses-followup2/func/Preprocessed_{folder_name}_ses-followup2_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'
        # file_path=os.path.join('/home/jiaty/UVIT_dataset',folder_name)
        # 加载Nifti文件
        img = nib.load(file_path)
        data = img.get_fdata()

        # 替换nan值为0
        data = np.nan_to_num(data)

        # 将数据转换为PyTorch张量
        # data_tensor = torch.from_numpy(data)

        # 使用最近邻插值将数据插值为64x64x64
        data_sampled_all=np.zeros([64,64,64,data.shape[-1]])
        # print(data_sampled_all.shape)
        for i in range(data.shape[-1]):
            data_i=data[:,:,:,i]
            # print(data_i.shape) #(61, 73, 61)
            data_i=np.expand_dims(np.expand_dims(data_i,0),0)
            data_i_tensor=torch.from_numpy(data_i)
            # print(data_i.shape)
            data_upsampled = F.interpolate(data_i_tensor, size=(64, 64, 64), mode='nearest')[0,0,:,:,:]
            data_upsampled = data_upsampled.numpy()
            data_sampled_all[:,:,:,i]=data_upsampled

        # 使用einops重塑和分块
        print(data_sampled_all.shape) #(64, 64, 64, 187)
        data_upsampled_re = rearrange(data_sampled_all, '(b1 p1) (b2 p2) (b3 p3) t -> (b1 b2 b3) (p1 p2 p3) t',b1=8,b2=8,b3=8,p1=8,p2=8,p3=8)
        print(data_upsampled_re.shape) #(512, 512, 187)
        patches = reduce(data_upsampled_re, 'a b t -> () b t', 'mean')
        print("patches.shape",patches.shape)
        patches=patches.squeeze(0)
        print("patches.shape", patches.shape) #patches.shape (512, 187)

        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(patches)
        print(corr_matrix.shape) #(187, 187)

        # 使用einops将相关系数矩阵重塑为64x64x64
        corr_tensor = rearrange(corr_matrix, '(b1 b2 b3) (p1 p2 p3) ->(b1 p1) (b2 p2) (b3 p3)', b1=8,b2=8,b3=8,p1=8,p2=8,p3=8)
        print("corr_tensor.shape",corr_tensor.shape)
        # 保存结果
        all_subject.append(corr_tensor)
    np.save(os.path.join("",'all_FC.npy'),np.array(all_subject))
        # output_path = os.path.join('/home/jiaty/UVIT_dataset','output_fc.nii.gz')
        # nib.save(nib.Nifti1Image(corr_tensor, img.affine), output_path)
    print("处理完成!")
def check_two_sample_result():
    data1=np.load("/home/jiaty/UVIT_result/2024-03-14-18-47-23noise_pred_8_10_512_8_1000.001/test_prediction_157_V2_naverge_1.npy")
    data2=np.load("/home/jiaty/UVIT_result/2024-03-14-18-47-23noise_pred_8_10_512_8_1000.001/test_prediction_157_FC_naverge_1.npy")
    # data1=data1[0,:,:,:]
    # data2=data2[0,:,:,:]
def spilt_train_test_data():
    all_data_fu2 = np.load("/home/jiaty/UVIT_dataset/FU2_sub_1157_56_post.npy")
    # FC_fu2 = np.load("/home/jiaty/UVIT_dataset/FU2_1157_rest_FC.npy")
    # index_data_fu2 = pd.read_csv("/home/jiaty/UVIT_dataset/FU2afterQCtrainindex.csv")
    data_fu2 = all_data_fu2[:, 0:4, :, :, :]
    # index_data_fu2['Index'] = index_data_fu2['Index'].astype(int)
    # # print(index_data["Index"].unique())
    # train_index_fu2 = index_data_fu2[index_data_fu2['Index'] == 1].index
    # # print(len(train_index)) #932
    # test_index_fu2 = index_data_fu2[index_data_fu2['Index'] == 0].index
    # train_FC_fu2 = FC_fu2[train_index_fu2]
    # test_FC_fu2 = FC_fu2[test_index_fu2]
    # train_data_fu2 = data_fu2[train_index_fu2]
    # test_data_fu2 = data_fu2[test_index_fu2]

    all_data_fu3 = np.load("/home/jiaty/UVIT_dataset/FU3_sub_1038_56_post.npy")
    # FC_fu3 = np.load("/home/jiaty/UVIT_dataset/FU3_1038_rest_FC.npy")
    # index_data_fu3 = pd.read_csv("/home/jiaty/UVIT_dataset/FU3afterQCtrainindex.csv")
    data_fu3 = all_data_fu3[:, 0:4, :, :, :]
    # # print(all_data.shape)#(1157, 4, 64, 64, 64)
    # index_data_fu3['Index'] = index_data_fu3['Index'].astype(int)
    # # print(index_data["Index"].unique())
    # train_index_fu3 = index_data_fu3[index_data_fu3['Index'] == 1].index
    # # print(len(train_index)) #932
    # test_index_fu3 = index_data_fu3[index_data_fu3['Index'] == 0].index
    # # print(len(test_index)) #225
    # train_data_fu3 = data_fu3[train_index_fu3]
    # test_data_fu3 = data_fu3[test_index_fu3]
    # train_FC_fu3 = FC_fu3[train_index_fu3]
    # # train_data=all_data[:1000,:,:,:]
    # test_data=all_data[1000:,:,:,:]
    # test_FC_fu3 = FC_fu3[test_index_fu3]
    # final_train_data = np.concatenate((train_data_fu2, train_data_fu3), axis=0)
    # finnal_test_data = np.concatenate((test_data_fu2, test_data_fu3), axis=0)
    # finnal_train_FC_data = np.concatenate((train_FC_fu2, train_FC_fu3), axis=0)
    # finnal_test_FC_data = np.concatenate((test_FC_fu2, test_FC_fu3), axis=0)


    all_data_baseline = np.load("/home/jiaty/UVIT_dataset/BL_sub_1556_28_post.npy")
    data_baseline = all_data_baseline[:, [0, 1,2], :, :, :]
    # FC_baseline = np.load("/home/jiaty/UVIT_dataset/BL_1556_FC.npy")
    # index_data_baseline = pd.read_csv("/home/jiaty/UVIT_dataset/BLafterQCtrainindex.csv")
    # data_baseline = all_data_baseline[:, [0, 1], :, :, :]
    # index_data_baseline['Index'] = index_data_baseline['Index'].astype(int)
    # # print(index_data["Index"].unique())
    # train_index_baseline = index_data_baseline[index_data_baseline['Index'] == 1].index
    # # print(len(train_index)) #932
    # test_index_baseline = index_data_baseline[index_data_baseline['Index'] == 0].index
    # train_FC_baseline = FC_baseline[train_index_baseline]
    # test_FC_baseline = FC_baseline[test_index_baseline]
    # train_data_baseline = data_baseline[train_index_baseline]
    # test_data_baseline = data_baseline[test_index_baseline]

    np.save("/home/jiaty/UVIT_dataset/FU2_sub_1157_56_post_anhc.npy",data_fu2)
    np.save("/home/jiaty/UVIT_dataset/FU3_sub_1038_56_post_anhc.npy",data_fu3)
    np.save("/home/jiaty/UVIT_dataset/BL_sub_1556_28_post_anc.npy", data_baseline)
    # np.save("/home/jiaty/UVIT_dataset/FU2_3_test_FC_424.npy", finnal_test_FC_data)


# finnal_train_data_base = np.concatenate((train_data_fu2, train_data_fu3, train_data_baseline), axis=0)
    # finnal_test_data_base = np.concatenate((test_data_fu2, test_data_fu3, test_data_baseline), axis=0)
    # np.save("/home/jiaty/UVIT_dataset/FU2_3_baseline_train_post_3022.npy",finnal_train_data_base)
    # np.save("/home/jiaty/UVIT_dataset/FU2_3_baseline_test_post_729.npy",finnal_test_data_base)
    # data1
def t_test():
    model_inference=np.load("/home/jiaty/UVIT_result_H_v2/2024-04-06-19-59-37_noise_pred_64_8_10_512_8_300_0.0001/inference_corr_real_byself_noise_fu2.npy")
    data_mean=np.load("/home/jiaty/UVIT_result_H_v2/2024-04-06-19-59-37_noise_pred_64_8_10_512_8_300_0.0001/all_real_mean_corr_each_real_fu2.npy")
    print(model_inference.shape)
    data_mean=data_mean.flatten()
    print(data_mean.shape)
    inference_mean=model_inference-data_mean
    t,p=stats.ttest_1samp(inference_mean,0)
    print(t)
    print(p)
    """
    N
    24.422336725277248
4.454108469490372e-65
    H
    36.92114736556701
3.288293087823815e-97
    """
def change_mat_to_npy():
    # mat=loadmat()
    # file_path = "/home/jiaty/HCP_dataset/EFTMIDBLmeancondata.mat"
    file_path='/home/jiaty/HCP_dataset/test.mat'
    home_jiaty = '/home/jiaty/UVIT_dataset'
    img = nib.load(os.path.join(home_jiaty, 'con_angry.nii'))
    # data=loadmat(file_path)
    # arr=data['data']
    # print(arr[30,35,:])
    # nii1 = nib.Nifti1Image(data_i, new_affine)#
    # new_spacing = [3, 3, 3]
    # img.header.set_zooms(new_spacing)
    #
    with h5py.File(file_path, 'r') as file:
        print(list(file.keys()))
    #
    #     # array_data = np.array(file['reduced_tensor'])
        array_data = np.array(file['data'])
    #     print(array_data.shape)# #(61, 73, 61, 2, 1017)
    #     print(array_data)
        b=array_data[:,:,:,0,0]
        print(b[30,35,:])
    #     # array_data=array_data.transpose([4,0,1,2,3])   #1027 61 73 62 2#2 1 0 反了 0 1 2错了
    #     # array_data=array_data.transpose([0,4,1,2,3])
    #     # print(array_data.shape)
    #     a=array_data[30,:,:]
    #     # print(a)
    #     print(array_data[30,35,:])
    #     print(array_data[:,35,30])
    #     b=array_data.transpose([2,1,0])
    #     print(b[30, 35, :])
    #     print(b[:, 35, 30])
        # for i in range(61):
        #     temp=a[i,:,:] #61 73
        #     a[i,:,:]=a[60-i,:,:]
        #     a[60 - i,:, :]=temp
        # print(a[:,35,30])
        # print(a.shape)
        # img = nib.Nifti1Image(a, img.affine)
        # img_b=nib.Nifti1Image(b, img.affine)
        # nib.save(img, "/home/jiaty/HCP_dataset/a.nii")
        # nib.save(img, "/home/jiaty/HCP_dataset/b.nii")
        # np.save('/home/jiaty/HCP_dataset/EFTMIDBLmeancondata.npy', array_data)
def load_data():
    SSTBL=np.load('/home/jiaty/SST_dataset/SSTBLcommon.npy')
    SSTBL_FC = np.load('/home/jiaty/SST_dataset/SSTBLcommon_FC.npy')
    index_SSTBL = pd.read_csv("/home/jiaty/SST_dataset/SSTBLtraintestindex.csv")
    print(SSTBL.shape) #(1296, 3, 61, 73, 61)
    print(SSTBL_FC.shape) #(1296, 64, 64, 64)

    print(index_SSTBL.shape)#(1296, 1)
    print(index_SSTBL['index'].sum()) #1296-1034=262
    SSTBL_filename=pd.read_csv("/home/jiaty/SST_dataset/SSTBLcommon.csv")
    print(SSTBL_filename.shape)
    SSTBL_filename['Sub_ID']=[str(i)[2:-2]for i in SSTBL_filename['Sub_ID'].values]
    SSTBL_filename_index=pd.concat([index_SSTBL,SSTBL_filename],axis=1)
    print(SSTBL_filename_index.shape)
    print(SSTBL_filename_index.head())
    SSTBL_filename_index.to_csv('/home/jiaty/SST_dataset/SSTBLtraintestindex_commonfilename.csv',index=False)

    # test_Data=nib.load("/home/jiaty/HCP_dataset/test.nii").get_fdata()
    # check_data=EFTMIDFU2mean[0,0,:,:,:]
    # home_jiaty = '/home/jiaty/UVIT_dataset'
    # img = nib.load(os.path.join(home_jiaty, 'con_angry.nii'))
    # check_img= nib.Nifti1Image(check_data, img.affine)
    # nib.save(check_img, "/home/jiaty/HCP_dataset/c.nii")
    # print(test_Data.shape)

    # print(np.array_equal(test_Data,check_data))

def reshape_mask_into_64():
    SSTBLgs_mask = nib.load('/home/jiaty/SST_dataset/SSTBLgsmask.nii').get_fdata()
    SSTBLss_mask = nib.load('/home/jiaty/SST_dataset/SSTBLssmask.nii').get_fdata()
    SSTfu2gs_mask = nib.load('/home/jiaty/SST_dataset/SSTFU2gsmask.nii').get_fdata()
    SSTfu2ss_mask = nib.load('/home/jiaty/SST_dataset/SSTFU2ssmask.nii').get_fdata()
    SSTfu3gs_mask = nib.load('/home/jiaty/SST_dataset/SSTFU3gsmask.nii').get_fdata()
    SSTfu3ss_mask = nib.load('/home/jiaty/SST_dataset/SSTFU3ssmask.nii').get_fdata()



def computed_seedfc():
    #templated_region,templated_voxel,data
    templated_region = np.random.randint(0, 101, size=(61, 73, 61))
    templated_voxel=np.random.choice([0, 1], size=(61, 73, 61))
    all_1_voxel_index=np.argwhere(templated_voxel == 1)
    print("all_1_voxel_index",all_1_voxel_index.shape) #all_1_voxel_index (135331, 3)
    data=np.random.random((61, 73, 61,100))
    region_mean=[]
    for i in range(1,101):
        region_index_i = np.argwhere(templated_region==i)
        print("(2693, 3)",region_index_i.shape) #(2692, 3)
        data_region_i=data[region_index_i[:,0],region_index_i[:,1],region_index_i[:,2],:] #
        data_region_i_mean=np.mean(data_region_i,axis=0)
        print(data_region_i_mean.shape) #(100,)
        region_mean.append(data_region_i_mean)
    print("region_mean",np.array(region_mean).shape)
    seed_fc=np.zeros_like(templated_voxel)
    for region_T_i in region_mean:
        print(region_T_i.shape)#(100,)
        for voxel_j in range(all_1_voxel_index.shape[0]):
            data_voxel=data[all_1_voxel_index[voxel_j,0],all_1_voxel_index[voxel_j,1],all_1_voxel_index[voxel_j,2],:]
            print(data_voxel.shape)#(100,)
            t,p=np.corrcoef(region_T_i,data_voxel)[0]
            print(t,p)
            seed_fc[all_1_voxel_index[voxel_j,0],all_1_voxel_index[voxel_j,1],all_1_voxel_index[voxel_j,2]]=t

#239*239 64*64*64

def computed_map_test_and_inference(test_data,inference_path,save_path):
    inference=np.load(inference_path)
    inference=inference[0,:,:,:,:]
    image_mask, image_mask1 = load_mask()
    index = np.where(image_mask1==1)
    correlation_matrix=np.zeros_like(image_mask1)
    print(correlation_matrix.shape)
    # print("np.array(index).shape",np.array(index).shape) # (3, 67683)
    # print(index[0][0],index[1][0],index[2][0])
    # print(correlation_matrix[index[0][0],index[1][0],index[2][0]])
    # print(test_data[:,index[0][0],index[1][0],index[2][0]])
    for i in range(np.array(index).shape[1]):
        t,p=pearsonr(test_data[:,index[0][i],index[1][i],index[2][i]],inference[:,index[0][i],index[1][i],index[2][i]])
        correlation_matrix[index[0][i], index[1][i], index[2][i]]=t
    print("correlation_matrix",correlation_matrix.shape) #correlation_matrix (61, 73, 61)
    home_jiaty = '/home/jiaty/UVIT_dataset'

    correlation_matrix = correlation_matrix[:, :, :61]
    correlation_matrix = correlation_matrix[3:, :, :]
    data_expanded_second = np.zeros((61, 73, 61))
    data_expanded_second[:, 5:69, :] = correlation_matrix
    correlation_matrix=data_expanded_second
    print("correlation_matrix", correlation_matrix.shape)
    img = nib.load(os.path.join(home_jiaty, 'con_happy.nii'))
    img = nib.Nifti1Image(correlation_matrix, img.affine)
    nib.save(img, os.path.join(save_path, "all_test_corr_all_inference_voxel.nii"))
def computed_train_dataset_mean_std_map(train_data,save_path):
    image_mask, image_mask1 = load_mask()
    index = np.where(image_mask1==1)
    mean_matrix=np.zeros_like(image_mask1)
    std_matrix = np.zeros_like(image_mask1)
    for i in range(np.array(index).shape[1]):
        mean_matrix[index[0][i],index[1][i],index[2][i]]=train_data[:,index[0][i],index[1][i],index[2][i]].mean()
        std_matrix[index[0][i],index[1][i],index[2][i]]=train_data[:,index[0][i],index[1][i],index[2][i]].std()
    home_jiaty = '/home/jiaty/UVIT_dataset'
    img = nib.load(os.path.join(home_jiaty, 'con_happy.nii'))
    print("mean_matrix",mean_matrix.shape)
    print("std_matrix",std_matrix.shape)
    mean_matrix = mean_matrix[:, :, :61]
    mean_matrix = mean_matrix[3:, :, :]
    data_expanded_second = np.zeros((61, 73, 61))
    data_expanded_second[:, 5:69, :] = mean_matrix
    mean_matrix=data_expanded_second

    std_matrix = std_matrix[:, :, :61]
    std_matrix = std_matrix[3:, :, :]
    data_expanded_second = np.zeros((61, 73, 61))
    data_expanded_second[:, 5:69, :] = std_matrix
    std_matrix=data_expanded_second
    print("mean_matrix",mean_matrix.shape)
    print("std_matrix",std_matrix.shape)

    img_mean = nib.Nifti1Image(mean_matrix, img.affine)
    img_std = nib.Nifti1Image(std_matrix, img.affine)
    nib.save(img_mean, os.path.join(save_path, "all_train_voxel_mean.nii"))
    nib.save(img_std, os.path.join(save_path, "all_train_voxel_std.nii"))
def show_max_corr(rr_path,type):
    index_SSTBL = pd.read_csv("/home/jiaty/SST_dataset/SSTBLtraintestindex.csv")
    sub_list_BL = pd.read_csv("/home/jiaty/SST_dataset/SSTBLcommonfiles.csv")
    sub_list_BL['Sub_ID'] = sub_list_BL['Sub_ID'].apply(lambda x: x[2:-2])
    # print(sub_list_BL['Sub_ID'].head())
    index_SSTBL_fu2 = pd.read_csv("/home/jiaty/SST_dataset/SSTFU2traintestindex.csv")
    sub_list_fu2 = pd.read_csv("/home/jiaty/SST_dataset/SSTFU2commonfiles.csv")
    sub_list_fu2["Sub_ID"] = sub_list_fu2["Sub_ID"].apply(lambda x: x.split("'")[0])
    index_SST_fu3 = pd.read_csv("/home/jiaty/SST_dataset/SSTFU3traintestindex.csv")
    sub_list_fu3 = pd.read_csv("/home/jiaty/SST_dataset/SSTFU3commonfiles.csv")
    sub_list_fu3["Sub_ID"] = sub_list_fu3["Sub_ID"].apply(lambda x: x.split("'")[0])
    # index_all = pd.concat([index_SSTBL, index_SSTBL_fu2], axis=0)
    # index_all = pd.concat([index_all, index_SST_fu3], axis=0)
    # sub_all = pd.concat([sub_list_BL, sub_list_fu2], axis=0)
    # sub_all = pd.concat([sub_all, sub_list_fu3], axis=0)
    # print("sub_all", sub_all.shape)
    # print(sub_all.head())
    # print("index_all", index_all.shape)
    # print(index_all.head())
    # index_sub = pd.concat([index_all, sub_all], axis=1)
    target_sub_list=[]
    if type=='fu2':
        index_sub_fu2=pd.concat([index_SSTBL_fu2,sub_list_fu2],axis=1)
        test_sub_fu2 = index_sub_fu2[index_SSTBL_fu2['index'] == 0]
        target_sub_list = test_sub_fu2["Sub_ID"].values.tolist()
    if type=='fu3':
        index_sub_fu3 = pd.concat([index_SST_fu3, sub_list_fu3], axis=1)
        test_sub_fu2 = index_sub_fu2[index_SSTBL_fu2['index'] == 0]
        target_sub_list = test_sub_fu2["Sub_ID"].values.tolist()
    if type=='BL':
        index_sub_BL = pd.concat([index_SSTBL, sub_list_BL], axis=1)
        test_sub_BL = index_sub_fu2[index_SSTBL_fu2['index'] == 0]
        target_sub_list = test_sub_fu2["Sub_ID"].values.tolist()
    # print("index_sub", index_sub.head())
    # test_sub = index_sub[index_sub["index"] == 0]
    # test_sub_list = test_sub["Sub_ID"].values.tolist()
    rr_list=np.load(rr_path)
    print("all rr",rr_list.shape) #all rr (232,)
    max_index = np.argmax(rr_list)
    print("the most corr", rr_list[max_index])
    print("the most corr index", max_index)
    print("the most corr Sub_ID",target_sub_list[max_index])

def save_npy_to_nii(predict_path,save_path):
    index_SSTBL = pd.read_csv("/home/jiaty/SST_dataset/SSTBLtraintestindex.csv")
    sub_list_BL=pd.read_csv("/home/jiaty/SST_dataset/SSTBLcommonfiles.csv")
    sub_list_BL['Sub_ID']=sub_list_BL['Sub_ID'].apply(lambda x:x[2:-2])
    print(sub_list_BL['Sub_ID'].head())
    index_SSTBL_fu2 = pd.read_csv("/home/jiaty/SST_dataset/SSTFU2traintestindex.csv")
    sub_list_fu2=pd.read_csv("/home/jiaty/SST_dataset/SSTFU2commonfiles.csv")
    sub_list_fu2["Sub_ID"] = sub_list_fu2["Sub_ID"].apply(lambda x: x.split("'")[0])
    index_SST_fu3 = pd.read_csv("/home/jiaty/SST_dataset/SSTFU3traintestindex.csv")
    sub_list_fu3=pd.read_csv("/home/jiaty/SST_dataset/SSTFU3commonfiles.csv")
    sub_list_fu3["Sub_ID"] = sub_list_fu3["Sub_ID"].apply(lambda x: x.split("'")[0])
    index_all=pd.concat([index_SSTBL,index_SSTBL_fu2],axis=0)
    index_all=pd.concat([index_all,index_SST_fu3],axis=0)
    sub_all=pd.concat([sub_list_BL,sub_list_fu2],axis=0)
    sub_all=pd.concat([sub_all,sub_list_fu3],axis=0)
    print("sub_all",sub_all.shape)
    print(sub_all.head())
    print("index_all",index_all.shape)
    print(index_all.head())
    index_sub=pd.concat([index_all,sub_all],axis=1)
    print("index_sub",index_sub.head())
    test_sub=index_sub[index_sub["index"]==0]
    test_sub_list=test_sub["Sub_ID"].values.tolist()
    print("test_sub_list",test_sub_list[:4])
    print("test_sub",test_sub.shape)
    predict_n=np.load(predict_path)
    home_jiaty = '/home/jiaty/UVIT_dataset'
    img = nib.load(os.path.join(home_jiaty, 'con_happy.nii'))
    image_mask, image_mask1 = load_mask()
    mask = image_mask == 1

    predict_n_nii=np.zeros((predict_n.shape[0],61,73,61))
    for i in range(predict_n.shape[0]):
          predict_n_nii[i,mask]=predict_n[i]
    print("predict_n_nii",predict_n_nii.shape)
    save_path=os.path.join(save_path,"pred_test_nii_61_73_61")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for i in range(predict_n_nii.shape[0]):
        # predict_i_nii = predict_n_nii[i, :, :, :]
        # predict_i_nii = predict_i_nii[:, :, :61]
        # predict_i_nii = predict_i_nii[3:, :, :]
        # data_expanded_second = np.zeros((61, 73, 61))
        # data_expanded_second[:, 5:69, :] = predict_i_nii
        # predict_i_ = data_expanded_second
        predict_i_nii=predict_n_nii[i]
        predict_i_nii = nib.Nifti1Image(predict_i_nii, img.affine)
        nib.save(predict_i_nii , os.path.join(save_path, "{}.nii".format(test_sub_list[i])))

def check_template():
    import torch

    # 创建示例张量
    template = nib.load(
        "/home/jiaty/Templates/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_3mm.nii.gz").get_fdata()
    sub_contrast = template[:, 5:69, :]
    data_zero = np.zeros((64 - 61, 64, 61))
    sub_contrast = np.concatenate((data_zero, sub_contrast), axis=0)
    data_expanded_second = np.zeros((64, 64, 64))
    data_expanded_second[:, :, :61] = sub_contrast
    image_mask64 = data_expanded_second
    print(image_mask64.shape)
    # print(template.shape)

def computed_masked_region_inference_corr_test(path,save_path,test_data,type,datasets_name,split):
    masked_corr=[]
    image_mask, image_mask1 = load_mask()
    index = image_mask1 > 0

    for k in range(1,101):
        path_k=path.replace(".npy","_{}.npy".format(k))
        inference = np.load(path_k)
        inference = inference[0, :, :, :, :]
        save_path_k=os.path.join(save_path,"masked_region_{}_result".format(k))
        if os.path.exists(save_path_k) is False:
            os.makedirs(save_path_k)
        if type == "all":
            pass
        elif type == 'fu2':
            inference = inference[:split[0], :, :, :]
            test_data = test_data[:split[0], :, :, :]
        elif type == 'fu3':
            if datasets_name == 'fu23':
                inference = inference[split[0]:, :, :, :]
                test_data = test_data[split[0]:, :, :, :]
            else:
                inference = inference[split[0]:split[1], :, :, :]
                test_data = test_data[split[0]:split[1], :, :, :]
        elif type == 'baseline' and datasets_name == 'fu23baseline':
            inference = inference[split[1]:, :, :, :]
            test_data = test_data[split[1]:, :, :, :]
        inference_all = []
        test_data_all = []
        dice_list = []
        iou_list = []
        rr_list = []
        for i in range(inference.shape[0]):
            inference_i = inference[i, :, :, :]
            inference_i = inference_i[index]
            inference_i = inference_i.flatten()
            inference_all.append(inference_i)

            test_data_i = test_data[i, :, :, :]
            test_data_i = test_data_i[index]
            test_data_i = test_data_i.flatten()
            test_data_all.append(test_data_i)

            # Calculate Dice coefficients and IoU values
            # dice, iou = compute_dice_iou(test_data_i, inference_i)
            # dice_list.append(dice)
            # iou_list.append(iou)

            # Calculate correlation coefficient
            rr = np.corrcoef(test_data_i, inference_i)[0, 1]
            rr_list.append(rr)
        masked_corr.append(rr_list)
        # dice_array = np.array(dice_list)
        # iou_array = np.array(iou_list)
        rr_array = np.array(rr_list)
        # Saving Dice coefficients, IoU values, and correlation coefficients
        # np.save(os.path.join(save_path, "dice_coefficients_{}.npy".format(type)), dice_array)
        # np.save(os.path.join(save_path, "iou_values_{}.npy".format(type)), iou_array)
        np.save(os.path.join(save_path_k, "correlation_coefficients_{}.npy".format(type)), rr_array)
        rr_pd = pd.DataFrame(rr_array)
        print(rr_pd.describe())
        max_index = np.argmax(rr_list)
        print("the most corr", rr_list[max_index])
        print("the most corr index", max_index)

        plt.hist(rr_array, color='skyblue', edgecolor='black')
        plt.title('Histogram of correlation coefficients {}'.format(type))
        plt.xlabel('Value')
        plt.ylabel('Corr')
        plt.grid(True)
        plt.savefig(os.path.join(save_path_k, "correlation_histogram_{}.jpg".format(type)), dpi=300)
        # plt.show()
        plt.close()

        corr_matrix = np.zeros((test_data.shape[0], test_data.shape[0]))
        # 计算相关性并填充相关性矩阵
        for i in range(test_data.shape[0]):
            for j in range(test_data.shape[0]):
                corr = np.corrcoef(test_data_all[i], inference_all[j])[0, 1]
                corr_matrix[i, j] = corr

        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap='viridis', origin='lower')
        plt.colorbar(label='Correlation')
        plt.title('Correlation Matrix of each_predict_corr_each_test')
        plt.xlabel('Column Index')
        plt.ylabel('Column Index')
        plt.savefig(os.path.join(save_path_k, "each_predict_corr_each_test_{}.jpg".format(type)), dpi=300)
        # plt.show()
        plt.close()
    print("masked corr",np.array(masked_corr).shape) #100 256
    np.save(os.path.join(save_path,"masked_region_yeo_corr_all.npy"),np.array(masked_corr))


def computed_region_important(save_path):
    each_masked_region_corr=np.load(os.path.join(save_path,"masked_region_yeo_corr_all.npy"))
    plt.figure(figsize=(14, 6))

    # 使用 seaborn 绘制热图
    sns.heatmap(each_masked_region_corr, cmap='viridis')
    # 添加标题和标签
    # plt.title('Heatmap of (100, 232) numpy array')
    plt.xlabel('subjects')
    plt.ylabel('regions')
    # 显示图像
    plt.savefig(os.path.join(save_path,"masked_region_yeo_corr_all_heatmap.jpg"),dpi=300)
    all_region_corr=np.load(os.path.join(save_path, "correlation_coefficients_fu2_all_region.npy"))
    print("each_masked_region_corr.shape",each_masked_region_corr.shape)
    print("all_region_corr.shape",all_region_corr.shape)
    template = nib.load(
        "/home/jiaty/Templates/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_3mm.nii.gz").get_fdata()
    sub_contrast = template[:, 5:69, :]
    data_zero = np.zeros((64 - 61, 64, 61))
    sub_contrast = np.concatenate((data_zero, sub_contrast), axis=0)
    data_expanded_second = np.zeros((64, 64, 64))
    data_expanded_second[:, :, :61] = sub_contrast
    image_mask64 = data_expanded_second
    # print(np.unique(image_mask64))
    home_jiaty = '/home/jiaty/UVIT_dataset'
    img = nib.load(os.path.join(home_jiaty, 'con_happy.nii'))
    save_path_imp=os.path.join(save_path,"all_region_important")
    if os.path.exists(save_path_imp) is False:
        os.makedirs(save_path_imp)
    print(all_region_corr)
    all_region_corr_repeat=np.tile(all_region_corr, (100, 1))
    print("all_region_corr_repeat.shape",all_region_corr_repeat.shape)
    print(all_region_corr_repeat[1,:])
    all_imp=all_region_corr_repeat-each_masked_region_corr
    print("all_imp.shape",all_imp.shape)
    all_imp_mean=np.mean(all_imp, axis=0)
    all_imp_mean=all_imp_mean.tolist()
    corr_nii_mean = image_mask64.copy()
    for k in range(1, 101):
        corr_nii_mean[corr_nii_mean==k]= all_imp_mean[k-1]
    corr_nii_mean = corr_nii_mean[:, :, :61]
    corr_nii_mean = corr_nii_mean[3:, :, :]
    data_expanded_inver = np.zeros((61, 73, 61))
    data_expanded_inver[:, 5:69, :] = corr_nii_mean
    corr_nii_i_mean = data_expanded_inver
    # corr_nii_i_64=corr_nii_i
    # print(corr_nii_i_64.shape)
    corr_nii_i_mean = nib.Nifti1Image(corr_nii_i_mean, img.affine)
    nib.save(corr_nii_i_mean,os.path.join(save_path_imp,"region_important_mean"))
    # for i in range(each_masked_region_corr.shape[1]): #计算每一个人的
    #     print("each_masked_region_corr[:,i]", each_masked_region_corr[:,i].shape)
    #     corr_nii_i = image_mask64.copy()
    #     all_region_corr_list=[all_region_corr[i]]*each_masked_region_corr.shape[0]
    #     print(len(all_region_corr_list)) #100
    #     imp_i=np.array(all_region_corr_list)-each_masked_region_corr[:,i] #
    #     imp_i=imp_i.tolist()
    #     print("imp_i.shape", len(imp_i))
    #     print(imp_i)
    #     for k in range(1,101):
    #         corr_nii_i[corr_nii_i==k]= imp_i[k-1]
    #     print(corr_nii_i.shape)
    #     corr_nii_i = corr_nii_i[:, :, :61]
    #     corr_nii_i = corr_nii_i[3:, :, :]
    #     data_expanded_inver = np.zeros((61, 73, 61))
    #     data_expanded_inver[:, 5:69, :] = corr_nii_i
    #     corr_nii_i_64 = data_expanded_inver
    #     # corr_nii_i_64=corr_nii_i
    #     # print(corr_nii_i_64.shape)
    #     corr_nii_i_64 = nib.Nifti1Image(corr_nii_i_64, img.affine)
    #     nib.save(corr_nii_i_64,os.path.join(save_path_imp,"region_important_{}".format(i)))


def computed_equcal():
    array1=nib.load("/home/jiaty/SST_task/inference/BLFU23_predict_stopsuccess_by_gosuccess_seedfc_0/2024-05-21-17-02-27/cfg_12_nstep_50_inference_mode:noise_pred/all_region_important/region_important_0.nii").get_fdata()
    array2 = nib.load(
        "/home/jiaty/SST_task/inference/BLFU23_predict_stopsuccess_by_gosuccess_seedfc_0/2024-05-21-17-02-27/cfg_12_nstep_50_inference_mode:noise_pred/all_region_important/region_important_1.nii").get_fdata()
    are_equal=np.array_equal(array1, array2)
    print("Are the two arrays equal?", are_equal)
if __name__ == '__main__':
    #N 2024-04-06-20-32-08_noise_pred_64_8_10_512_8_300_0.0001  H 2024-04-06-19-59-37_noise_pred_64_8_10_512_8_300_0.0001
    main_path_inference = "/home/jiaty/SST_task/inference/BLFU23_predict_stopsuccess_by_gosuccess_seedfc_0/2024-05-21-17-02-27/cfg_12_nstep_50_inference_mode:noise_pred"
    #predict MID by EFT
    file_name='test_prediction_SST_naverge_1.npy'
    path=os.path.join(main_path_inference,file_name)
    save_path=os.path.dirname(path)

    # preprocess_data_add_zero_reshape_to_64()
    # train_data,test_data,test_num_list,train_num_list,index_sub_all=load_data_fu23baseline(target=1)  # predict_gosuccess_by_stopsuccess:0 predict_stopsuccess_by_gosuccess:1
    # train_data, test_data = load_data_fu23()
    # plot_lossCurve(save_path,main_path_inference)
    # mask_data_path=mask_predict_data_add_zero(path,save_path)

    # show_example(mask_data_path,save_path,index_sub_all)

    # computed_corr_inference_test(path,save_path,test_data,type='fu2',datasets_name='fu23baseline',split=test_num_list) #[195,262, 232]  #

    # computed_corr_train_mean_test(save_path,train_data,test_data,type='fu2',datasets_name='fu23baseline',split=test_num_list,train_split=train_num_list) #[ 892, 754,1034]
    # computed_corr_inference_inference(path,save_path,type='fu2',datasets_name='fu23',split=test_num_list)
    # computed_masked_region_inference_corr_test(path,save_path,test_data,type='fu2',datasets_name="fu23baseline",split=test_num_list)
    computed_region_important(save_path)
    # computed_equcal()
    # inference_data=os.path.join(main_path_inference,'predicted_masked_all_naverage_1_mean.npy')
    # computed_map_test_and_inference(test_data,path,save_path)
    # computed_train_dataset_mean_std_map(train_data,save_path)
    # superbigFLICA_path="/home/jiaty/SST_task/inference/BLFU23_predict_stopsuccess_by_gosuccess_seedfc_1"
    # superbigFLICA_path=os.path.join(superbigFLICA_path,"superbigFLICA")
    # if os.path.exists(superbigFLICA_path) is False:
    #     os.makedirs(superbigFLICA_path)
    # superbigFLICA_pred_test=os.path.join(superbigFLICA_path,'pred_test.npy')
    # # save_npy_to_nii(superbigFLICA_pred_test,superbigFLICA_path)
    # rr_path=os.path.join(superbigFLICA_path,"correlation_coefficients_fu2.npy")
    # show_max_corr(rr_path)

    # 执行当前Python文件（假设当前文件名为current_file.py）
    # exec(open("functions.py").read())
    #spilt_train_test_data()
    #
    # # 恢复标准输出
    # sys.stdout.close()
    # sys.stdout = sys.__stdout__
    # t_test()

    # 这个是计算单个数据
    # computed_corr_646464_inference_test225(path,save_path)
    # computed_corr_646464_inference_train923mean(path,save_path)
    # computed_corr_646464_train932mean_test225(save_path)
    # computed_corr_646464_trainrandom225_inference(path,save_path)
    # computed_corr_646464_trainrandom225_test225(save_path)

    # computed_corr_mean_to_all(mask_data_path,save_path)
    # computed_corr_real_mean_to_all_real(save_path)
    # show_rr_rr(save_path)
    #all_brain_t_test()
    # check_function()
    # computed_FC()
    # rr = np.load(
    #     "/home/jiaty/UVIT_result_N_BL23/2024-04-09-14-43-24_noise_pred_64_8_10_512_8_400_0.0001_cfg_12_nstep_50_dep_10_embed_512/all_real_mean_corr_each_real_baseline.npy")
    # print(rr)
    # print(rr.shape) #(729, 1)
    # print(rr.max())
    # max_index=np.argmax(rr)
    # print(rr[max_index])
    # print(max_index)
    # change_mat_to_npy()
    # load_data()
    # computed_seedfc()
    # computed_test()
    # check_template()
    print("done!!!")
    #/home/jiaty/IMAGEN_cross_task_predict/inference/inference_predict_EFT_by_MID_on_mean_x0_pred/2024-04-22-22-50-50prem_x0_predbs_64ps_8dep_10embd_512nh_8epo_400lr_0.0001p_0.05_usingFC_True/cfg_12_nstep_50_inference_mode:noise_pred




