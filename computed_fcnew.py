# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F


# from einops import rearrange, reduce
# import nibabel as nib
# def computed_FC():
#
#
#     # 假设文件夹名称存储在csv文件中
#     folder_names = pd.read_csv('/public/home/zhairq/tensordecoding/NASCAR/tensordecodingresult/BLpassQC1038.csv')['Sub_ID'].tolist()
#     #folder_names=folder_names[0:1]
#     # folder_names=['Preprocessed_sub-000000112288_ses-followup2_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz']
#     all_subject=np.zeros([1038,64,64,64])
#     zere_idx=[]
#     for idx,folder_name in enumerate(folder_names):
#         # 构造文件路径
#         folder_name=folder_name.split("'")[0]
#         file_path = f'/public/mig_old_storage/home1/ISTBI_data/IMAGEN_New_Preprocessed/Prepressed/fmriprep/{folder_name}/ses-followup3/func/Preprocessed_{folder_name}_ses-followup3_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'
#         # file_path=os.path.join('/home/jiaty/UVIT_dataset',folder_name)
#         # 加载Nifti文件
#         if os.path.exists(file_path):
#             img = nib.load(file_path)
#             data = img.get_fdata()
#
#             # 替换nan值为0
#             #data = np.nan_to_num(data)
#
#             # 将数据转换为PyTorch张量
#             # data_tensor = torch.from_numpy(data)
#
#             # 使用最近邻插值将数据插值为64x64x64
#             data_sampled_all=np.zeros([64,64,64,data.shape[-1]])
#             # print(data_sampled_all.shape)
#             for i in range(data.shape[-1]):
#                 data_i=data[:,:,:,i]
#                 # print(data_i.shape) #(61, 73, 61)
#                 data_i=np.expand_dims(np.expand_dims(data_i,0),0)
#                 data_i_tensor=torch.from_numpy(data_i)
#                 # print(data_i.shape)
#                 data_upsampled = F.interpolate(data_i_tensor, size=(64, 64, 64), mode='nearest')[0,0,:,:,:]
#                 data_upsampled = data_upsampled.numpy()
#                 data_sampled_all[:,:,:,i]=data_upsampled
#
#             # 使用einops重塑和分块
#             # print(data_sampled_all.shape) #(64, 64, 64, 187)
#             data_upsampled_re = rearrange(data_sampled_all, '(b1 p1) (b2 p2) (b3 p3) t -> (b1 b2 b3) (p1 p2 p3) t',b1=8,b2=8,b3=8,p1=8,p2=8,p3=8)
#             # print(data_upsampled_re.shape) #(512, 512, 187)
#             patches = np.nanmean(data_upsampled_re, axis=1)
#             print("patches.shape",patches.shape)
#             #patches=patches.squeeze(1)
#             # print("patches.shape", patches.shape) #patches.shape (512, 187)
#
#             # 计算相关系数矩阵
#             corr_matrix = np.corrcoef(patches)
#             # print(corr_matrix.shape) #(187, 187)
#
#             # 使用einops将相关系数矩阵重塑为64x64x64
#             corr_tensor = rearrange(corr_matrix, '(b1 b2 b3) (p1 p2 p3) ->(b1 p1) (b2 p2) (b3 p3)', b1=8,b2=8,b3=8,p1=8,p2=8,p3=8)
#             # print("corr_tensor.shape",corr_tensor.shape)
#             # 保存结果
#             all_subject[idx,:,:,:]=corr_tensor
#         else:
#             zere_idx.append(idx)
#     mean_all_subject=np.mean(all_subject,axis=0)
#     print("mean_all_subject",mean_all_subject)
#     for i in zere_idx:
#         all_subject[i,:,:,:]=mean_all_subject
#     print("all_subject",np.array(all_subject).shape)
#     np.save(os.path.join("/public/home/zhairq/tensordecoding/NASCAR/tensordecodingresult/",'FU3_1038_rest_FC.npy'),np.array(all_subject))
#
#     # output_path = os.path.join('/public/home/zhairq/tensordecoding/NASCAR/tensordecodingresult/','output_fc_test.nii.gz')
#     # nib.save(nib.Nifti1Image(corr_tensor, img.affine), output_path)
#     print("处理完成!")
# if __name__ == '__main__':
# #     computed_FC()








# # import os
# # import numpy as np
# # import pandas as pd
# # import torch
# # import torch.nn.functional as F
# # from einops import rearrange
# # import nibabel as nib


# # def computed_FC():
# #     folder_names = pd.read_csv('/public/home/zhairq/tensordecoding/NASCAR/tensordecodingresult/ABCD2469.csv')[
# #         'Sub_ID'].tolist()

# #     all_subject = []
# #     zero_idx = []
# #     for idx, folder_name in enumerate(folder_names):
# #         folder_name = folder_name.split("'")[0]
# #         file_path = f'/public/home/zhairq/ABCDbids/postfmriprep/fmriprep/{folder_name}/func/{folder_name}_task-rest_run-1_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz'
# #         try:
# #             if os.path.exists(file_path):
# #                 print(idx)
# #                 img = nib.load(file_path)
# #                 data = img.get_fdata()

# #                 data_sampled_all = np.zeros([64, 64, 64, data.shape[-1]])
# #                 for i in range(data.shape[-1]):
# #                     data_i = data[:, :, :, i]
# #                     data_i = np.expand_dims(np.expand_dims(data_i, 0), 0)
# #                     data_i_tensor = torch.from_numpy(data_i)
# #                     data_upsampled = F.interpolate(data_i_tensor, size=(64, 64, 64), mode='nearest')[0, 0, :, :, :]
# #                     data_upsampled = data_upsampled.numpy()
# #                     data_sampled_all[:, :, :, i] = data_upsampled

# #                 data_upsampled_re = rearrange(data_sampled_all, '(b1 p1) (b2 p2) (b3 p3) t -> (b1 b2 b3) (p1 p2 p3) t',
# #                                               b1=8, b2=8, b3=8, p1=8, p2=8, p3=8)
# #                 patches = np.nanmean(data_upsampled_re, axis=1)

# #                 corr_matrix = np.corrcoef(patches)
# #                 corr_tensor = rearrange(corr_matrix, '(b1 b2 b3) (p1 p2 p3) -> (b1 p1) (b2 p2) (b3 p3)', b1=8, b2=8,
# #                                         b3=8, p1=8, p2=8, p3=8)

# #                 all_subject.append(corr_tensor)
# #             else:
# #                 zero_idx.append(idx)
# #         except Exception:
# #             pass

# #     all_subject = np.array(all_subject)
# #     mean_all_subject = np.mean(all_subject, axis=0)
# #     for i in zero_idx:
# #         all_subject = np.insert(all_subject, i, mean_all_subject, axis=0)

# #     np.save(os.path.join("/public/home/zhairq/tensordecoding/NASCAR/tensordecodingresult/", 'ABCD_2469_rest_FC.npy'),
# #             np.array(all_subject))
# #     print("处理完成!")


# # if __name__ == '__main__':
# #     computed_FC()








# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from einops import rearrange
# import nibabel as nib




# def computed_FC():
#     folder_names = pd.read_csv('/public/home/zhairq/tensordecoding/NASCAR/tensordecodingresult/HCPMSfilename.csv')['Sub_ID'].tolist()

#     all_subject = []
#     zero_idx = []
#     for idx, folder_name in enumerate(folder_names):
#         if idx == 1050:
#             continue  # 跳过 idx 等于 100 的情况
#         folder_name = folder_name.split("'")[0]
#         file_path = f'/public/home/gongwk/hcpdata/hcprest/rest1/{folder_name}/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_hp2000_clean.nii.gz'
#         try:
#             print(f"Processing folder: {folder_name}")
#             print(f"File path: {file_path}")
#             if os.path.exists(file_path):
#                 print(f"Index: {idx}")
#                 img = nib.load(file_path)
#                 data = img.get_fdata()

#                 data_sampled_all = np.zeros([100, 100, 100, data.shape[-1]])
#                 for i in range(data.shape[-1]):
#                     data_i = data[:, :, :, i]
#                     data_i = np.expand_dims(np.expand_dims(data_i, 0), 0)
#                     data_i_tensor = torch.from_numpy(data_i)
#                     data_upsampled = F.interpolate(data_i_tensor, size=(100, 100, 100), mode='nearest')[0, 0, :, :, :]
#                     data_upsampled = data_upsampled.numpy()
#                     data_sampled_all[:, :, :, i] = data_upsampled

#                 data_upsampled_re = rearrange(data_sampled_all, '(b1 p1) (b2 p2) (b3 p3) t -> (b1 b2 b3) (p1 p2 p3) t', b1=10, b2=10, b3=10, p1=10, p2=10, p3=10)
#                 patches = np.nanmean(data_upsampled_re, axis=1)

#                 if patches.size > 0:
#                     mean_patches = np.nanmean(patches, axis=0)
#                     all_subject.append(mean_patches)
#                 else:
#                     print(f"Empty slice found at index {idx}. Skipping...")
#                     zero_idx.append(idx)
#             else:
#                 print(f"File not found for folder: {folder_name}")
#                 zero_idx.append(idx)
#         except Exception as e:
#             print(f"An error occurred for folder: {folder_name}. Error: {e}")
#             pass

#     all_subject = np.array(all_subject)
#     mean_all_subject = np.mean(all_subject, axis=0)
#     for i in zero_idx:
#         all_subject = np.insert(all_subject, i, mean_all_subject, axis=0)

#     np.save(os.path.join("/public/home/zhairq/tensordecoding/NASCAR/tensordecodingresult/", 'HCPMS_rest1RL_FC.npy'), np.array(all_subject))
#     print("处理完成!")


# if __name__ == '__main__':
#     computed_FC()





# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from einops import rearrange
# import nibabel as nib

# def computed_FC():
#     folder_names = pd.read_csv('/public/home/zhairq/tensordecoding/NASCAR/tensordecodingresult/HCPMSfilename.csv')['Sub_ID'].tolist()

#     all_subject = []
#     zero_idx = []
#     for idx, folder_name in enumerate(folder_names):
#         folder_name = folder_name.split("'")[0]
#         file_path = f'/public/home/gongwk/hcpdata/hcprest/rest1/{folder_name}/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_hp2000_clean.nii.gz'
#         try:
#             print(f"Processing folder: {folder_name}")
#             print(f"File path: {file_path}")
#             if os.path.exists(file_path):
#                 print(f"Index: {idx}")
#                 img = nib.load(file_path)
#                 data = img.get_fdata()

#                 data_sampled_all = np.zeros([100, 100, 100, data.shape[-1]])
#                 for i in range(data.shape[-1]):
#                     data_i = data[:, :, :, i]
#                     data_i = np.expand_dims(np.expand_dims(data_i, 0), 0)
#                     data_i_tensor = torch.from_numpy(data_i)
#                     data_upsampled = F.interpolate(data_i_tensor, size=(100, 100, 100), mode='nearest')[0, 0, :, :, :]
#                     data_upsampled = data_upsampled.numpy()
#                     data_sampled_all[:, :, :, i] = data_upsampled

#                 data_upsampled_re = rearrange(data_sampled_all, '(b1 p1) (b2 p2) (b3 p3) t -> (b1 b2 b3) (p1 p2 p3) t', b1=10, b2=10, b3=10, p1=10, p2=10, p3=10)
#                 patches = np.nanmean(data_upsampled_re, axis=1)

#                 if patches.size > 0:
#                     mean_patches = np.nanmean(patches, axis=0)
#                     all_subject.append(mean_patches)
#                 else:
#                     print(f"Empty slice found at index {idx}. Skipping...")
#                     zero_idx.append(idx)
#             else:
#                 print(f"File not found for folder: {folder_name}")
#                 zero_idx.append(idx)
#         except Exception as e:
#             print(f"An error occurred for folder: {folder_name}. Error: {e}")
#             zero_idx.append(idx)  # 记录出错的索引
#             continue  # 继续执行下一个循环

#     all_subject = np.array(all_subject)
#     mean_all_subject = np.mean(all_subject, axis=0)
#     for i in zero_idx:
#         all_subject = np.insert(all_subject, i, mean_all_subject, axis=0)

#     np.save(os.path.join("/public/home/zhairq/tensordecoding/NASCAR/tensordecodingresult/", 'HCPMS_rest1RL_FC.npy'), np.array(all_subject))
#     print("处理完成!")


# if __name__ == '__main__':
#     computed_FC()






import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


from einops import rearrange, reduce
import nibabel as nib
def computed_FC():


    # 假设文件夹名称存储在csv文件中
    folder_names = pd.read_csv('/public/home/zhairq/tensordecoding/NASCAR/tensordecodingresult/HCPMSLRfilename.csv')['Sub_ID'].tolist()#HCPMSLRfilename.csv
    #folder_names=folder_names[0:1]
    # folder_names=['Preprocessed_sub-000000112288_ses-followup2_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz']
    all_subject=np.zeros([1049,64,64,64]) #1049
    zere_idx=[]
    for idx,folder_name in enumerate(folder_names):
        print(idx)
        # if idx == 100:
        #     continue  # 跳过 idx 等于 100 的情况
        # 构造文件路径
        folder_name=folder_name.split("'")[0]
        file_path = f'/public/home/gongwk/hcpdata/hcprest/rest1/{folder_name}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean.nii.gz'
        # file_path=os.path.join('/home/jiaty/UVIT_dataset',folder_name)
        # 加载Nifti文件
        if os.path.exists(file_path):
            img = nib.load(file_path)
            data = img.get_fdata()
            non_zero_count = np.count_nonzero(data)
            print(data)
            print(non_zero_count)
            # 替换nan值为0
            #data = np.nan_to_num(data)

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
            # print(data_sampled_all.shape) #(64, 64, 64, 187)
            data_upsampled_re = rearrange(data_sampled_all, '(b1 p1) (b2 p2) (b3 p3) t -> (b1 b2 b3) (p1 p2 p3) t',b1=8,b2=8,b3=8,p1=8,p2=8,p3=8)
            # print(data_upsampled_re.shape) #(512, 512, 187)
            patches = np.nanmean(data_upsampled_re, axis=1)
            print(patches)
            print("patches.shape",patches.shape)
            #patches=patches.squeeze(1)
            # print("patches.shape", patches.shape) #patches.shape (512, 187)

            # 计算相关系数矩阵
            corr_matrix = np.corrcoef(patches)
            # print(corr_matrix.shape) #(187, 187)

            # 使用einops将相关系数矩阵重塑为64x64x64
            corr_tensor = rearrange(corr_matrix, '(b1 b2 b3) (p1 p2 p3) ->(b1 p1) (b2 p2) (b3 p3)', b1=8,b2=8,b3=8,p1=8,p2=8,p3=8)
            # print("corr_tensor.shape",corr_tensor.shape)
            # 保存结果
            all_subject[idx,:,:,:]=corr_tensor
            print(corr_tensor)
            non_nan_count = np.count_nonzero(~np.isnan(corr_matrix))
            print(non_nan_count)
        else:
            zere_idx.append(idx)
    mean_all_subject=np.mean(all_subject,axis=0)
    print("mean_all_subject",mean_all_subject)
    for i in zere_idx:
        all_subject[i,:,:,:]=mean_all_subject
    print("all_subject",np.array(all_subject).shape)
    np.save(os.path.join("/public/home/zhairq/tensordecoding/NASCAR/tensordecodingresult/", 'HCPMS_rest1LR_FC64.npy'), np.array(all_subject))

    # output_path = os.path.join('/public/home/zhairq/tensordecoding/NASCAR/tensordecodingresult/','output_fc_test.nii.gz')
    # nib.save(nib.Nifti1Image(corr_tensor, img.affine), output_path)
    print("处理完成!")
if __name__ == '__main__':
    computed_FC()