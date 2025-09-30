def computed_region_important(save_path):
    # each_masked_region_corr=np.load(os.path.join(save_path,"masked_region_yeo_corr_all.npy"))
    each_masked_region_corr = np.load(os.path.join(save_path, "random_array.npy"))
    print("each_masked_region_corr.shape", each_masked_region_corr.shape)
    plt.figure(figsize=(14, 6))

    # 使用 seaborn 绘制热图
    sns.heatmap(each_masked_region_corr, cmap='viridis')
    # 添加标题和标签
    # plt.title('Heatmap of (100, 232) numpy array')
    plt.xlabel('subjects')
    plt.ylabel('regions')
    # 显示图像
    plt.savefig(os.path.join(save_path, "masked_region_yeo_corr_all_heatmap.jpg"), dpi=300)
    # all_region_corr=np.load(os.path.join(save_path, "masked_region_yeo_corr_all.npy"))
    # all_region_corr=np.load(os.path.join(save_path, "correlation_coefficients_fu2_all_region.npy"))
    print("each_masked_region_corr.shape", each_masked_region_corr.shape)
    # print("all_region_corr.shape",all_region_corr.shape)
    # template = nib.load("/home/jiaty/Templates/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_3mm.nii.gz").get_fdata()
    template = nib.load("/home/jiaty/Templates/HCPex_3mm.nii").get_fdata()
    sub_contrast = template[:, 5:69, :]
    data_zero = np.zeros((64 - 61, 64, 61))
    sub_contrast = np.concatenate((data_zero, sub_contrast), axis=0)
    data_expanded_second = np.zeros((64, 64, 64))
    data_expanded_second[:, :, :61] = sub_contrast
    image_mask64 = data_expanded_second
    # print(np.unique(image_mask64))
    home_jiaty = '/home/jiaty/UVIT_dataset'
    img = nib.load("/home/jiaty/Templates/HCPex_3mm.nii")
    save_path_imp = os.path.join(save_path, "all_region_important")
    if os.path.exists(save_path_imp) is False:
        os.makedirs(save_path_imp)
    # print(all_region_corr)
    # all_region_corr_repeat=np.tile(all_region_corr, (100, 1))
    # print("all_region_corr_repeat.shape",all_region_corr_repeat.shape)
    # print(all_region_corr_repeat[1,:])
    # all_imp=all_region_corr_repeat-each_masked_region_corr

    all_imp = each_masked_region_corr
    # all_imp=arr
    print("all_imp.shape", all_imp.shape)
    all_imp_mean = np.mean(all_imp, axis=1)
    all_imp_mean = all_imp_mean.tolist()
    corr_nii_mean = image_mask64.copy()
    for k in range(1, 427):  # HCPex427 schaefear 101
        corr_nii_mean[corr_nii_mean == k] = all_imp_mean[k - 1]
    corr_nii_mean = corr_nii_mean[:, :, :61]
    corr_nii_mean = corr_nii_mean[3:, :, :]
    data_expanded_inver = np.zeros((61, 73, 61))
    data_expanded_inver[:, 5:69, :] = corr_nii_mean
    corr_nii_i_mean = data_expanded_inver
    # corr_nii_i_64=corr_nii_i
    # print(corr_nii_i_64.shape)
    corr_nii_i_mean = nib.Nifti1Image(corr_nii_i_mean, img.affine)
    nib.save(corr_nii_i_mean, os.path.join("/home/jiaty/", "region_important_tvalue_SST_AUDvsHC.nii"))
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