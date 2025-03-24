import numpy as np
import torch
from torch.utils.data import Dataset
import random
import nibabel as nib




class My3DDataset_inferece(Dataset):
    def __init__(self, none_test_data, test_data, FC_data ,VBM_data,seedfc_data,dti_data,target_idx, condition, p_uncond, mask_img,self_mask_rate, using_fc=True,classfifier_flag=False,number_list=[225,424],flatten=False,None_condition=False,using_mask=False,using_vbm=False,norm=False,mask_rate=0.01,using_seedfc=False,using_dti=False,region_mask_id=None):  # for classifier free guidance
        """
        :param data_path:
        :param condition: Specifies which data indices to use as conditions (list type)
        """
        # Load numpy data
        self.none_test_data = none_test_data#np.load(data_path)
        self.test_data=test_data
        self.condition=condition
        self.FC_data=FC_data
        self.VBM_data=VBM_data
        self.classfifier_flag=classfifier_flag
        self.p_uncond=p_uncond
        self.using_fc=using_fc
        self.using_vbm=using_vbm
        self.num_list=number_list
        self.None_condition = None_condition
       
        self.using_mask=using_mask
        self.mask_img=mask_img
        self.self_mask_rate = self_mask_rate
        if self.using_mask:
            self.get_mask()
        self.flatten=flatten
        self.target_idx = target_idx
        self.get_mean_data()
        self.norm=norm
        self.mask_rate = mask_rate
        self.using_seedfc=using_seedfc
        self.seedfc_data=seedfc_data
        self.dti_data=dti_data
        self.using_dti=using_dti
        self.region_mask_id=region_mask_id
        if self.region_mask_id is not None:
            self.template=self.load_region_template()
    def normalized(self, x):
        x = (x - np.mean(x)) / (np.std(x)+0.001)
        return x

    def get_mask(self):
        mask_64 = []
        for image_mask in self.mask_img:
            sub_contrast = image_mask[:, 5:69, :]
            data_zero = np.zeros((64 - 61, 64, 61))
            sub_contrast = np.concatenate((data_zero, sub_contrast), axis=0)
            data_expanded_second = np.zeros((64, 64, 64))
            data_expanded_second[:, :, :61] = sub_contrast
            image_mask64 = data_expanded_second
            mask_64.append(image_mask64)
        self.mask_img = mask_64
    def load_region_template(self):
        template = nib.load(
            "/home/jiaty/Templates/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_3mm.nii.gz").get_fdata()
        template=torch.FloatTensor(template)
        return template
    def get_mean_data(self):
        
        self.mean_data_fu2 = np.nanmean(self.none_test_data[:self.num_list[0],self.target_idx,:,:,:],axis=0)
        if len(self.num_list)==2:
            self.mean_data_fu3 = np.nanmean(self.none_test_data[self.num_list[0]:, self.target_idx, :, :, :], axis=0)
        elif len(self.num_list) == 3:
            self.mean_data_fu3 = np.nanmean(self.none_test_data[self.num_list[0]:self.num_list[0]+self.num_list[1],  self.target_idx, :, :, :], axis=0)
            self.mean_data_bl = np.nanmean(self.none_test_data[self.num_list[0]+self.num_list[1]:, self.target_idx, :, :, :], axis=0)

    def __len__(self):
        return self.test_data.shape[0]

    def __getitem__(self, idx):
 
        if idx<self.num_list[0]:
            input_data=self.mean_data_fu2
        elif idx < self.num_list[1]+self.num_list[0]:
            input_data = self.mean_data_fu3
        elif len(self.num_list)==3 and idx < self.num_list[1]+self.num_list[0]+self.num_list[2]:
            input_data = self.mean_data_bl
        input_data=np.expand_dims(input_data,0)

        if self.using_fc:
            person_FC=self.FC_data[idx]
            person_FC=np.nan_to_num(person_FC,nan=0)
        if self.using_vbm:
            person_vbm =self.VBM_data[idx]
        if self.using_seedfc:
            person_seedfc=self.seedfc_data[idx]
        if self.using_dti:
            person_dti=self.dti_data[idx]

        labels=self.test_data[idx]
    
        if isinstance(self.condition, int):
            labels= labels[1:self.condition+1]
        if isinstance(self.condition, list):
            labels=labels[self.condition]
        if len(labels.shape)==3:
            labels=np.expand_dims(labels,0) 
        if self.classfifier_flag:
            if random.random() < self.p_uncond:
                labels = np.zeros_like(labels)

        if self.using_mask:
            if idx < self.num_list[0]:
                labels = labels * self.mask_img[0]
            elif idx < self.num_list[0] + self.num_list[1]:
                labels = labels * self.mask_img[1]
            elif idx < self.num_list[0] + self.num_list[1] + self.num_list[2]:
                labels = labels * self.mask_img[2]


        if self.norm:
            input_data = self.normalized(input_data)
            if self.using_vbm:
                person_vbm = self.normalized(person_vbm)
            if self.using_fc:
                person_FC = self.normalized(person_FC)
            labels = self.normalized(labels)
            if self.using_seedfc:
                person_seedfc=self.normalized(person_seedfc)
            if self.using_dti:
                person_dti=self.normalized(person_dti)

        input_data = torch.FloatTensor(input_data)
        if self.self_mask_rate is not None:
            rate1 = np.percentile(labels, self.self_mask_rate / 2)
            rate2 = np.percentile(labels, 100 - self.self_mask_rate / 2)
            labels = np.where((labels > rate1) & (labels < rate2), 0, labels)  #

        labels = torch.FloatTensor(labels)

        if self.None_condition is False:
            labels=labels
        else:
            labels=torch.zeros_like(input_data)
        if self.region_mask_id is not None:
            indices = torch.nonzero(self.template == self.region_mask_id, as_tuple=False).t()
            labels[:, indices[0], indices[1], indices[2]] = 0
        labels = torch.FloatTensor(labels)

        if self.using_fc:
            person_FC = torch.FloatTensor(np.expand_dims(person_FC, 0))
            labels = torch.cat([labels, person_FC], dim=0)
        if self.using_vbm:
            person_vbm=torch.FloatTensor(np.expand_dims(person_vbm,0))
            labels=torch.cat([labels,person_vbm],dim=0)
        if self.using_seedfc:
            person_seedfc=torch.FloatTensor(np.expand_dims(person_seedfc,0))
            labels=torch.cat([labels,person_seedfc],dim=0)
        if self.using_dti:
            person_dti=torch.FloatTensor(np.expand_dims(person_dti,0))
            labels=torch.cat([labels,person_dti],dim=0)

        if self.flatten:
            input_data = torch.flatten(input_data, start_dim=1, end_dim=-1)  # batch 67683
            labels = torch.flatten(labels, start_dim=0, end_dim=-1)
            labels = torch.unsqueeze(labels, dim=0)
            person_FC = torch.flatten(person_FC, start_dim=1, end_dim=-1)

        if self.mask_rate > 0:
            mask = np.random.choice([0, 1], size=labels.shape, p=[self.mask_rate, 1 - self.mask_rate])
            mask = torch.FloatTensor(mask)
            labels = labels * mask


        this_is_a_rubbish_parame = 0
        return input_data, labels, this_is_a_rubbish_parame
    @property
    def data_shape(self):
        # Returns shape of a single 3D volume
        return self.test_data[0].shape
    @property
    def person_image_num(self):
        return self.test_data.shape[1]



class My3DDataset_target(Dataset):
    def __init__(self, data, FC_data,VBM_data,seedfc_data,dti_data,target_indx, condition, p_uncond,mask_img,train_num_list,self_mask_rate, using_FC=True,classfifier_flag=True,flatten=False,None_condition=False,using_mask=False, using_VBM=False,norm=False,mask_rate=0.01,using_seedfc=False,using_dti=False):  # for classifier free guidance
        """

        :param data_path:
        :param condition: Specifies which data indices to use as conditions (list type)
        """
        # Load numpy data
        self.data = data#np.load(data_path)
        self.FC_data= FC_data
        self.VBM_data=VBM_data
        self.condition=condition
        self.classfifier_flag=classfifier_flag
        self.p_uncond=p_uncond
        self.using_FC=using_FC
        self.flatten=flatten
        self.target_indx=target_indx
        self.None_condition=None_condition
        self.using_mask=using_mask
        self.mask_img=mask_img
        self.self_mask_rate = self_mask_rate
        self.using_seedfc=using_seedfc
        self.seedfc_data=seedfc_data
        self.dti_data=dti_data
        self.using_dti=using_dti
        if self.using_mask:
            self.get_mask()
        self.train_num_list=train_num_list
        self.using_VBM=using_VBM

        self.norm = norm
        self.mask_rate = mask_rate

    def normalized(self, x):
        x = (x - np.mean(x)) / (np.std(x)+0.001)
        return x

    def get_mask(self):
        mask_64=[]
        for image_mask in self.mask_img:
            sub_contrast = image_mask[:, 5:69, :]
            data_zero = np.zeros((64 - 61, 64, 61))
            sub_contrast = np.concatenate((data_zero, sub_contrast), axis=0)
            data_expanded_second = np.zeros((64, 64, 64))
            data_expanded_second[:, :, :61] = sub_contrast
            image_mask64 = data_expanded_second
            mask_64.append(image_mask64)
        self.mask_img=mask_64

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Retrieve all 3D data for a subject
        person_data = self.data[idx]

        # Use first 3D volume as input, others as labels
        input_data = person_data[self.target_indx]
        input_data = np.expand_dims(input_data, 0)
        if self.using_FC:
            person_FC=self.FC_data[idx]
            person_FC=np.nan_to_num(person_FC,nan=0)
        if self.using_VBM:
            person_vbm =self.VBM_data[idx]
        if self.using_seedfc:
            person_seedfc=self.seedfc_data[idx]
        if self.using_dti:
            person_dti=self.dti_data[idx]
     

        if isinstance(self.condition, int):
            labels= person_data[1:self.condition+1]
        if isinstance(self.condition, list):
            labels=person_data[self.condition]
    
        if len(labels.shape)==3:
            labels=np.expand_dims(labels,0) 

        if self.classfifier_flag:
            if random.random() < self.p_uncond:
                labels = np.zeros_like(labels)

     
        if self.using_mask:
            if idx < self.train_num_list[0]:
                labels=labels*self.mask_img[0]
            elif idx <self.train_num_list[0]+self.train_num_list[1]:
                labels = labels * self.mask_img[1]
            elif idx <self.train_num_list[0]+self.train_num_list[1]+self.train_num_list[2]:
                labels = labels * self.mask_img[2]



        if self.norm:
            input_data = self.normalized(input_data)
            if self.using_VBM:
                person_vbm = self.normalized(person_vbm)
            if self.using_FC:
                person_FC = self.normalized(person_FC)
            labels = self.normalized(labels)
            if self.using_seedfc:
                person_seedfc=self.normalized(person_seedfc)
            if self.using_dti:
                person_dti=self.normalized(person_dti)


        if self.self_mask_rate is not None:
            rate1 = np.percentile(labels, self.self_mask_rate / 2)
            rate2 = np.percentile(labels, 100 - self.self_mask_rate / 2)
            labels = np.where((labels > rate1) & (labels < rate2), 0, labels)


        input_data = torch.FloatTensor(input_data)
        labels = torch.FloatTensor(labels)



        if self.None_condition is False:
            labels=labels
        else:
            labels=torch.zeros_like(input_data)
        if self.using_FC:
            person_FC = torch.FloatTensor(np.expand_dims(person_FC, 0))
            labels = torch.cat([labels, person_FC], dim=0)
        if self.using_VBM:
            person_vbm=torch.FloatTensor(np.expand_dims(person_vbm,0))
            labels=torch.cat([labels,person_vbm],dim=0)
        if self.using_seedfc:
            person_seedfc=torch.FloatTensor(np.expand_dims(person_seedfc,0))
            labels=torch.cat([labels,person_seedfc],dim=0)
        if self.using_dti:
            person_dti=torch.FloatTensor(np.expand_dims(person_dti,0))
            labels=torch.cat([labels,person_dti],dim=0)

        if self.flatten:
            input_data = torch.flatten(input_data, start_dim=1, end_dim=-1)  
            labels=torch.flatten(labels,start_dim=0,end_dim=-1)
            labels=torch.unsqueeze(labels,dim=0)
            person_FC=torch.flatten(person_FC,start_dim=1,end_dim=-1)

        if self.mask_rate > 0:
            mask = np.random.choice([0, 1], size=labels.shape, p=[self.mask_rate, 1 - self.mask_rate])
            mask = torch.FloatTensor(mask)
            labels = labels * mask

      
        labels = torch.FloatTensor(labels)

        this_is_a_rubbish_parame=0

        return input_data, labels, this_is_a_rubbish_parame


    @property
    def data_shape(self):
        # Returns shape of a single 3D volume
        return self.data[0][0].shape
    @property
    def person_image_num(self):
        return self.data.shape[1]
