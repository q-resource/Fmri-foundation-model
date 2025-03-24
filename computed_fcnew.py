import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
import nibabel as nib


def computed_FC():
    """
    Compute Functional Connectivity (FC) matrices from fMRI data

    Processing pipeline:
    1. Load subject IDs from CSV file
    2. Preprocess and resample fMRI data for each subject
    3. Calculate FC matrix using Pearson correlation
    4. Handle missing data by imputing group mean
    5. Save all FC matrices as numpy array

    Output:
    - Saves FU3_1038_rest_FC.npy containing 1038 subjects' FC matrices (64x64x64)
    """
    
    # Load subject IDs from CSV
    subj_csv_path = '/public/home/zhairq/tensordecoding/NASCAR/tensordecodingresult/BLpassQC1038.csv'
    folder_names = pd.read_csv(subj_csv_path)['Sub_ID'].tolist()

    # Initialize array for storing FC matrices [subjects x 64 x 64 x 64]
    all_subjects_fc = np.zeros((1038, 64, 64, 64))
    missing_subjects = []

    for subj_idx, subj_id in enumerate(folder_names):
        # Clean subject ID formatting
        subj_id = subj_id.split("'")[0]
        
        # Construct fMRI data path
        fmri_path = f'/public/mig_old_storage/home1/ISTBI_data/IMAGEN_New_Preprocessed/Prepressed/fmriprep/{subj_id}/ses-followup3/func/Preprocessed_{subj_id}_ses-followup3_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'

        if not os.path.exists(fmri_path):
            missing_subjects.append(subj_idx)
            continue

        try:
            # Load and process fMRI data
            fmri_img = nib.load(fmri_path)
            fmri_data = fmri_img.get_fdata()

            # Resample to 64x64x64 resolution
            resampled_data = np.zeros((64, 64, 64, fmri_data.shape[-1]))
            for t in range(fmri_data.shape[-1]):
                vol = torch.from_numpy(fmri_data[..., t].copy())
                vol = vol.unsqueeze(0).unsqueeze(0)  # Add batch/channel dims
                resampled_vol = F.interpolate(vol, size=(64, 64, 64), mode='nearest')
                resampled_data[..., t] = resampled_vol[0, 0].numpy()

            # Reorganize data into 8x8x8 blocks
            reshaped_data = rearrange(
                resampled_data,
                '(b1 p1) (b2 p2) (b3 p3) t -> (b1 b2 b3) (p1 p2 p3) t',
                b1=8, b2=8, b3=8, p1=8, p2=8, p3=8
            )

            # Calculate block-wise mean time series
            block_means = np.nanmean(reshaped_data, axis=1)

            # Compute correlation matrix
            corr_matrix = np.corrcoef(block_means)

            # Reconstruct 3D FC matrix
            fc_matrix = rearrange(
                corr_matrix,
                '(b1 b2 b3) (p1 p2 p3) -> (b1 p1) (b2 p2) (b3 p3)',
                b1=8, b2=8, b3=8, p1=8, p2=8, p3=8
            )

            all_subjects_fc[subj_idx] = fc_matrix

        except Exception as e:
            print(f"Error processing subject {subj_id}: {str(e)}")
            missing_subjects.append(subj_idx)

    # Handle missing data
    if missing_subjects:
        group_mean = np.mean(all_subjects_fc, axis=0)
        for idx in missing_subjects:
            all_subjects_fc[idx] = group_mean

    # Save results
    output_path = '/public/home/zhairq/tensordecoding/NASCAR/tensordecodingresult/FU3_1038_rest_FC.npy'
    np.save(output_path, all_subjects_fc)
    print(f"Processing completed. Saved {len(folder_names)-len(missing_subjects)} subjects' FC matrices.")


if __name__ == '__main__':
    computed_FC()