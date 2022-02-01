# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:24:03 2021

@author: Hari Rallapalli
"""


import os
import numpy as np
import nibabel as nib

# input_path = r'G:\MINCVM\PCLKO\PCP2-DTR\maps\\'
# output_path = r'G:\MINCVM\PCLKO\PCP2-DTR\maps\Extracted\\'

input_path = r'\\nindsdirfs2\shares\LFMI\FMM\Shared\rallapallih\ScanData\20211202_Controls\Maps\\'
output_path = r'\\nindsdirfs2\shares\LFMI\FMM\Shared\rallapallih\ScanData\20211202_Controls\Maps\Extracted\\'

for file in os.listdir(input_path):
    if file.endswith('.nii.gz'):
        
        
        image_name = file
        img = nib.load((input_path + image_name))
        
        img_data = img.get_fdata()
        img_data = np.asarray(img_data)
        sizer = np.asarray(img_data.shape)
        
        sub_img = np.empty((sizer[0],sizer[1]))
        affine = np.diag([1,2,3,4])
        
        sub_img = img_data[:,:,2]
        
        sub_img_nii = nib.Nifti1Image(sub_img,affine)
        
        nib.save(sub_img_nii,(output_path+image_name[:-7]+'_MapExtracted.nii.gz'))