# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:29:51 2021

@author: rallapallih2
"""

import os
import numpy as np
import nibabel as nib



#%%
# define paths to NIFTI format images and load them

PathToImageInversion1100 = r'\\nindsdirfs2\shares\LFMI\FMM\Shared\rallapallih\ScanData\20210715_124732_HR_Zip14_NoMn_M2_HR_Zip14_NoMn_1_1\23\ACQ_BRUKER_PVMT1_MDEFTX23P1.nii.gz'
PathToImageInversion1500 = r'\\nindsdirfs2\shares\LFMI\FMM\Shared\rallapallih\ScanData\20210715_124732_HR_Zip14_NoMn_M2_HR_Zip14_NoMn_1_1\24\ACQ_BRUKER_PVMT1_MDEFTX24P1.nii.gz'
PathToImageInversion2000 = r'\\nindsdirfs2\shares\LFMI\FMM\Shared\rallapallih\ScanData\20210715_124732_HR_Zip14_NoMn_M2_HR_Zip14_NoMn_1_1\25\ACQ_BRUKER_PVMT1_MDEFTX25P1.nii.gz'

ImageInversion1100 = nib.load(PathToImageInversion1100)
ImageInversion1500 = nib.load(PathToImageInversion1500)
ImageInversion2000 = nib.load(PathToImageInversion2000)

ImageAffine = ImageInversion1100.affine

#%%
# get image matrix data and shape
Inversion1100 = ImageInversion1100.get_fdata()
Inversion1500 = ImageInversion1500.get_fdata()
Inversion2000 = ImageInversion2000.get_fdata()

ImageShape = Inversion1100.shape
NumPoints = np.product(ImageShape)

#%%

# From equation 4 of https://qmrlab.org/jekyll/2018/10/23/T1-mapping-inversion-recovery.html, 
# signal(InversionTime) = abs(a + b*exp(-TI/T1))
# I am going to assume a = 0 and b = 1 because I don't know any better

InversionTimes = np.array((1100,1500,2000), dtype = 'float64')
InversionTimes = np.vstack([InversionTimes, np.ones(len(InversionTimes))]).T

#%%

FitSlope = np.empty(shape = ImageShape, dtype = 'float64')
FitOffset = np.empty(shape = ImageShape, dtype = 'float64')
FitResiduals =  np.empty(shape = ImageShape, dtype = 'float64')
for i in range(ImageShape[0]):
       for j in range(ImageShape[1]):
            for k in range(ImageShape[2]):
                
                Signal = np.array((np.log(Inversion1100[i][j][k]),np.log(Inversion1500[i][j][k]),np.log(Inversion2000[i][j][k])))
            
            
                (FitSlope[i][j][k], FitOffset[i][j][k]), FitResiduals[i][j][k] = np.linalg.lstsq(InversionTimes/1000, Signal, rcond=None)[0:2]
                
#%%

FitSlopeCorrected = np.abs(FitSlope)

OutFit = nib.Nifti1Image(FitSlopeCorrected, ImageAffine)
OutResiduals = nib.Nifti1Image(FitResiduals, ImageAffine)

nib.save(OutFit, os.path.join(r'\\nindsdirfs2\shares\LFMI\FMM\Shared\rallapallih\ScanData\20210715_124732_HR_Zip14_NoMn_M2_HR_Zip14_NoMn_1_1', 'testT1Map.nii.gz'))
nib.save(OutResiduals, os.path.join(r'\\nindsdirfs2\shares\LFMI\FMM\Shared\rallapallih\ScanData\20210715_124732_HR_Zip14_NoMn_M2_HR_Zip14_NoMn_1_1', 'testResiduals.nii.gz'))  