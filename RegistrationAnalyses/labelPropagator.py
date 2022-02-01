# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 16:24:47 2021

@author: rallapallih2
"""
import os
from pathlib import Path
import numpy as np
import nibabel as nib
import sys
from scipy import ndimage
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)
kernel = np.ones((3,3,3))

atlasLabels = pd.read_csv(r'\\nindsdirfs2\shares\LFMI\FMM\Shared\rallapallih\DSURQE_Atlas\DSURQE_40micron_R_mapping.csv')


imageBasePath = r'\\nindsdirfs2\shares\LFMI\FMM\Shared\rallapallih\Zip14\RawImages'
atlasBasePath = r'\\nindsdirfs2\shares\LFMI\FMM\Shared\rallapallih\Zip14\RawImages\WarpedAtlas'

animal = []
roiIndex = []
roiHemi = []
roiName = []
roiImage = []
        

for file in os.listdir(imageBasePath):
    if file.endswith(".nii.gz"):
        
        fileBaseName = Path(file)
        fileBaseName = fileBaseName.with_suffix('').stem
        
        print('Now propagating labels for ' + fileBaseName)
# PathToImage = r'\\hpcdrive.nih.gov\data\MEMRI_Timecourse_Registration\RawImages\Cryocoil_MEMRITimecourse_F1_T_00hrs.nii.gz'

        pathToAtlas = os.path.join(atlasBasePath,fileBaseName+'_Atlas.nii.gz')

        image = nib.load(os.path.join(imageBasePath,file))
        atlas = nib.load(pathToAtlas)
        
        print('Loaded image. Warped atlas name is ' + fileBaseName+'_Atlas.nii.gz')
        
        imageShape = image.shape
        
        imageData = image.get_fdata()
        atlasData = atlas.get_fdata()
        
        
        

        print('Starting label propagation. Right hemisphere labels:')
        
        for n,label in enumerate(atlasLabels["right label"]):
            
            animal.append(fileBaseName)
            roiIndex.append(label)
            roiHemi.append("right")
            roiName.append(atlasLabels["Structure"][n])
            
            tmpData = np.where(atlasData == label)
            
            tmpIndices = np.zeros(imageShape)
            tmpIndices[tmpData] = 1
            tmpIndices = ndimage.binary_erosion(tmpIndices)
            
            tmpData = np.where(tmpIndices == True)
            
            roiImage.append(imageData[tmpData])
            
        print("Done.")
        print("Left hemisphere labels:")
            
        for n,label in enumerate(atlasLabels["left label"]):
            
            animal.append(fileBaseName)
            roiIndex.append(label)
            roiHemi.append("left")
            roiName.append(atlasLabels["Structure"][n])
            
            tmpData = np.where(atlasData == label)
            
            tmpIndices = np.zeros(imageShape)
            tmpIndices[tmpData] = 1
            tmpIndices = ndimage.binary_erosion(tmpIndices)
            
            tmpData = np.where(tmpIndices == True)
            
            roiImage.append(imageData[tmpData])
            
        print("Done.")



d = {"image":animal,
     "structure":roiName,
     "hemisphere":roiHemi,
     "index":roiIndex,
     "values":roiImage}

df = pd.DataFrame(d)

analysisPath = os.path.join(imageBasePath,'Analysis')
if not os.path.exists(analysisPath):
    os.makedirs(analysisPath)

df.to_csv(os.path.join(analysisPath,'VoxelWiseIntensities.csv'))
