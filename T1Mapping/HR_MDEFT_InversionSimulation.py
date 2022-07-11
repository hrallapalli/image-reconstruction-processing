# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:40:30 2022

@author: rallapallih2
"""

import os
import numpy as np
import nibabel as nib

#%%
baseimagepath = r'\\nindsdirfs2\shares\LFMI\FMM\Shared\rallapallih\ScanData\20220503_140933_Zip14_AAV_ICPMS_Round2_B_M1_Zip14_AAV_ICPMS_Round2_B_1_1'
analysispath = os.path.join(baseimagepath, "analysis")

if not os.path.exists(os.path.join(analysispath,'simulation')):
    os.makedirs(os.path.join(analysispath,'simulation'))

pathtomap = os.path.join(analysispath,'test3T1Map.nii.gz')

t1map = nib.load(pathtomap)

MapAffine = t1map.affine

#%%
t1map = t1map.get_fdata()
MapShape = t1map.shape

#%%
SimTimes = np.arange(100,5000,100)
a = 1135
b = -2000

MidSim = np.empty(shape = (MapShape[0],MapShape[1],len(SimTimes)), dtype = 'float64')

for n,t in enumerate(SimTimes):
    print(t)
    tmpSim = np.empty(shape = MapShape, dtype = 'float64')
    
    for i in range(MapShape[0]):
        for j in range(MapShape[1]):
             for k in range(MapShape[2]):
                
                
                tmpSim[i][j][k] = np.absolute(a + b*np.exp(-t/(t1map[i][j][k]*1000)))
                
    MidSim[:,:,n] = tmpSim[:,:,26]

    OutSim = nib.Nifti1Image(tmpSim, MapAffine)
    nib.save(OutSim, os.path.join(analysispath,'simulation','OutSim_'+str(t)+'.nii.gz'))
    

MidSim = nib.Nifti1Image(MidSim, MapAffine)
nib.save(MidSim, os.path.join(analysispath,'simulation','MidSim.nii.gz'))
                          




