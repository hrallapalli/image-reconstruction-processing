import glob
import numpy as np
import nibabel as nib
import csv
import os
import datetime
import sys


np.set_printoptions(threshold=sys.maxsize)

map_path = r'G:\MINCVM\PCLKO\PCP2-DTR\maps\Extracted' + '\\'
atlas_path = r'G:\MINCVM\PCLKO\PCP2-DTR\seg' + '\\'

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
fieldnames = ['Filename','PCL','WM','ML','IGL','Reference']

with open(r'G:\MINCVM\PCLKO\PCP2-DTR\results\MAPPING_RESULTS_'+ timestamp + '.csv', 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter = ',')
    writer.writeheader()

    for file in glob.glob(map_path+"*.nii.gz"):
	
        mouseID = os.path.basename(file)[:-25]
        map_name = file
        map = nib.load((file))
        map_data = np.asarray(map.get_fdata(), dtype = np.float32)
        sizer = np.asarray(map.shape)
		
        atlas_PCL = nib.load((atlas_path + mouseID + 'PCL_Labels.nii'))
        atlas_WM = nib.load((atlas_path + mouseID + 'WM_Labels.nii'))        
        atlas_IGL = nib.load((atlas_path + mouseID + 'IGL_Labels.nii'))        
        atlas_ML = nib.load((atlas_path + mouseID + 'ML_Labels.nii'))
        atlas_Reference = nib.load((atlas_path + mouseID + 'Reference_Labels.nii'))
        
        # atlas_PCL = plt.imread((atlas_path + mouseID + 'PCL_Labels.tif'))
        # atlas_WM = plt.imread((atlas_path + mouseID + 'WM_Labels.tif'))        
        # atlas_IGL = plt.imread((atlas_path + mouseID + 'IGL_Labels.tif'))        
        # atlas_ML = plt.imread((atlas_path + mouseID +  'ML_Labels.tif'))
        
        atlas_data_PCL = np.squeeze(np.array(atlas_PCL.get_fdata(), dtype = np.bool))
        atlas_data_WM = np.squeeze(np.array(atlas_WM.get_fdata(), dtype = np.bool))
        atlas_data_IGL = np.squeeze(np.array(atlas_IGL.get_fdata(), dtype = np.bool))		
        atlas_data_ML = np.squeeze(np.array(atlas_ML.get_fdata(), dtype = np.bool))

             
        atlas_data_Reference = np.squeeze(np.array(atlas_Reference.get_fdata(), dtype = np.bool))
        
        # dat_PCL = map_data[np.transpose(atlas_data_PCL)].flatten()
        # dat_WM = map_data[np.transpose(atlas_data_WM)].flatten()
        # dat_IGL = map_data[np.transpose(atlas_data_IGL)].flatten()
        # dat_ML = map_data[np.transpose(atlas_data_ML)].flatten()
        
        
        dat_PCL = map_data[atlas_data_PCL].flatten()
        dat_WM = map_data[atlas_data_WM].flatten()
        dat_IGL = map_data[atlas_data_IGL].flatten()
        dat_ML = map_data[atlas_data_ML].flatten()
        dat_Reference = map_data[atlas_data_Reference].flatten()
        
        writer.writerow({'Filename':file,'PCL':dat_PCL,'WM':dat_WM,'ML':dat_ML,'IGL':dat_IGL, 'Reference':dat_Reference})
        print(file)