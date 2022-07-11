import glob
import numpy as np
import nibabel as nib
import csv
import os
import datetime
import sys


np.set_printoptions(threshold=sys.maxsize)

map_path = r'C:\Users\rallapallih2\Desktop\MEMRI-timecourse-registration' + '\\'
atlas_path = r'C:\Users\rallapallih2\Desktop\MEMRI-timecourse-registration\labels' + '\\'

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
fieldnames = ['Filename',','ex_tissue','lh_noise','lh_s1','rh_noise','rh_s1']

# atlas_allseg = nib.load((atlas_path + 'allsegmentations.nii'))  
atlas_ex_tissue = nib.load((atlas_path + 'ex_tissue.nii'))
atlas_lh_noise = nib.load((atlas_path + 'lh_noise.nii'))
atlas_lh_s1 = nib.load((atlas_path + 'lh_s1.nii'))
atlas_rh_noise = nib.load((atlas_path + 'rh_noise.nii'))
atlas_rh_s1 = nib.load((atlas_path + 'rh_s1.nii'))


# atlas_data_allseg = np.squeeze(np.array(atlas_allseg.get_fdata(), dtype = np.bool))
atlas_data_ex_tissue = np.squeeze(np.array(atlas_ex_tissue.get_fdata(), dtype = np.bool))
atlas_data_lh_noise = np.squeeze(np.array(atlas_lh_noise.get_fdata(), dtype = np.bool))
atlas_data_lh_s1 = np.squeeze(np.array(atlas_lh_s1.get_fdata(), dtype = np.bool))
atlas_data_rh_noise = np.squeeze(np.array(atlas_rh_noise.get_fdata(), dtype = np.bool))
atlas_data_rh_s1 = np.squeeze(np.array(atlas_rh_s1.get_fdata(), dtype = np.bool))



with open(r'C:\Users\rallapallih2\Desktop\MEMRI-timecourse-registration\analysis\Intensities_'+ timestamp + '.csv', 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter = ',')
    writer.writeheader()

    for file in glob.glob(map_path+"*.nii.gz"):
	
        mouseID = os.path.basename(file)[:-7]
        map_name = file
        map = nib.load((file))
        map_data = np.asarray(map.get_fdata(), dtype = np.float32)
        sizer = np.asarray(map.shape)

        
        # dat_allseg = map_data[atlas_data_allseg].flatten()
        dat_ex_tissue = map_data[atlas_data_ex_tissue].flatten()
        dat_lh_noise = map_data[atlas_data_lh_noise].flatten()
        dat_lh_s1 = map_data[atlas_data_lh_s1].flatten()
        dat_rh_noise = map_data[atlas_data_rh_noise].flatten()
        dat_rh_s1 = map_data[atlas_data_rh_s1].flatten()
        
        
       
        writer.writerow({'Filename':mouseID ,'ex_tissue':dat_ex_tissue,'lh_noise':dat_lh_noise ,'lh_s1':dat_lh_s1 ,'rh_noise':dat_rh_noise ,'rh_s1':dat_rh_s1})

        print(file)