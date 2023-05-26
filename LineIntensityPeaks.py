# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:20:18 2023

@author: rallapallih2
"""

import numpy as np
from scipy.signal import chirp, find_peaks, peak_widths, savgol_filter
import matplotlib.pyplot as plt
import pandas as pd

peaks_out = np.array([])
interpeak_out = np.array([])
widths_out = np.array([])
line_id = 0

FILE = r'\Users\rallapallih2\Desktop\test.csv'

df = pd.read_csv(FILE)

for column in df:
    x = df[column].values.flatten()
    x = x[~np.isnan(x)]
    
    ### Turn this on for histology
    
    # x = savgol_filter(x, window_length = 101, polyorder = 5)
    # peaks, _ = find_peaks(x, width = 30)
    # resolution = 1.6
    
    
    ### Turn this on for MRI
    
    peaks, _ = find_peaks(x, width = 3)
    resolution = 80/3
    
    results_half = peak_widths(x, peaks, rel_height=0.5)
    #results_half[0]  # widths
    results_full = peak_widths(x, peaks, rel_height=1)
    #results_full[0]  # widths
#%%
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.hlines(*results_half[1:], color="C2")
    # plt.hlines(*results_full[1:], color="C3")
    plt.show()

#%%
    
    
    
    peaks_vector = resolution*peaks
    results_half_vector = resolution*results_half[0]
    results_full_vector = resolution*results_full[0]
    
    #%%
    interpeak = np.diff(peaks_vector)
    interpeak = np.insert(interpeak,0,interpeak[0])
    
    #%%
    peaks_out = np.append(peaks_out,peaks_vector)
    interpeak_out = np.append(interpeak_out,interpeak)
    widths_out = np.append(widths_out, results_half_vector)
    

#%%
plt.figure()
plt.scatter(interpeak_out, widths_out)
plt.xlabel('Inter-peak distance (microns)')
plt.ylabel('Peak width (microns)')
plt.show()