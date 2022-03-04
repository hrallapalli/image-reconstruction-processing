
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:18:45 2017

@author: doddst
"""

# reco update for accumulating 3d flash image

#import tkFileDialog, tkSimpleDialog
import tkinter.filedialog, tkinter.simpledialog
import math 
import numpy as np
import time
import matplotlib.pyplot as plt



def readpvpar(pvstr, directoryname):

    fmethod = open(directoryname + '/method')
    str1 = fmethod.readline()

    while (pvstr not in str1):
#    while (str1[:len(pvstr)-1] != pvstr)
        str1 = fmethod.readline()

#    if (str1(size(str1, 2) - 1) == mstring(')')):
    str1.split('\n')
    print(str1)
    if str1.endswith(')\n'):
        str1 = fmethod.readline()
        str2 = ''
#        str2 = cat(2, str2, strsplit(strtrim(str1)))
        str2 = str1.split()
        parametername = [float(x) for x in str2]     
    else:
        str2 = ''
#        str2 = strsplit(str1, mstring('='))
        str2 = str1.split('=')
        print(str2)
        parametername = float(str2[1])

    return parametername




# pick experiment number for directory name
#directoryname = uigetdir(mstring('/home/doddst/data/EPmemri_grp3_mouse2.E61'), mstring('Pick a Directory'))
    
directorybase = tkinter.filedialog.askdirectory(initialdir = '/home/nmr')

#x = inputdlg(mstring('Enter experiments (space-separated numbers):'), mstring('Experiment numbers'), mcat([1, 50]))
#x = tkinter.simpledialog.askstring('Enter experiments (space-separated numbers):', 'Experiment numbers')
#x = x.split()
#expnos = []
#for i in x:
#    expnos.append(int(i))
    
#totexp = len(expnos)
#t1map=zeros(readnum,phasenum,slicenum,totexp);

startfilenum = 5
endfilenum = 50
specnum=256

b = np.ndarray(shape=(specnum,(endfilenum-startfilenum+1)), dtype = np.complex128)

for i in range(startfilenum,endfilenum):
    
    a = np.array(2*specnum,dtype = np.int32) 
    directoryname = directorybase + '/' + str(i)
    file1 = open(directoryname + '/rawdata.job0')
#   a = fread(file1, size(a), mstring('int32'))
    a=np.fromfile(file1,count=-1,dtype=np.int32)
    a.resize(specnum,2)
    acomplex = 1j*a[:,1] + a[:,0]
    b[:,i-startfilenum] = acomplex
    
  
    
    


