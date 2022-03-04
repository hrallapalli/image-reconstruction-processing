
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
    
directorybase = tkinter.filedialog.askdirectory(initialdir = '/opt/PV-360.2.0.pl.1/data/nmr/')

#x = inputdlg(mstring('Enter experiments (space-separated numbers):'), mstring('Experiment numbers'), mcat([1, 50]))
x = tkinter.simpledialog.askstring('Enter experiments (space-separated numbers):', 'Experiment numbers')
x = x.split()
expnos = []
for i in x:
    expnos.append(int(i))
    
totexp = len(expnos)
#t1map=zeros(readnum,phasenum,slicenum,totexp);

plt.show()
while True:
    print('start delay')
    time.sleep(30)
    print('end delay')
    np.sum
    directoryname = directorybase + '/' + str(expnos[0])
#    fmethod = open(directoryname + '/' +  str(expnos[expnum]) + '/method') 
                    
    ncoils = int(readpvpar('##$PVM_EncNReceivers=', directoryname));
    print(ncoils)
                              
    slicethickness = readpvpar('##$PVM_SliceThick=',directoryname);
    print(slicethickness)                          
    slicenum = readpvpar('##$PVM_SPackArrNSlices=',directoryname);
    slicenum = int(slicenum[0])
    print(slicenum)                   
    phaseoffset = readpvpar('##$PVM_SPackArrPhase1Offset=',directoryname);
#    phaseoffset = int(phaseoffset[0])
    print(phaseoffset)                      
    sliceoffset = readpvpar('##$PVM_SPackArrSliceOffset=', directoryname);
#    sliceoffset = int(sliceoffset[0])                       
    print(sliceoffset)                      
    matrixsize = readpvpar('##$PVM_Matrix=', directoryname)  
    print(matrixsize)
    readnum = int(matrixsize[0])
    phasenum = int(matrixsize[1])
    print(readnum,phasenum)
    fovmatrix = readpvpar('##$PVM_Fov=', directoryname)  
    print(fovmatrix)
    readfov = float(fovmatrix[0])
    phasefov = float(fovmatrix[1])
    print(readfov,phasefov)
    phaseoffset = int(phasenum*(phaseoffset[0]/phasefov))
    print(phaseoffset)

    #fidnum = ceil(readnum*ncoils/128.0)*128.0;
    #readnum = 2560
    ncoils = 4
    nechoes = 4
    slicefov = slicenum * slicethickness;
    print(slicefov, sliceoffset[0])
    
 
    fmethod = open(directoryname + '/method')
 
    str1 = fmethod.readline()
    while ('##$PVM_EncSteps1=' not in str1):
#    while (str1[:len(pvstr)-1] != pvstr)
        str1 = fmethod.readline()

#    if (str1(size(str1, 2) - 1) == mstring(')')):
    str1.split('\n')
    print(str1)
    str2list = []
    while (len(str2list) != phasenum):
        str1 = fmethod.readline()
#        str2 = cat(2, str2, strsplit(strtrim(str1)))
        str2list.extend(str1.split())
    phaseorder = [int(phasestep) for phasestep in str2list] 
#    print phaseorder    


    fidnum = int(math.ceil(readnum * ncoils / 128.0) * 128.0)
    fidnum = readnum*ncoils

    a = np.array([2 * fidnum * phasenum * slicenum], dtype = np.int32)
#    d = np.array([readnum, phasenum, slicenum, 1])

    file1 = open(directoryname + '/rawdata.job0')
#   a = fread(file1, size(a), mstring('int32'))
    a=np.fromfile(file1,count=-1,dtype=np.int32)
    slicenum = int(math.floor(np.size(a)/(fidnum*phasenum*2)))
    print(slicenum)
    
    sliceoffset = int(slicenum*(sliceoffset[0]/slicefov))
    print(sliceoffset)
    
    a.resize((fidnum * phasenum * slicenum),2)
    
 #   a = reshape(a, 2, fidnum * phasenum * slicenum * trainnum)
 
#    a = complex(a(1, mslice[:]), a(2, mslice[:]))
    acomplex = 1j*a[:,1] + a[:,0]
    
#    acomplex.resize((fidnum,trainnum,phasenum/nseg,nseg,slicenum))
#    a=np.reshape(acomplex[0:fidnum*phasenum*slicenum],(fidnum,phasenum,slicenum), order = 'F')
    a=np.reshape(acomplex,(readnum,ncoils,phasenum,slicenum), order = 'F')
#    a = reshape(a, fidnum, phasenum / nseg, trainnum, nseg, slicenum)
#    a1 = a(mslice[1:readnum * ncoils], mslice[:], mslice[:])
#    a1 = a[0:readnum*ncoils,:,:]
#    a = np.reshape(a1,(readnum, ncoils, phasenum, slicenum))
    #rescale coil 2 if necessary
    #a(:,2,:,:,:,:) = a(:,2,:,:,:,:);
#    del a1
    
#    atrain1 = zeros(readnum, ncoils, phasenum, slicenum)
    atranspose = np.transpose(a,(0, 2, 3, 1))
    np.fft.fftshift(atranspose,axes=(0, 1, 2))
    afft = np.fft.fftshift(np.fft.fftn(atranspose,axes=(0, 1, 2)),axes=(0, 1, 2))
    #np.fft.fftshift(atrain1fft,axes=(0,2,3))
#    afft = np.fft.fftn(atranspose,axes=(0, 1, 2))
    #shift for slice and phase

    sliceshift = int(np.round((sliceoffset * slicenum/matrixsize[2])))
    print(sliceshift)
    phaseshift = int(np.round((phaseoffset / phasefov) * phasenum)) 
    print(phaseshift)
    
    afftroll = np.roll(afft,-phaseoffset,axis = 1);
    afft = np.roll(afftroll,sliceshift,axis = 2);         

    b=(abs(afft))
    
    plt.imshow(b[readnum//2,:,:,0], cmap='gray',vmax = np.max(b[readnum//2,:,:,2]))
    plt.pause(.001)
 #   plt.draw()

    
    
    


