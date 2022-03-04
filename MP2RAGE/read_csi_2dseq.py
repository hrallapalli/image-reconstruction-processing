
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
    time.sleep(1)
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
    matrixsize = readpvpar('##$PVM_EncMatrix=', directoryname)  
    print(matrixsize)
    readnum = int(matrixsize[0])
    phasenum = int(matrixsize[1])
#    slicenum = int(matrixsize[2])
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
    ncoils = 1
    nreps = 16

#    slicefov = slicenum * slicethickness;
#    print(slicefov, sliceoffset[0])
    
 
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

#    readnum = 465

    fidnum = int(math.ceil(readnum * ncoils / 128.0) * 128.0)
    fidnum = readnum
    specnum = 256
    a = np.array([2 * fidnum * phasenum * specnum], dtype = np.int16)
#    d = np.array([readnum, phasenum, slicenum, 1])

    file1 = open(directoryname + '/pdata/1/2dseq')
#   a = fread(file1, size(a), mstring('int32'))
    a=np.fromfile(file1,count=-1,dtype=np.int16)
    b=np.reshape(a,(readnum,phasenum,specnum))
    
#    slicenum = int(math.floor(np.size(a)/(fidnum*phasenum*2)))
    print('current number of slices is ', slicenum)
    
    sliceoffset = int(slicenum*(sliceoffset[0]/slicefov))
    print(sliceoffset)
    
#    a.resize((fidnum * phasenum * slicenum),2)
    
    a = np.reshape(a[0:fidnum*phasenum*slicenum*nreps], (fidnum * phasenum * specnum * nreps,1))
 
#    a = complex(a(1, mslice[:]), a(2, mslice[:]))
    acomplex = 1j*a[:,1] + a[:,0]
    
#    acomplex.resize((fidnum,trainnum,phasenum/nseg,nseg,slicenum))
#    a=np.reshape(acomplex[0:fidnum*phasenum*slicenum],(fidnum,phasenum,slicenum), order = 'F')
    a=np.reshape(acomplex,(readnum,phasenum,slicenum,nreps), order = 'F')
#    a = reshape(a, fidnum, phasenum / nseg, trainnum, nseg, slicenum)
#    a1 = a(mslice[1:readnum * ncoils], mslice[:], mslice[:])
#    a1 = a[0:readnum*ncoils,:,:]
#    a = np.reshape(a1,(readnum, ncoils, phasenum, slicenum))
    #rescale coil 2 if necessary
    #a(:,2,:,:,:,:) = a(:,2,:,:,:,:);
#    del a1
    
#    atrain1 = zeros(readnum, ncoils, phasenum, slicenum)
#    atranspose = np.transpose(a,(0, 2, 3, 1))
    np.fft.fftshift(a,axes=(0, 1, 2))
    afft = np.fft.fftshift(np.fft.fftn(a,axes=(0, 1, 2)),axes=(0, 1, 2))
    #np.fft.fftshift(atrain1fft,axes=(0,2,3))
#    afft = np.fft.fftn(atranspose,axes=(0, 1, 2))
    #shift for slice and phase

    sliceshift = int(np.round((sliceoffset * slicenum/matrixsize[2])))
    print(sliceshift)
    phaseshift = int(np.round((phaseoffset / phasefov) * phasenum)) 
    print(phaseshift)
    
 #   afftroll = np.roll(afft,-phaseoffset,axis = 1);
 #   afft = np.roll(afftroll,sliceshift,axis = 2);         

    b=(abs(afft))
    
# find edge of brain for motion correction, 1 dimension
    slide_edge = np.array([nreps],dtype = np.int32)
    
    slopeb = np.diff(np.sum(b[188:192,125,30:100,:],axis=0),axis=0)
    result = np.argmax(slopeb,axis = 0)
    
#   add phaseshift along slice direction based on edge shift

    phaseshift = -0.03
    aphase = np.angle(a)
    ashift = a
    i=1
    k=1
    for i in range(nreps):
        for k in range(slicenum):
            ashift[:,:,k,i] = np.abs(a[:,:,k,i])*(np.cos(aphase[:,:,k,i] + phaseshift*result[i]*k) + 1j*np.sin(aphase[:,:,k,i] + phaseshift*result[i]*k))    

    np.fft.fftshift(ashift,axes=(0, 1, 2))
    asliceshiftfft = np.fft.fftshift(np.fft.fftn(ashift,axes=(0, 1, 2)),axes=(0, 1, 2))
    bshift = abs(asliceshiftfft)
#smooth along time axis    

    bshiftsmooth = np.apply_along_axis(lambda m: np.convolve(m,np.ones(2),mode='full'),axis = 3,arr=bshift)
    
    fid = open(directoryname + '/timecourseimg.raw','wb')
    bshift = bshift.copy(order = 'C')
    bshift = bshift*(30000.0/np.max(bshift))
    bshift = bshift.astype(dtype = np.int32)
    fid.write(bshift)
    fid.close()
    
    fid = open(directoryname + '/timecourseshiftsmoothimg.raw','wb')
    bshiftsmooth = bshiftsmooth.copy(order = 'C')
    bshiftsmooth = bshiftsmooth*(30000.0/np.max(bshiftsmooth))
    bshiftsmooth = bshiftsmooth.astype(dtype = np.int32)
    fid.write(bshiftsmooth)
    fid.close()   
    
    plt.imshow(b[:,:,slicenum//3], cmap='gray',vmax = np.max(0.7*b[readnum//4,:,:]))
    plt.pause(.001)
 #   plt.draw()

    
    
    


