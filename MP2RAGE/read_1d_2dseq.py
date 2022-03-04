#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:18:45 2017

@author: doddst
"""

# mp2rage reco python; start with m2py converter
# start with complex reconstruction
# needs complex reco from bruker with individual coils
# adapt from frank ye's idl code as much as possible


from readpvpar import readpvpar
from scipy.interpolate import interp1d
import tkinter.filedialog, tkinter.simpledialog
import math 
import numpy as np
import os
import matplotlib.pyplot as plt

trainnum = 2
#nseg = 4;

# pick experiment number for directory name
directorybase = tkinter.filedialog.askdirectory(initialdir = '/opt/PV-360.2.0.pl.1/data/nmr')
#directorybase = tkFileDialog.askdirectory(initialdir = '/home/doddst/data/20190117_095902_Exvivo_MouseBrain_Training_01_1_7_MP2RAGE/')
#x = inputdlg(mstring('Enter experiments (space-separated numbers):'), mstring('Experiment numbers'), mcat([1, 50]))
#x = tkSimpleDialog.askstring('Enter experiments (space-separated numbers):', 'Experiment numbers')
x = tkinter.simpledialog.askstring('Enter experiments (space-separated numbers):', 'Experiment numbers')
x = x.split()
expnos = []
for i in x:
    expnos.append(int(i))
    
totexp = len(expnos)
#t1map=zeros(readnum,phasenum,slicenum,totexp);

for expnum in range(len(expnos)):
    directoryname = directorybase + '/' + str(expnos[expnum])
#    fmethod = open(directoryname + '/' +  str(expnos[expnum]) + '/method')
    nseg = int(readpvpar('##$SegmNumber=', directoryname));
    print(nseg)
    
    segduration = readpvpar('##$SegmDuration=', directoryname);
    segduration = segduration/1000.0
    print(segduration)  
                    
    ncoils = int(readpvpar('##$PVM_EncNReceivers=', directoryname));
    print(ncoils)

    ti1=readpvpar('##$PVM_InversionTime=', directoryname)/1000.0;
    print(ti1)
    ti2 = readpvpar('##$TI2=', directoryname)/1000.0;
    print(ti2)              
 #   alphadeg = readpvpar('##$MP2RAGE_1stExcPulseAngle=', directoryname);
    alphadeg = 10.0 
    print(alphadeg)                    
 #   alpha2deg = readpvpar('##$MP2RAGE_2ndExcPulseAngle=', directoryname);
    alpha2deg = 10.0
    print(alpha2deg)
    slicethickness = readpvpar('##$PVM_SliceThick=',directoryname);
    print(slicethickness)                          
    slicenum = readpvpar('##$PVM_SPackArrNSlices=',directoryname);
    slicenum = int(slicenum[0])
    print(slicenum)                   
    phaseoffset = readpvpar('##$PVM_SPackArrPhase1Offset=',directoryname);
    phaseoffset = int(phaseoffset[0])
    print(phaseoffset)                      
    sliceoffset = readpvpar('##$PVM_SPackArrSliceOffset=', directoryname);
    sliceoffset = int(sliceoffset[0])                       
    print(sliceoffset)                      
    mp2ragetr = readpvpar('##$SegmRepTime=', directoryname)/1000.0;
    print(mp2ragetr)
    tr = readpvpar('##$EchoRepTime=', directoryname)/1000.0;
    print(tr) 
    matrixsize = readpvpar('##$PVM_Matrix=', directoryname)  
    print(matrixsize)
    readnum = int(matrixsize[0])
    phasenum = int(matrixsize[1])
    slicenum = int(matrixsize[2])
    print(readnum,phasenum)
    fovmatrix = readpvpar('##$PVM_Fov=', directoryname)  
    print(fovmatrix)
    readfov = float(matrixsize[0])
    phasefov = float(matrixsize[1])
    print(readfov,phasefov)


    #fidnum = ceil(readnum*ncoils/128.0)*128.0;


    slicefov = slicenum * slicethickness;
    print(slicefov)
  


    fidnum = int(math.ceil(readnum * ncoils / 128.0) * 128.0)
    fidnum = readnum*ncoils

    phaseonseg = int(phasenum/nseg)

#    a = np.array([2 * fidnum * phasenum * slicenum * trainnum], dtype = np.int32)
    fid = np.array([2 * fidnum * phasenum * slicenum * trainnum])
    d = np.array([readnum, phasenum, slicenum, 1])

#    file1 = open(directoryname + '/rawdata.job0')
    file1 = open(directoryname + '/pdata/1/2dseq')
#   a = fread(file1, size(a), mstring('int32'))
    fid=np.fromfile(file1,dtype=np.int16)
#    a=fid
#    a = np.reshape(fid,((fidnum * phasenum * slicenum * trainnum),2))
 #   a = reshape(a, 2, fidnum * phasenum * slicenum * trainnum)
 
#    a = complex(a(1, mslice[:]), a(2, mslice[:]))
#    acomplex = 1j*a[:,1] + a[:,0]

# check modified 08-09-21 sd
