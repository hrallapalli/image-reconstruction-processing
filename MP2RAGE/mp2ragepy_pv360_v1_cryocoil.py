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
    file1 = open(directoryname + '/pdata/2/2dseq')
#   a = fread(file1, size(a), mstring('int32'))
    fid=np.fromfile(file1,dtype=np.int16)
#    a=fid
    a = np.reshape(fid,((fidnum * phasenum * slicenum * trainnum),2))
 #   a = reshape(a, 2, fidnum * phasenum * slicenum * trainnum)
 
#    a = complex(a(1, mslice[:]), a(2, mslice[:]))
    acomplex = 1j*a[:,1] + a[:,0]

# check modified 08-09-21 sd
    a = np.reshape(fid,(2,ncoils,trainnum,slicenum, phasenum, readnum))
    acomplex = 1j*a[1,:,:,:,:,:] + a[0,:,:,:,:,:]
    atrain1fft = np.squeeze(acomplex[:,0,:,:,:])
    atrain2fft = np.squeeze(acomplex[:,1,:,:,:])
    imc = np.zeros([slicenum,phasenum,readnum])
    for i in range(ncoils):
        imc = imc + atrain1fft[i,:,:,:]*np.conj(atrain2fft[i,:,:,:])
    cosmap = np.cos(np.angle(imc))
    im = np.reshape(np.squeeze(a[:,:,0,:,:,:]),(2*ncoils,readnum*phasenum*slicenum))
    im = np.asarray(im,dtype = float)
    mag1 = np.sqrt(np.reshape(np.mean(im*im,axis=0),(slicenum,phasenum,readnum)))
    im = np.reshape(np.squeeze(a[:,:,1,:,:,:]),(2*ncoils,readnum*phasenum*slicenum))
    im = np.asarray(im,dtype = float)
    mag2 = np.sqrt(np.reshape(np.mean(im*im,axis=0),(slicenum,phasenum,readnum)))

    mp2ragenewimg = (mag1*mag2*cosmap)/(mag1*mag1 + mag2*mag2)

#from frank's idl code    
# default values of some pars, in case they are not set in input.par
    Thr_cosMap = 1.0    	# To specify cos(ph) threshold for making mask, default value 0.5
		    	# if M1 > M2, and cos(ph) far from -1, then it's likely a noise voxel
    Thr_s 	= 10.0      	# To specify tuning parameter Thr as Thr_s*std
#verbose = 0		; default as 0, to turn on/off informational output
#reco_num= 2		; by default, reco 2 contains the complex data
#reco_output_num1= 3	; by default, reco 3 contains the float real number
#reco_output_num2= 4	; by default, reco 4 contains the float real number
#eff 	= 0.97		; effective inversion factor
#T1_min	= 700.0		; T1 window, minimum
#T1_max 	= 4200.0	; T1 window, maximum
#T1_precision = 4	; precision is upto the fraction of 1/T1_precision milisecond 
#save_path=[]		; null
#AFNI_orientation = 1    ;
   

#define noise map
    nmap = np.logical_and(np.greater(mag1, mag2), np.less(abs(cosmap),Thr_cosMap))
#    std_nmap = np.std([mag1(where(nmap GT 0)), mag2(where(nmap GT 0))])  
    std_nmap = np.std(np.concatenate((mag1[np.nonzero(nmap)],mag2[np.nonzero(nmap)])))
    thr = Thr_s * std_nmap

#robust ratio images -fye
#calculated but not used yet
    bb = thr * thr
    rmp2rage = (mag1*mag2*cosmap- bb) /(mag1*mag1+ mag2*mag2 + 2*bb) 
    rRatio   =  (mag1*mag2*cosmap- bb) /( mag2*mag2 + bb) 

    
    #threshold mask    
 #   atrain2sumfilt = medfilt1(atrain2sum, 3, mcat([]), 1)
 #   atrain2sumfilt = medfilt1(atrain2sumfilt, 3, mcat([]), 2)
 #   atrain2sumfilt = medfilt1(atrain2sumfilt, 3, mcat([]), 3)
 #   maxsum2 = max(max(max(atrain2sum)))
    maxsum2 = np.amax(mag2)
    mp2ragemask = abs(mag2) > thr

#   mp2ragewa = (mp2ragesos* abs(atrain2fft[:,:,:,0]))/atrain2sum

    mp2ragewamask = (mp2ragenewimg + 0.5) * mp2ragemask - 0.5
    
    #t1map
    t1map = np.zeros([slicenum,phasenum,readnum],dtype=np.float)
    alpha = 2.0 * 3.141592654 * 8.0 / 360.0# flipangle (set both to be the same at the moment) (converted to radians)
    alpha2 = 2.0 * 3.141592654 * 8.0 / 360.0
    mp2ragetr = 6.0
    tr = 10.0e-3# tr for gre part
    eff = .90# inversion efficiency
    ta = ti1 - (segduration / 2.0 - tr)
    tb = (ti2 - ti1) - segduration
    tc = mp2ragetr - (ti2 + segduration / 2.0); print(tc)
    n = phasenum / nseg#rf pulses in ge train
    t1array = np.zeros([60,],dtype=float)
    mzss = np.zeros([60,],dtype=float)
    greti1 = np.zeros([60,],dtype=float)
    greti2 = np.zeros([60,],dtype=float)  
    mp2ragecalc = np.zeros([60,],dtype=float)
    m0 = 1.0

    # calculate steady state magnetization
    for ii in range(1,60):
        t1 = 0.1 + (ii - 1) * 0.1
        t1array[ii] = t1
        ea = math.exp(-ta / t1)
        eb = math.exp(-tb / t1)
        ec = math.exp(-tc / t1)
        e1 = math.exp(-tr / t1)
        cosalpha1e1n = (math.cos(alpha) * e1) ** n
        firstbracket = (1.0 - ea) * cosalpha1e1n + (1.0 - e1) * ((1.0 - cosalpha1e1n) / (1.0 - math.cos(alpha) * e1))
        secondbracket = firstbracket * eb + (1.0 - eb)
        thirdbracket = secondbracket * ((math.cos(alpha) * e1) ** n) + (1.0 - e1) * ((1.0 - (math.cos(alpha) * e1) ** n) / (1.0 - math.cos(alpha) * e1))
        denom = 1.0 + eff * ((math.cos(alpha) * math.cos(alpha2)) ** n) * math.exp(-mp2ragetr / t1)
        mzss[ii] = (thirdbracket * ec + 1.0 - ec) / denom

        # calculate greti1 signal
        greti1[ii] = math.sin(alpha) * (((-eff * mzss[ii] * ea) + 1.0 - ea) * (math.cos(alpha) * e1) ** (n / 2 - 1) + (1.0 - e1) * ((1.0 - (math.cos(alpha) * e1) ** (n / 2 - 1)) / (1.0 - math.cos(alpha) * e1)))

        # calculate greti2 signal
        greti2[ii] = math.sin(alpha2) * ((mzss[ii] - (1.0 - ec)) / (ec * (math.cos(alpha2) * e1) ** (n / 2)) - (1.0 - e1) * (((math.cos(alpha2) * e1) ** (-n / 2) - 1.0) / (1.0 - math.cos(alpha2) * e1)))

        # calculate mp2rage signal
        mp2ragecalc[ii] = greti1[ii] * greti2[ii] / (greti1[ii] ** 2 + greti2[ii] ** 2)


    t1func = interp1d(mp2ragecalc,t1array,bounds_error=False,fill_value = 0.0)
    for ii in range(slicenum):
    #         ii
         for jj in range(phasenum):
             for kk in range(readnum):
                if mp2ragemask[ii,jj,kk]:
#                  t1func = interp1d(t1array,mp2ragecalc)
                  mp2ragevalue = mp2ragewamask[ii,jj,kk]
                  t1map[ii,jj,kk] = t1func(mp2ragevalue)

    fid = open(directoryname + '/mp2rageimg.raw','wb')
    # write out data in ms
    fid.write(mp2ragewamask * 1000.0)
    fid.close
    
    fid = open(directoryname + '/t1mapimg.raw','wb')
    # write out data in ms
    fid.write(t1map * 1000.0)
    fid.close
    
    # r1 map
    #fid = open(directoryname + '/r1mapimg.raw','wb')
    #fid.write(1.0/t1map)
    #fid.close    
    
    # output to afni 
    #afnistr = 'to3d -anat -prefix r' + str(expnum) + ' -xFOV ' + str(readfov/2.0) + 'S-I -yFOV ' + str(phasefov/2.0) + 'L-R -zFOV ' + str(slicefov / 2.0) + 'A-P 3Df:0:0:' + str(readnum) + ':' + str(phasenum) + ':' + str(slicenum) + ':mp2rageimg'
    #print(afnistr)
#    cd(directoryname)
#    num2str(expnum)
#    afnicmd = strcat(mcat([mstring('to3d -anat -prefix r'), num2str(expnum), mstring(' -xFOV '), num2str(readfov / 2.0), mstring('S-I -yFOV '), num2str(phasefov / 2.0), mstring('L-R -zFOV '), num2str(slicefov / 2.0), mstring('A-P 3Df:0:0:'), int2str(readnum), mstring(':'), int2str(phasenum), mstring(':'), int2str(slicenum), mstring(':t1map')])); print afnicmd
    #os.system(afnistr)
    #     status = system('align_script.sh');

#status = system(mstring('align_script.sh'))