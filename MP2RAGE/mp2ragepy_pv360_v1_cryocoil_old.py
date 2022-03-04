#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:18:45 2017

@author: doddst
"""

# mp2rage reco python; start with m2py converter
# start with complex reconstruction

#readnum=256;
#ncoils=2;
#fidnum = ceil(readnum*ncoils/128.0)*128.0;
#phasenum=256;
#slicenum=128;
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
directorybase = tkinter.filedialog.askdirectory(initialdir = '/opt/PV-360.2.0.pl.1')
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

#no need for 2dseq file 
#    fmethod = open(directoryname + '/method')
 
#    str1 = fmethod.readline()
#    while ('##$PVM_EncSteps1=' not in str1):
#    while (str1[:len(pvstr)-1] != pvstr)
#        str1 = fmethod.readline()

#    if (str1(size(str1, 2) - 1) == mstring(')')):
#    str1.split('\n')
#    print(str1)
#    str2list = []
#    while (len(str2list) != phasenum):
#        str1 = fmethod.readline()
#        str2 = cat(2, str2, strsplit(strtrim(str1)))
#        str2list.extend(str1.split())
#    phaseorder = [int(phasestep) for phasestep in str2list] 
#    print(phaseorder)    


    fidnum = int(math.ceil(readnum * ncoils / 128.0) * 128.0)
    fidnum = readnum*ncoils

    phaseonseg = int(phasenum/nseg)

#    a = np.array([2 * fidnum * phasenum * slicenum * trainnum], dtype = np.int32)
    fid = np.array([2 * fidnum * phasenum * slicenum * trainnum])
    d = np.array([readnum, phasenum, slicenum, 1])

#    file1 = open(directoryname + '/rawdata.job0')
    file1 = open(directoryname + '/pdata/3/2dseq')
#   a = fread(file1, size(a), mstring('int32'))
    fid=np.fromfile(file1,dtype=np.int16)
#    a=fid
    a = np.reshape(fid,((fidnum * phasenum * slicenum * trainnum),2))
 #   a = reshape(a, 2, fidnum * phasenum * slicenum * trainnum)
 
#    a = complex(a(1, mslice[:]), a(2, mslice[:]))
    acomplex = 1j*a[:,1] + a[:,0]

# check modified 08-09-21 sd
    a = np.reshape(fid,(2,ncoils,trainnum,slicenum, readnum, phasenum))
    acomplex = 1j*a[1,:,:,:,:,:] + a[0,:,:,:,:,:]
    atrain1fft = np.squeeze(acomplex[:,0,:,:,:])
    atrain2fft = np.squeeze(acomplex[:,1,:,:,:])
    imc = np.zeros([slicenum,readnum,phasenum])
    for i in range(ncoils):
        imc = imc + atrain1fft[i,:,:,:]*np.conj(atrain1fft[i,:,:,:])
    cosmap = np.cos(np.angle(imc))
    im = np.reshape(np.squeeze(a[:,:,0,:,:,:]),(2*ncoils,readnum*phasenum*slicenum))
    im = np.asarray(im,dtype = float)
    mag1 = np.sqrt(np.reshape(np.mean(im*im,axis=0),(slicenum,readnum,phasenum)))
    im = np.reshape(np.squeeze(a[:,:,1,:,:,:]),(2*ncoils,readnum*phasenum*slicenum))
    im = np.asarray(im,dtype = float)
    mag2 = np.sqrt(np.reshape(np.mean(im*im,axis=0),(slicenum,readnum,phasenum)))

    mp2ragenewimg = (mag1*mag2*cosmap)/(mag1*mag1 + mag2*mag2)
    
#    acomplex.resize((fidnum,trainnum,phasenum/nseg,nseg,slicenum))
#    a=np.reshape(acomplex,(readnum,ncoils,phaseonseg,trainnum,nseg,slicenum), order = 'F')
#    a = reshape(a, fidnum, phasenum / nseg, trainnum, nseg, slicenum)
#    a1 = a(mslice[1:readnum * ncoils], mslice[:], mslice[:])
#   a1 = a[0:readnum*ncoils,:,:,:,:]
#    a = np.reshape(a1,(readnum, ncoils, phasenum / nseg, trainnum, nseg,slicenum),order='F')
    #rescale coil 2 if necessary
    #a(:,2,:,:,:,:) = a(:,2,:,:,:,:);
#    a1=1
    
#    atrain1 = zeros(readnum, ncoils, phasenum, slicenum)
    atrain1 = np.zeros([readnum,ncoils,phasenum,slicenum],dtype=np.complex)

    atrain2 = np.zeros([readnum,ncoils,phasenum,slicenum],dtype=np.complex)
    # split up images and reorder phase encodes
   # for ii in range(nseg):
   #     for jj in range(phaseonseg):
 #           print(ii,jj,(jj + (ii) * phaseonseg),phaseorder[(jj + (ii) * phaseonseg)] + (phasenum // 2))
#            atrain1[:,:, phaseorder[(jj + (ii - 1) * phasenum / nseg)] + (phasenum / 2), :] = a[:,:, jj,0, ii,:]
 #           atrain1[:,:, phaseorder[(jj + (ii) * phaseonseg)] + (phasenum // 2), :] = a[:,:,jj,0,ii,:] 
            
            
            
  #  for ii in range(nseg):
  #      for jj in range(phaseonseg):
  #          atrain2[:,:, phaseorder[(jj + (ii) * phaseonseg)] + (phasenum // 2), :] = a[:,:,jj,1,ii,:]
    
    np.fft.fftshift(atrain1,axes=(0,2,3))    
    atrain1 = np.transpose(atrain1,axes = (0,2,3,1))
#    atrain1fft = np.fft.fftshift(np.fft.fftn(atrain1[:,:,:,:],axes=(0,1,2)),axes=(0,1,2))
    #np.fft.fftshift(atrain1fft,axes=(0,2,3))
    np.fft.fftshift(atrain2,axes=(0,2,3)) 
    atrain2 = np.transpose(atrain2,axes = (0,2,3,1))
#    atrain2fft = np.fft.fftshift(np.fft.fftn(atrain2[:,:,:,:],axes=(0,1,2)),axes=(0,1,2))
#    np.fft.fftshift(atrain2fft,axes=(0,2,3))

    #shift for slice and phase

    sliceshift = int(np.round((sliceoffset / slicefov) * slicenum))
    print(sliceshift)
    phaseshift = int(np.round((phaseoffset / phasefov) * phasenum)) 
    print(phaseshift)
    
 #   atrain1roll = np.roll(atrain1fft,phaseshift,axis = 1);
 #   atrain1fft = np.roll(atrain1roll,sliceshift,axis = 2);       
 #   atrain2roll = np.roll(atrain2fft,phaseshift,axis = 1);
 #   atrain2fft = np.roll(atrain2roll,sliceshift,axis = 2);  
#    atrain1fft = circshift(atrain1fft, mcat([0, 0 - phaseshift, sliceshift]))
#    atrain2fft = circshift(atrain2fft, mcat([0, 0 - phaseshift, sliceshift]))

    #combine for mprage
#    maxsignal1 = max(max(max(max(abs(atrain1fft)))))
#    maxsignal2 = max(max(max(max(abs(atrain2fft)))))
    maxsignal1 = np.amax(np.abs(atrain1fft))
    maxsignal2 = np.amax(np.abs(atrain2fft))

    mp2rage = ((np.conj(atrain1fft) * atrain2fft).real) / (np.abs(atrain1fft) * np.abs(atrain1fft) + np.abs(atrain2fft)*np.abs(atrain2fft))
    mp2ragethresh = mp2rage * np.logical_or((np.abs(atrain1fft) > 0.05*maxsignal1), (np.abs(atrain2fft) > 0.05*maxsignal2))


#   sum of squares for n coils
#   combine mp2rage images with weighting function atrain2 (greti2)
    mp2ragesos = np.zeros([readnum,phasenum,slicenum],dtype=np.float64)
    atrain2sos = np.zeros([readnum,phasenum,slicenum],dtype=np.float64)
    for ii in range(readnum):
        for jj in range(phasenum):
            for kk in range(slicenum):
                           mp2ragesos[ii,jj,kk] = np.sum(np.squeeze(mp2ragethresh[ii,jj,kk,:])*(np.abs(np.squeeze(atrain2fft[ii,jj,kk,:])/np.abs(np.sum(atrain2fft[ii,jj,kk,:])))))
                           atrain2sos[ii,jj,kk] = np.sqrt(np.sum(np.squeeze(atrain2fft[ii,jj,kk,:]) * (np.conj(np.squeeze(atrain2fft[ii,jj,kk,:])))))
    #        end

    #    end
    #end


    atrain2sum = (np.abs(atrain2fft[:,:,:,0]))

    #atrain2sum = atrain2sos;

    #threshold mask    
 #   atrain2sumfilt = medfilt1(atrain2sum, 3, mcat([]), 1)
 #   atrain2sumfilt = medfilt1(atrain2sumfilt, 3, mcat([]), 2)
 #   atrain2sumfilt = medfilt1(atrain2sumfilt, 3, mcat([]), 3)
 #   maxsum2 = max(max(max(atrain2sum)))
    maxsum2 = np.amax(atrain2sos)
    mp2ragemask = abs(atrain2sos) > 0.000001 * maxsum2

    mp2ragewa = (mp2ragesos* abs(atrain2fft[:,:,:,0]))/atrain2sum

    mp2ragewamask = (mp2ragesos + 0.5) * mp2ragemask - 0.5
    
    #t1map
    t1map = np.zeros([readnum,phasenum,slicenum],dtype=np.float)
    alpha = 2.0 * 3.141592654 * 10.0 / 360.0# flipangle (set both to be the same at the moment) (converted to radians)
    alpha2 = 2.0 * 3.141592654 * 10.0 / 360.0
    mp2ragetr = 8.0
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
    for ii in range(readnum):
    #         ii
         for jj in range(phasenum):
             for kk in range(slicenum):
    #      %           mp2ragewa(ii,jj,kk) = (mp2rage(ii,1,jj,kk)*atrain2fft(ii,1,jj,kk) + mp2rage(ii,2,jj,kk)*atrain2fft(ii,2,jj,kk))/atrain2sos(ii,jj,kk);
    #              t1map[ii,jj,kk] = -np.interp(mp2ragewamask[ii,jj,kk],-mp2ragecalc,-t1array)
    #              t1func = interp1d(t1array,mp2ragecalc)
                  mp2ragevalue = mp2ragewamask[ii,jj,kk]
                  t1map[ii,jj,kk] = t1func(mp2ragevalue)
    #             end
    #         end
    #     end

    #     t1map(:,:,:,expnum) = interp1(mp2ragecalc,t1array,mp2ragewa);
#    t1map = np.interp(mp2ragecalc, t1array, mp2ragewa)
#    t1mapwmask = t1map *elmul* mp2ragemask
#
#    imagesc(squeeze(t1mapwmask(80, mslice[:], mslice[:])))

#    fid = fopen(strcat(directoryname, mstring('/t1map')), mstring('wb'))
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