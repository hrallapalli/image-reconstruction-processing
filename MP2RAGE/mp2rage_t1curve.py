#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:18:45 2017

@author: doddst
"""

# mp2rage reco python; start with m2py converter
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



         
 #   alphadeg = readpvpar('##$MP2RAGE_1stExcPulseAngle=', directoryname);
alphadeg = 10.0 
                   
 #   alpha2deg = readpvpar('##$MP2RAGE_2ndExcPulseAngle=', directoryname);
alpha2deg = 10.0

                    




phasenum = 180
nseg = 3
segduration = 0.85
ti1 = 1.3
ti2 = 3.5
    #t1map

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


plt.plot(t1array,mp2ragecalc)