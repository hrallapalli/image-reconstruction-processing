#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:55:15 2017

@author: doddst
"""

# read paravision parameter
# from method file when parameter name and value are on the same line
# or when single value is on next line assumes a bracket is last character
# on previous line

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
