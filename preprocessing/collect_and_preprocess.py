#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import h5py
import fjammod as fjam
import numpy as np


# read event path from command line if given: 
if len(sys.argv) > 1:
  basepath = sys.argv[1]
else:
  basepath = os.getcwd()  

print("Reading events from path", basepath)


def create_mat_pack(path, band=0, packet_amount=10):
  pkgcounter = 0
  pkgsize = 349184
  data_arr = np.zeros((2, (pkgsize * packet_amount)))
  
  while(pkgcounter < packet_amount):  
    offset = pkgcounter * 1024*1024
    Carray,CenterF,SamplerateMHz=fjam.read1pack(path, offset=offset)
    start = pkgcounter*pkgsize
    data_arr[0, start:start+pkgsize] = Carray[0::2,band]
    data_arr[1, start:start+pkgsize] = Carray[1::2,band]
    print("last packet:", pkgcounter)    
    pkgcounter = pkgcounter + 1
    print(data_arr) 
   
  return data_arr
  
#open output file
f = h5py.File('events.hdf5', 'a')

#read group names from file if any
grnames = list(f.keys())


for d in os.listdir(basepath):
  fpath = basepath + d + "/"
  evdirs = os.listdir(fpath)
  for e in evdirs:
    eventpath = fpath + e + "/"
    groupname = d+"_"+e #default groupname
    print(eventpath, groupname)

    if groupname in grnames:
      grp_X = f[groupname]
      print("using existing group "+groupname)
    else:
      grp_X = f.create_group(groupname)
      print("group "+groupname+" created")

    evfiles = os.listdir(eventpath)
    evfiles = filter(lambda item: "DAT" in item, evfiles)

    eventcounter = 1

    for evf in evfiles:
      dsname = evf[:8]    
      print("event", eventcounter, evf[:8])
      print("path", eventpath + evf)
      mat = create_mat_pack(eventpath+evf, 0, 20)
      print("final", mat)
      grp_X.create_dataset(dsname, np.shape(mat), data=mat)
      eventcounter = eventcounter + 1
      mat = []


print(list(f.keys()))
f.close()
    

