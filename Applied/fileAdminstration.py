# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:31:40 2015

@author: trashtos
"""

import sys
sys.path.append("/home/trashtos/GitHub/OBIA")
import os
import glob

from ownUtilities import grep

#################

listVectors = glob.glob( "/home/trashtos/CleaningTiles/SegmentationFinal/tile_row10col20/*.txt")



subList = grep('/home/trashtos/CleaningTiles/SegmentationFinal/tile_row29col8/tile_row29col8_stack_cubic_80_80_1', listVectors, True)

#for file in subList:
 #   os.remove(file)
import shutil
import errno

from shutil import ignore_patterns
######## 
# move files
outDir= "/home/trashtos/CleaningTiles/SegmentationFinal/tile_row10col20/HCIR"
for file in listVectors:
    oufile = os.path.join( outDir, os.path.basename(file) )
    try:
        shutil.move(file, oufile )
    except OSError as e:
        print('File not copied. Error: %s' % e)
        
HCIR = [base + "_stack_cubic_CIR.tif"  for base in cleanTiles()]

outDir = "/home/trashtos/CleaningTiles/SegmentationFinal"
for file in HCIR:
    oufile = os.path.join( outDir, os.path.basename(file) )
    try:
        shutil.copy(file, oufile )
    except OSError as e:
        print('File not copied. Error: %s' % e)

        
# clip 29_8 stack
# subset the badns = 2, 
from  rsgislibWrappers import selectImageBands
import rsgislib
from rsgislib import imageutils 
from ownUtilities import cleanTiles

stackTilesCubic = [base + "_stack_cubic.kea" for base in cleanTiles()][:-1] 

segStacks = [stackTilesCubic[-1], stackTilesCubic[-2], stackTilesCubic[16], 
                        stackTilesCubic[5], stackTilesCubic[7], stackTilesCubic[12] ]



bandList = [2,13,15,17]

for stack in stackTilesCubic:
    inImage = stack
    outImage =  stack[:-4] +"_HCIR.kea"
    gdalFormat = 'KEA'
    dataType = rsgislib.TYPE_32FLOAT    
    imageutils.selectImageBands(inImage, outImage, gdalFormat, dataType, bandList)
    
import subprocess
for stack in stackTilesCubic:
    cmd = "gdal_translate -ot Float32 -of GTiff " + stack[:-4] +"_HCIR.kea"   + " "  + stack[:-4] +"_HCIR.tif"  
    subprocess.call(cmd, shell = True)
    