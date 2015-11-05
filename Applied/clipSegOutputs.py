# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 08:52:01 2015

@author: trashtos
"""
import sys
sys.path.append("/home/trashtos/GitHub/OBIA")
import os
from gdalWrappers import vectorClip
# clip them all
import re
import glob
from multiprocessing import Pool

def grep(pattern,word_list):
    expr = re.compile(pattern)
    return ([elem for elem in word_list if expr.match(elem)])
    
    
coverVector = "/media/trashtos/Meerkat/0000/0000/0000_0000.shp"  

listVectors = glob.glob( "/media/trashtos/Meerkat/cleanTiles/Rows29/Cols8/*.shp")

subList = grep('/media/trashtos/Meerkat/cleanTiles/Rows29/Cols8/tile_row29col8_stack_cubic_80_80_4', listVectors)


outDir = "/home/trashtos/CleaningTiles/SegmentationFinal/tile_row29col8"

def wrapper(inVector):
    outDir = "/home/trashtos/CleaningTiles/SegmentationFinal/tile_row29col8"
    coverVector = "/media/trashtos/Meerkat/0000/0000/0000_0000.shp"  
    outVector = os.path.join(outDir, os.path.basename(inVector))
    try:
        vectorClip(inVector, coverVector, outVector)
    except:
        print('Error with ', inVector)
        
        
p = Pool(3)   

p.map(wrapper, subList )



  
#for i in range(len(subList)):
#    outVector = os.path.join(outDir, os.path.basename(subList[i]))
#    vectorClip(subList[i], coverVector, outVector)