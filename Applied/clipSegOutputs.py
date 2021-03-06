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

listVectors = sorted(glob.glob( "/media/trashtos/Meerkat/cleanTiles/Rows10/Cols20/*.shp"), key=os.path.getctime)

#subList = grep('/media/trashtos/Meerkat/cleanTiles/Rows10/Cols20/tile_row10col20_stack_cubic_80_80_3', listVectors)


#outDir = "/home/trashtos/CleaningTiles/SegmentationFinal/tile_row10col20"

def wrapper(inVector):
    outDir = "/home/trashtos/CleaningTiles/SegmentationFinal/tile_row10col20"
    coverVector = "/media/trashtos/Meerkat/0000/0000/0000_0000.shp"  
    outVector = os.path.join(outDir, os.path.basename(inVector))
    
    if os.path.exists(outVector):
        pass
    else:
        try:
            vectorClip(inVector, coverVector, outVector)
        except:
            print ("error in file", inVector)
        
        return (outVector)
        
            
            
 
    
        
        
p = Pool(3)   


newVectors = p.map(wrapper, listVectors[-108:] )



  
#for i in range(len(subList)):
#    outVector = os.path.join(outDir, os.path.basename(subList[i]))
#    vectorClip(subList[i], coverVector, outVector)
