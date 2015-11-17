# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 08:08:01 2015
u
@author: trashtos
"""

import shutil
import errno
from shutil import ignore_patterns
import os

def copySelective(src, dest, pattern):
    
    try:
        shutil.copytree(src, dest, ignore = ignore_patterns (pattern))
    except OSError as e:
        # If the error was caused becasue the source was not a directory
        if e.errno == errno.ENOTDIR:
            shutil.copytree(src, dest)
        else:
            print("Directory not copied. Error: %s" %e)
            
def copyFile(src, dest):
    
    try:
        shutil.copy(src, dest)
    except shutil.Error as e:
        print("File not copied. Error: %s" %e)
    except IOError as e:
        print("Sys: File not copied. Error: %s" %e.strerror)
        #

                     
            
            
src = "/media/trashtos/Meerkat/Ramiro_Masterarbeit/Tiles/"

dest = "/media/trashtos/Meerkat/cleanTiles/"

pattern = ["*.kea", "mp*.spd", "*core.spd", "*10m.spd"]
 



	#Read directories/files

#os.chdir("/media/trashtos/My Passport/")
inputDIR = os.path.abspath("/media/trashtos/Meerkat/Ramiro_Masterarbeit/Tiles")
outDIR = os.path.abspath("/media/trashtos/Meerkat/cleanTiles")
baseName = "_2014_DHDNGK4_1m_rmn_pmfmccgrd_h.spd"
    # index
rowIndex = range(1,31, 1)
rowIndex = range(1,31, 1)
         
         #11 - 15 (this at the end)
         #  move to 12
colIndex = range(1,33, 1)
	#Set names for files
for i in range(1,len(rowIndex)):
    for j in range(1, len(colIndex)):
        inDirectory = os.path.join(inputDIR, ("Rows" +str(rowIndex[i])),  ("Cols"+str(colIndex[j])) )
        outDirectory = os.path.join(outDIR, ("Rows" +str(rowIndex[i])),  ("Cols"+str(colIndex[j])) )
        filename = "tile_row"+ str(rowIndex[i]) + "col" + str(colIndex[j]) + baseName 
        fullName = os.path.join(inDirectory, filename) 

        
        if(os.path.exists(fullName)): 
            print(fullName)
            copyFile(fullName , outDirectory)
        
        a = "/media/trashtos/Meerkat/Ramiro_Masterarbeit/Tiles/Rows30/Cols10/tile_row30col10_2014_DHDNGK4_1m_rmn_pmfmccgrd_h.spd"
          
            
# create outboundary
            # gdaltindex ./tile.shp ./cleanTiles/Rows1/Cols24/tile_row1col24_2014_DHDNGK4_rmn_pmfmccgrd_1m_CanopyCover.kea 

import subprocess
rowIndex = range(1,31, 1)

colIndex = range(1,33, 1)
baseName = "_2014_DHDNGK4_rmn_pmfmccgrd_1m_CanopyCover.kea"
	#Set names for files
for i in range(len(rowIndex)):
    for j in range(len(colIndex)):
        outDirectory = os.path.join(outDIR, ("Rows" +str(i)),  ("Cols"+str(j)) )
        #fullName = "./Rows" +str(i)+"./Cols"+str(j)+ "/tile_row"+ str(i) + "col" + str(j) + "baseName  
        filename = "tile_row"+ str(i) + "col" + str(j) + baseName 
        fullName = os.path.join(outDirectory, filename) 
                

        
        if(os.path.exists(fullName)): 
            #print(fullName)
            subprocess.call(("gdaltindex  -t_srs EPSG:31468 " + ("./Rows" +str(i)+"/Cols"+str(j)+ "/tile_row"+ str(i) + "col" + str(j)  + ".shp")+ " " + fullName) , shell =True )
            
            
# clipping with gdal (github clip cheat shetet ) but before find out what are my tiles
            
# make tuples      
tiles = [(7,18), (7,19), (8,21),(8,29), (10,14), (10,20), (10,24),(11,19),
        (12,15), (12,22), (12,27),(12,28), (14,18),(15,15),(18,15), (21,13),
        (22,8), (23,11), (23,6),(24,9),(24,7), (27,9),(27,12), (29,8) ] 

from rsgislib import imageutils
import os

outDIR = os.path.abspath("/media/trashtos/Meerkat/cleanTiles")


clippingSHP = []
for tile in tiles:
     outDirectory = os.path.join(outDIR, ("Rows" +str(tile[0])),  ("Cols"+str(tile[1])) )
     filename = "tile_row"+ str(tile[0]) + "col" + str(tile[1]) + ".shp"
     tileName = os.path.join(outDirectory, filename)      
     if(os.path.exists(tileName)):
         clippingSHP.append(tileName)
     else:
         print (tileName)
         
         
# the cir
CIR2012 = "/media/trashtos/Meerkat/Ramiro_Masterarbeit/CIRMosaik20120819und201208201.kea"

import rsgislib
from rsgislib import imageutils
inputImage = './Rasters/injune_p142_casi_sub_utm.kea'
inputVector = './Vectors/injune_p142_plot_location_utm.shp'
outputImage = './TestOutputs/injune_p142_casi_sub_utm_subset.kea'
gdalformat = 'KEA'
datatype = rsgislib.TYPE_32FLOAT
imageutils.subset(inputImage, inputVector, outputImage, gdalformat, datatype)
    

