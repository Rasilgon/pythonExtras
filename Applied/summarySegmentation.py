# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:11:10 2015

@author: trashtos
"""

import glob
import pandas as pd
import numpy as np
import os
from ownUtilities import grep
from gdalWrappers import vectorClip

def normalize(array):
    subarray = array[np.isfinite(array)]
    return((array - np.min(subarray[np.nonzero(subarray)])) / (np.max(subarray[np.nonzero(subarray)])- np.min(subarray[np.nonzero(subarray)])))

def normalizeClean(array):
    return((array - np.min(array)) / (np.max(array)- np.min(array)))

def getNames(fileName):
    try:
        bands = [x for x in os.path.basename(fileName).split("_")[6]]
        bandNames= ["MaxHALL","MaxHAll_strStd1", "MaxHAll_strStd2", "MaxHAll_strStd3",
                 "NDVI", "NDVI_Kuwahara5", "NDVI_Laplacian5","NDVI_Mean5", 
                 "EVI", "EVI_Kuwahara5", "EVI_Laplacian5","EVI_Mean5", 
                 "NIR", "NIR_Mean5"  ,          
                "RED" ,"RED_Mean5",  
                "GREEN", "GREEN_Mean5",  
                 "GRVI", "GRVI_Mean5","GRVI_Laplacian5" ]
        ids = bands[1:]
        values = ["0","0","0"]
        for i in range(len(ids)):
            # string
            if len(ids)==1:
                 values[0] = ids[0]
                 
            elif len(ids)==2:
                if ids[0] == "1" or ids[0] == "2":
                    values[0] = ids[0] + ids[1]
                else:
                    values[0] = ids[0]
                    values[1] = ids[1]
                    
            elif len(ids)==3:
                if ids[0] == "1" or ids[0] == "2":
                    values[0] = ids[0] + ids[1]
                    values[1] = ids[2]
                elif ids[1] == "1" or ids[1] == "2":
                    values[0] = ids[0]
                    values[1] = ids[1] + ids[2]
                else:
                    values[0] = ids[0] 
                    values[1] = ids[1] 
                    values[2] = ids[2]   
                    
            elif len(ids)==4:
                if ids[0] == "1" or ids[0] == "2":
                    values[0] = ids[0] + ids[1]
                    if ids[2] == "1" or ids[2] == "2":
                        values[1] = ids[2]+ ids[3]
                    else:
                        values[1] = ids[2]
                        values[2] = ids[3]
                        
                elif ids[1] == "1" or ids[1] == "2":
                    values[0] = ids[0]
                    values[1] = ids[1] + ids[2]
                    values[2] = ids[3]
                    
                elif ids[2] == "1" or ids[2] == "2":
                    values[0] = ids[0]
                    values[1] = ids[1] 
                    values[2] = ids[2] + ids[3] 

            elif len(ids)==5:
                if ids[0] == "1" or ids[0] == "2":
                    values[0] = ids[0] + ids[1]
                    if ids[2] == "1" or ids[2] == "2":
                        values[1] = ids[2]+ ids[3]
                        values[2] = ids[4]  
                    else:
                        values[1] = ids[2]
                        values[2] = ids[3] + ids[4]
                        
                elif ids[1] == "1" or ids[1] == "2":
                    values[0] = ids[0]
                    values[1] = ids[1] + ids[2]
                    values[2] = ids[3] + ids[4] 
                    

                 
            elif len(ids) == 6:
                values[0] = ids[0] + ids[1]
                values[1] = ids[2] + ids[3]
                values[2] = ids[4] + ids[5]
 
        val =[int(bands[0])-1] + [int(x) -1 for x in values if x != "0"]
        names = [bandNames[i] for i in val]
        return ((" + ").join(names))

    except:
        return ("No names")                    
            
               
                
 
            

files = glob.glob("/home/trashtos/CleaningTiles/SegmentationFinal/tile_row10col20/HCIR/*.txt")

files = grep('/home/trashtos/CleaningTiles/SegmentationFinal/tile_row10col20/HCIR/tile_row10col20_stack_cubic_80_80_4_clumpsexport', files, mask=False)

files = grep('/home/trashtos/CleaningTiles/SegmentationFinal/tile_row10col20/HCIR/tile_row10col20_stack_cubic_80_80_3_clumpsexport', files, mask=False)
files = grep('/home/trashtos/CleaningTiles/SegmentationFinal/tile_row10col20/HCIR/tile_row10col20_stack_cubic_80_80_1', files, mask=False)
files = grep('/home/trashtos/CleaningTiles/SegmentationFinal/tile_row10col20/HCIR/tile_row10col20_stack_cubic_80_80_2', files, mask=False)

#files.append(outVector[:-4] + "_pd.txt")

#files1 = grepInv('/home/trashtos/CleaningTiles/SegmentationFinal/tile_row29col8/tile_row29col8_stack_cubic_80_80_1', files)
#files1 = grepInv('/home/trashtos/CleaningTiles/SegmentationFinal/tile_row29col8/tile_row29col8_stack_cubic_80_80_2', files1)


#files1 = grep('/home/trashtos/CleaningTiles/SegmentationFinal/tile_row29col8/tile_row29col8_stack_cubic_80_80_4', files)

intraVar = np.zeros(len(files))
interVar = np.zeros(len(files))
normVar = np.zeros(len(files))
numObj = np.zeros(len(files))

for i in range(len(files)):
    #print(i)
    df = pd.read_csv(files[i], sep=',', prefix ="NO")
    try:
        intraVar[i] = df.iloc[0,0]
    except:
        print("Could not")
    try:
        interVar[i] = df.iloc[1,0]
    except:
        print("Could not")
    try:
        normVar[i] = df.iloc[2,0]
    except:
        print("Could not")
    try:
        numObj[i] = df.iloc[3,0]
    except:
        print("Could not")
    

summaryRaw = pd.DataFrame({
            'file':files,
            'combination':[getNames(fila) for fila in files],
            'intraVar':intraVar,
            'interVar':interVar,
            'normVar':normVar, 
            'numObj':numObj, 

            })
 
# REmove duplicates
summaryRaw = summaryRaw.drop_duplicates(['combination'], take_last=True)
           
summaryRaw['NintraVar'] = normalize(np.asarray(summaryRaw['intraVar']))
summaryRaw['NinterVar'] = normalize(np.asarray(summaryRaw['interVar']))
summaryRaw['NnumObj'] = normalize(np.asarray(summaryRaw['numObj']))


summary = summaryRaw.loc[(summaryRaw.numObj > 0),].copy()    
summary['Range']=   np.abs(summary['interVar']-summary['intraVar'])
summary['NRange'] = normalizeClean(np.asarray(summary['Range']))
summary['FYeah2'] = (summary['NRange'] + summary['NnumObj'])/2.

summary['Score'] = (summary["NintraVar"] + summary["NinterVar"]+ summary["NnumObj"])/3.
summary['NnumObj'] = np.round(summary['NnumObj'], decimals=3)

summary['NinterVar'] = np.round(summary['NinterVar'], decimals=3)
summary['NintraVar'] = np.round(summary['NintraVar'], decimals=3)

summary['ScoreRange'] = np.abs(summary['NinterVar']-summary['NintraVar'])

summary['FYeah'] = (summary['ScoreRange'] + summary['NnumObj'])/2.


summary['FYeahWHo'] = (summary['ScoreRange'] + summary['NnumObj'])/2.

dof = summary.sort(['FYeah2'], ascending=[True])

dof.to_csv("/home/trashtos/CleaningTiles/summary_tile_row10col20_HCIR_nonzero_34.csv", index=False)





####
inImage=  "/home/trashtos/CleaningTiles/SegmentationFinal/tile_row29col8_subarea_segIdeal.tif"
refImage =  "/media/trashtos/Meerkat/cleanTiles/Rows29/Cols8/tile_row29col8_stack_cubic_HCIR.tif"
outImage = "/home/trashtos/CleaningTiles/SegmentationFinal/tile_row29col8_subarea_segIdeal_R.tif"

clipReproject(inImage, refImage, outImage)

import subprocess
cmd = "gdal_polygonize.py " + outImage + """ -f "ESRI Shapefile" """ + os.path.splitext(outImage)[0] + ".shp"
subprocess.call(cmd, shell = True)

inVector = os.path.splitext(outImage)[0] + ".shp"
coverVector = "/media/trashtos/Meerkat/0000/0000/0000_0000.shp"  
outVector = "/home/trashtos/CleaningTiles/SegmentationFinal/tile_row29col8_subarea_segIdeal_OK.shp"
vectorClip(inVector, coverVector, outVector)

# values of this one
values = segQuality(outVector, refImage)
df = pd.DataFrame(values)
df.to_csv(outVector[:-4] + "_pd.txt", header = False, index = False)




###
import numpy as np
#df = pd.read_csv("/home/trashtos/CleaningTiles/summary_tile_row29col8_nonzero_SORTED.csv")
df = pd.read_csv("/home/trashtos/CleaningTiles/summary_tile_row29col8_nonzero_2out_sorted.csv")
df['NnumObj'] = np.round_(df['NnumObj'], decimals=3)
df['NinterVar'] = np.round_(df['NinterVar'], decimals=3)
df['NintraVar'] = np.round_(df['NintraVar'], decimals=3)

df['ScoreRange'] = np.abs(df['NinterVar']-df['NintraVar'])*df['NnumObj']**2
df['NScoreRange'] = normalizeClean(np.asarray(df['ScoreRange']))
df['ScoreRangeRaw'] = np.abs(df['interVar']-df['intraVar'])*df['NnumObj']**2
df['NScoreRangeRaw'] = normalizeClean(np.asarray(df['ScoreRangeRaw']))
#dof = df.sort(['NnumObj', 'NinterVar', 'NintraVar'], ascending=[True, True, True])

dof = df.sort(['NScoreRange'], ascending=[True])

dof.to_csv("/home/trashtos/CleaningTiles/summary_tile_row27col12_nonzero_RangeSorted.csv")


combinations = dof['combination']


subset = combinations[:int(len(combinations)*0.25)]

values = [comb.split(" + ") for comb in subset]
import itertools
bands = list(itertools.chain.from_iterable(values))

from collections import Counter

res = Counter(bands)

bandsDf = pd.DataFrame.from_dict(res, orient='index').sort(axis= 0)
a = [(g[0], len(list(g[1]))) for g in itertools.groupby(bands)]
