# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:49:50 2015

@author: trashtos
"""

#### Modules ###########
import sys
import fiona
from shapely.geometry import shape
sys.path.append("/home/trashtos/GitHub/OBIA")
from rasterstats import zonal_stats
import pandas as pd
import numpy as np
import glob
from multiprocessing import Pool
from ownUtilities import grep
import os

### utilities ##########
def mymean(x):
    return (np.mean(x))

def varianza(x):
    return (np.var(x)) #numpy.nanvar(x
   
#
def segQuality(inVector, inImage):
    # open the vector
    lyr = fiona.open(inVector)
    features = [x for x in lyr]
    values = np.zeros([len(features), 5], dtype=float)
    # loop over features
    for  i in range(len(features)):
        geometry1 = shape(features[i]['geometry'])
        
        restFeatures = features[:i] + features[(i+ 1):]
                
        try:
            value = zonal_stats(geometry1, inImage, stats=['count'], add_stats={'mymean':mymean, "myvarianza":varianza } )
        except:
            value = [{'count': np.nan, 'mymean': np.nan, 'myvarianza':np.nan}]    

        df =  pd.DataFrame.from_dict(value, orient='columns', dtype=None)
        
            
        for j in range(len(restFeatures)):        
            geometry2 = shape(features[j]['geometry'])   
            if geometry2.intersects(geometry1) == True:
                #print("They touch")
                try:
                    value = zonal_stats(geometry2, inImage, stats=['count'], add_stats={'mymean':mymean, "myvarianza":varianza } )
                except:
                    value = [{'count': np.nan, 'mymean': np.nan, 'myvarianza':np.nan}]

                df = df.append(pd.DataFrame.from_dict(value, orient='columns', dtype=None))
            
        values[i,0] = df.iloc[0,0] # count
        values[i,1] = df.iloc[0,1] # mean
        values[i,2] = df.iloc[0,2] # myvarianza
        values[i,3] = np.nanvar(df.iloc[:,1]) # varianza between
        values[i,4] = len(df.iloc[1:]) # neighbours
    # get overal values   
    intraVarWeighted = np.nansum( values[:,0]*values[:,2] ) / np.nansum(values[:,0])
    interVarWeighted = np.nansum( values[:,4]*values[:,3] ) / np.nansum(values[:,4])
    normVariance = (intraVarWeighted - interVarWeighted) / (intraVarWeighted + interVarWeighted)
    numberSegments =  len(values[:,4])
    #
    result = [inVector, intraVarWeighted, interVarWeighted, normVariance, numberSegments] 
    
          

    return( result)

############
def wrapper(parList):
    inVector = str(parList[0])
    inImage = str(parList[1])
    try:
        values = segQuality(inVector, inImage)
    except:
        values = [inVector, "There was an error with the file"]

    df = pd.DataFrame(values)
    df.to_csv(inVector[:-4] + "_pd.txt", header = False, index = False)
#/home/fr/fr_fr/fr_rs215/Masterarbeit/SegmentationFinal/tile_row29col8Sumamry.txt", header = True, index = False)




#    outFile =os.path.split(inVector)[0] + ".txt"
#    with open(outFile, "w") as f:
#        for val in values:
#            f.write(str(val) + ",")




#listVectors = glob.glob( "/home/trashtos/CleaningTiles/SegmentationFinal/tile_row29col8/*.shp")
listVectors = sorted(glob.glob("/home/trashtos/CleaningTiles/SegmentationFinal/tile_row27col12/*.shp"), key=os.path.getctime)
#subList= np.asarray(grep('/home/trashtos/CleaningTiles/SegmentationFinal/tile_row29col8/tile_rowcol8_stack_cubic_80_80_2', listVectors, mask = False))

inImage=  "/home/trashtos/CleaningTiles/SegmentationFinal/tile_row27col12_stack_cubic_HCIR.tif"
values = np.zeros([len(listVectors[-108:]), 4], dtype=float)


listImage = np.asarray([inImage for i in range(len(listVectors[-108:]))])
#outFile = "/home/trashtos/CleaningTiles/SegmentationFinal/tile_row27col8.txt"

listPar = [ list(x) for x in zip(listVectors[-108:], listImage)]

#subset = listPar[:14]

cl = 1*3
p = Pool(cl)

p.map(wrapper, listPar)



listVectors = sorted(glob.glob("/home/trashtos/CleaningTiles/SegmentationFinal/tile_row29col8/*.shp"), key=os.path.getctime)
#subList= np.asarray(grep('/home/trashtos/CleaningTiles/SegmentationFinal/tile_row29col8/tile_rowcol8_stack_cubic_80_80_2', listVectors, mask = False))

inImage=  "/home/trashtos/CleaningTiles/SegmentationFinal/tile_row29col8_stack_cubic_HCIR.tif"
values = np.zeros([len(listVectors[-108:]), 4], dtype=float)


listImage = np.asarray([inImage for i in range(len(listVectors[-108:]))])
#outFile = "/home/trashtos/CleaningTiles/SegmentationFinal/tile_row27col8.txt"

listPar = [ list(x) for x in zip(listVectors[-108:], listImage)]

#subset = listPar[:14]

cl = 1*3
p = Pool(cl)

p.map(wrapper, listPar)

#df = pd.DataFrame(result, columns = ["file", "intraVarWeighted", "interVarWeighted", "normVariance", "numberSegments"])


#df.to_csv("/home/fr/fr_fr/fr_rs215/Masterarbeit/SegmentationFinal/tile_row29col8Sumamry.txt", header = True, index = False)





