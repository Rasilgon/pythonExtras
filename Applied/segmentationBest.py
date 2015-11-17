# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 21:48:39 2015

@author: trashtos
"""

import pandas as pd
import numpy as np
import os


# Read the two files
row27col12 = pd.read_csv("/home/trashtos/CleaningTiles/summary_tile_row27col12_HCIR_nonzero_34.csv")
#row29col8 = pd.read_csv("/home/trashtos/CleaningTiles/summary_tile_row29col8_HCIR_nonzero_234.csv")
row29col8 = pd.read_csv("/home/trashtos/CleaningTiles/summary_tile_row29col8_HCIR_nonzero_34.csv")
row10col20 = pd.read_csv("/home/trashtos/CleaningTiles/summary_tile_row10col20_HCIR_nonzero_34.csv")

#comb27_2 = set()


# Subset firrst  785  of eacg and merge them to new table

sub27 = row27col12.iloc[:785,:].copy()
sub29 = row29col8.iloc[:785,:].copy()
sub10 = row10col20.iloc[:785,:].copy()


dfs = [sub27, sub29, sub10]
for df in dfs:    
    df['NintraVar'] = normalize(np.asarray(df['intraVar']))
    df['NinterVar'] = normalize(np.asarray(df['interVar']))
    df['NnumObj'] = normalize(np.asarray(df['numObj']))   
    df = df.loc[(df.numObj > 0),].copy()    
    df['Range']=   np.abs(df['interVar']-df['intraVar'])
    df['NRange'] = normalizeClean(np.asarray(df['Range']))
    df['FYeah2'] = (df['NRange'] + df['NnumObj'])/2.    
    df['Score'] = (df["NintraVar"] + df["NinterVar"]+ df["NnumObj"])/3.
    df['NnumObj'] = np.round(df['NnumObj'], decimals=3)    
    df['NinterVar'] = np.round(df['NinterVar'], decimals=3)
    df['NintraVar'] = np.round(df['NintraVar'], decimals=3)    
    df['ScoreRange'] = np.abs(df['NinterVar']-df['NintraVar'])    
    df['FYeah'] = (df['ScoreRange'] + df['NnumObj'])/2.   
    df['FYeahWHo'] = (df['ScoreRange'] + df['NnumObj'])/2.
    df = df.sort(['FYeah2'], ascending=[True])
    
    
allDf= pd.merge(dfs[0], dfs[1], how='outer', on=['combination'], left_on=None, right_on=None,
      left_index=False, right_index=False, sort=True,
      suffixes=('_27', '_29'), copy=True)
      
allDf= pd.merge(allDf, dfs[2], how='outer', on=['combination'], left_on=None, right_on=None,
      left_index=False, right_index=False, sort=True,
      suffixes=('', '_10'), copy=True)   
      
allDf = allDf.dropna(axis=0, how='any', thresh=None, subset=['FYeah2_27', 'FYeah2_29', 'FYeah2'], inplace=False)      
     
#allDf['FinalScore'] = ( allDf.fillna(0)['FYeah2_27'] + allDf.fillna(0)['FYeah2_29'] + allDf.fillna(0)['FYeah2'])/3.
allDf['FinalScore'] = ( allDf['FYeah2_27'] + allDf['FYeah2_29'] + allDf['FYeah2'])/3.
         
allDf['NFinalScore'] = normalize(np.asarray(allDf['FinalScore']))
      
allDf = allDf.sort(['NFinalScore'], ascending=[True])  


allDf.to_csv("/home/trashtos/CleaningTiles/summary_nonzero_34_noNA.csv", index=False)
      
###########
     
# Read the two files
 #    /home/trashtos/CleaningTiles/summary_tile_row27col12_size_nonzero.csv
row27col12 = pd.read_csv("/home/trashtos/CleaningTiles/summary_tile_row27col12_size_nonzero.csv")
#row29col8 = pd.read_csv("/home/trashtos/CleaningTiles/summary_tile_row29col8_HCIR_nonzero_234.csv")
row29col8 = pd.read_csv("/home/trashtos/CleaningTiles/summary_tile_row29col8_size_nonzero.csv")
row10col20 = pd.read_csv("/home/trashtos/CleaningTiles/summary_tile_row10col20_size_nonzero.csv")


dfs = [row27col12, row29col8, row10col20]
for df in dfs:    
    df['NintraVar'] = normalize(np.asarray(df['intraVar']))
    df['NinterVar'] = normalize(np.asarray(df['interVar']))
    df['NnumObj'] = normalize(np.asarray(df['numObj']))   
    df = df.loc[(df.numObj > 0),].copy()    
    df['Range']=   np.abs(df['interVar']-df['intraVar'])
    df['NRange'] = normalizeClean(np.asarray(df['Range']))
    df['FYeah2'] = (df['NRange'] + df['NnumObj'])/2.    
    df['Score'] = (df["NintraVar"] + df["NinterVar"]+ df["NnumObj"])/3.
    df['NnumObj'] = np.round(df['NnumObj'], decimals=3)    
    df['NinterVar'] = np.round(df['NinterVar'], decimals=3)
    df['NintraVar'] = np.round(df['NintraVar'], decimals=3)    
    df['ScoreRange'] = np.abs(df['NinterVar']-df['NintraVar'])    
    df['FYeah'] = (df['ScoreRange'] + df['NnumObj'])/2.   
    df['FYeahWHo'] = (df['ScoreRange'] + df['NnumObj'])/2.
    df = df.sort(['FYeah2'], ascending=[True])
    
    
allDf= pd.merge(dfs[0], dfs[1], how='outer', on=['combination'], left_on=None, right_on=None,
      left_index=False, right_index=False, sort=True,
      suffixes=('_27', '_29'), copy=True)
      
allDf= pd.merge(allDf, dfs[2], how='outer', on=['combination'], left_on=None, right_on=None,
      left_index=False, right_index=False, sort=True,
      suffixes=('', '_10'), copy=True)   
      
allDf = allDf.dropna(axis=0, how='any', thresh=None, subset=['FYeah2_27', 'FYeah2_29', 'FYeah2'], inplace=False)      
     
#allDf['FinalScore'] = ( allDf.fillna(0)['FYeah2_27'] + allDf.fillna(0)['FYeah2_29'] + allDf.fillna(0)['FYeah2'])/3.
allDf['FinalScore'] = ( allDf['FYeah2_27'] + allDf['FYeah2_29'] + allDf['FYeah2'])/3.
         
allDf['NFinalScore'] = normalize(np.asarray(allDf['FinalScore']))

allDf['FinalRange'] = ( allDf['NRange_27'] + allDf['NRange_29'] + allDf['NRange'])/3.
         
allDf['NFinalRange'] = normalize(np.asarray(allDf['FinalRange']))
      
allDf = allDf.sort(['NFinalRange'], ascending=[True])  


allDf.to_csv("/home/trashtos/CleaningTiles/summary_nonzero_size_noNA.csv", index=False)
      



##########













combisUno = list(row27col12['combination'][:500])+ list(row29col8['combination'][:500])

len(combisUno)

unique = list(set(combisUno))

len(unique)

# Convert unique into band combinations
bandNames= {"MaxHALL":1,"MaxHAll_strStd1":2, "MaxHAll_strStd2":3, "MaxHAll_strStd3":4,
                 "NDVI":5, "NDVI_Kuwahara5":6, "NDVI_Laplacian5":7,"NDVI_Mean5":8, 
                 "EVI":9, "EVI_Kuwahara5":10, "EVI_Laplacian5":11,"EVI_Mean5":12, 
                 "NIR":13, "NIR_Mean5":14,          
                "RED":15 ,"RED_Mean5":16,  
                "GREEN":17, "GREEN_Mean5":18,  
                 "GRVI":19, "GRVI_Mean5":20,"GRVI_Laplacian5":21 }
                 

combinations = list(np.zeros(len(unique)))

for i in range(len(unique)):
    combinations[i] = [bandNames[x] for x in os.path.basename(unique[i]).split(" + ")]
    
    
# 
primeras = list(row27col12['combination'])+ list(row29col8['combination'])
resumen = list(set(primeras))
len(resumen)
len(list(set(row27col12['combination'])))
len(list(set(row29col8['combination'])))

########################################################################
import sys
sys.path.append("/home/trashtos/GitHub/OBIA")
from ownUtilities import cleanTiles
from rsgislibWrappers import  ShepherdSegTest

tmpath = "/media/trashtos/Meerkat/Ramiro_Masterarbeit/Segmentation/Temp"

cluster = 80
pixel = 80

stackTilesCubic = [base + "_stack_cubic.kea" for base in cleanTiles()] 

segStacks = [stackTilesCubic[-1], stackTilesCubic[-2], stackTilesCubic[16], 
                        stackTilesCubic[5], stackTilesCubic[7], stackTilesCubic[12] ]




stack = segStacks[3] 
print(stack)

def wrapper(combi):
    tmpath = "/media/trashtos/Meerkat/Ramiro_Masterarbeit/Segmentation/Temp"
    inImage = '/media/trashtos/Meerkat/cleanTiles/Rows27/Cols12/tile_row27col12_stack_cubic.kea'
    outShape = os.path.splitext(inImage)[0] + "_80_80"+ ''.join([str(i) for i in combi]) +  "_clumpsexport.shp"
    
    if not os.path.exists(outShape):
        try:
            ShepherdSegTest(inImage, 80, 80,tmpath, band =  combi)
        except:
            print ("error in file", combi)
        
i = 0
for comb in combinations:
    
    tmpath = "/media/trashtos/Meerkat/Ramiro_Masterarbeit/Segmentation/Temp"
    outShape = os.path.splitext(stack)[0] + "_80_80_"+ ''.join([str(i) for i in comb]) +  "_clumpsexport.shp"
    
    if os.path.exists(outShape):
        pass
    else:
        #try:
         #   ShepherdSegTest(stack, 80, 80,tmpath, band =  comb)
       # except:
       #     print ("error in file", comb)
        i += 1
        

# get subset of files and compare



