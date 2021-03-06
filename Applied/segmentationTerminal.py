# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:07:45 2015

@author: trashtos
"""
#
########################################################################
import sys
sys.path.append("/home/trashtos/GitHub/OBIA")
from ownUtilities import cleanTiles
from rsgislibWrappers import  ShepherdSegTest


#######################################################################


#######
from itertools import chain, combinations
def all_subsets(ss, length):
  return chain(*map(lambda x: combinations(ss, x), range(0, length)))



height = [1, 2, 3,4]

#combinations = [list(x) for x in all_subsets(list(range(5, 22))) if len(x) < 4]

#comb = [list(x)  for x in all_subsets(range(5, 22))]
tmpath = "/media/trashtos/Meerkat/Ramiro_Masterarbeit/Segmentation/Temp"

cluster = 80
pixel = 80

stackTilesCubic = [base + "_stack_cubic.kea" for base in cleanTiles()] 

segStacks = [stackTilesCubic[-1], stackTilesCubic[-2], stackTilesCubic[16], 
                        stackTilesCubic[5], stackTilesCubic[7], stackTilesCubic[12] ]




stack = segStacks[1] 
print(stack)

def wrapper(combi):
    tmpath = "/media/trashtos/Meerkat/Ramiro_Masterarbeit/Segmentation/Temp"
    bands = [3] + list(combi)
    inImage = '/media/trashtos/Meerkat/cleanTiles/Rows27/Cols12/tile_row27col12_stack_cubic.kea'
    try:
        ShepherdSegTest(inImage, 80, 80,tmpath, band =  bands)
    except:
        print ("error in file")
        

for comb in all_subsets(range(5, 22)):
    wrapper(comb )

#wrapper(comb[833] )
