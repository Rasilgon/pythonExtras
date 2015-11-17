# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:28:31 2015

@author: trashtos
"""

# Teh best ios 3 7 12 21

# Now we test clump size and parameter size
# 
values = range(30, 201, 10)



#
########################################################################
import sys
sys.path.append("/home/trashtos/GitHub/OBIA")
from ownUtilities import cleanTiles
from rsgislibWrappers import  ShepherdSegTest


#######################################################################


#combinations = [list(x) for x in all_subsets(list(range(5, 22))) if len(x) < 4]

#comb = [list(x)  for x in all_subsets(range(5, 22))]
tmpath = "/media/trashtos/Meerkat/Ramiro_Masterarbeit/Segmentation/Temp"



stackTilesCubic = [base + "_stack_cubic.kea" for base in cleanTiles()] 

segStacks = [stackTilesCubic[-1], stackTilesCubic[-2], stackTilesCubic[16], 
                        stackTilesCubic[5], stackTilesCubic[7], stackTilesCubic[12] ]




stack = segStacks[1] 
print(stack)

def wrapper(cluster, pixels, stack):
    tmpath = "/media/trashtos/Meerkat/Ramiro_Masterarbeit/Segmentation/Temp2"
    bands = [3, 7, 12, 21]
    try:
        ShepherdSegTest(stack, cluster, pixels,tmpath, band =  bands)
    except:
        print ("error in file")
        

i =0
for cluster in values:
    if cluster > 140:
        for pixel in values:  
            i +=1 
            
            
            wrapper(cluster,pixel, stack)
            

        



