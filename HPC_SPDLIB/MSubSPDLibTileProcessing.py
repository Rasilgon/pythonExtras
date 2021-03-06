#! /usr/bin/env python

############################################################################
# Purpose: A class to submit jobs to the bwHPC cluster
# Author: Ramiro Silveyra Gonzalez
# 
# Based on:
#
# 		Purpose:  A class to submit jobs to loadleveler
# 		Author: Pete Bunting
# 		Email: petebunting@mac.com
# 		Date: 19/04/2013
# 		Version: 1.0
# 		History:
# 		Version 1.0 - Created.
#		Copyright (c) 2013 Dr. Peter Bunting, Aberystwyth University
#
#############################################################################

import os.path
import sys
import argparse
import os
# Import python XML Parser
import xml.etree.ElementTree as ET

class BuildBSubCommands (object):
    
    def createBSubScripts(self, inputDIR, inputTilesXML):
        inputTilesXML = os.path.abspath(inputTilesXML)
        inputDIR = os.path.abspath(inputDIR)
        tree = ET.parse(inputTilesXML)
        root = tree.getroot()
        if root.tag == "tiles":
            rows = int(root.attrib['rows'])
            cols = int(root.attrib['columns'])
            print("ROWS = ", rows)
            print("COLS = ", cols)
            
            for row in range(rows):
                for col in range(cols):
                    dirRowPath = os.path.join(inputDIR, "Rows"+str(row+1)) 
                    dirRowColPath = os.path.join(dirRowPath, "Cols"+str(col+1)) 
                    lidarOutBase = "tile_row"+str(row+1)+"col"+str(col+1)+"_2014_DHDNGK4"#change to something makes sense for me
                    jobName = "TileProcessing_row"+str(row+1)+"col"+str(col+1)
                    jobFileName = jobName+".sh"
                    print (jobFileName)

                    
                    if(os.path.exists(dirRowColPath)):
                        ##### CREATE LoadLeveler File #####
                        jobFilePath = os.path.join(dirRowColPath, jobFileName)
                        jobFile = open(jobFilePath, 'w')                        
                        ##### ADD MOAB/Slurm header ####                        
                        jobFile.write("#!/bin/bash\n")
                        jobFile.write("########## Begin MOAB/Slurm header ##########\n")
                        jobFile.write("#")
                        jobFile.write("# Give job a reasonable name\n")
                        jobFile.write("# MSUB -N "+ jobName + str("\n"))
                        jobFile.write("# Request number of nodes and CPU cores per node for job\n")
                        jobFile.write("# #MSUB -l nodes=1:ppn=1 \n")
                        jobFile.write("# Estimated wallclock time for job\n")
                        jobFile.write("# MSUB -l walltime=00:59:00\n")
                        jobFile.write("# Write standard output and errors in same file\n")
                        jobFile.write("# MSUB -j oe \n")
                        jobFile.write("# Send mail when job begins, aborts and ends\n")
                        jobFile.write("# MSUB -m bae \n")
                        jobFile.write("# Set memory\n")
                        jobFile.write("# MSUB -l pmem=4000mb\n")
                        jobFile.write("########### End MOAB header ##########\n")
                        jobFile.write("ulimit -v 4194304 -m 4194304\n\n")
                        #### EXPORT GDAL DRIVER KEA
                        jobFile.write("############# export GDAL_DRIVER_KEA #############\n")
                        jobFile.write("export GDAL_DRIVER_PATH=~/anaconda3/lib/gdalplugins:$GDAL_DRIVER_PATH\n")
                        jobFile.write("export GDAL_DATA=~/anaconda3/envs/spdlibtest/share/gdal:$GDAL_DATA\n")
                        ##### ADD CHM COMMAND #####
                        jobFile.write("############# CHM #############\n")
                        jobFile.write("spdinterp --chm --height --in NATURAL_NEIGHBOR --minchmthres 0.5 -f KEA -b 1 -c 50 -r 50 --overlap 10 -i " + os.path.join(dirRowColPath, lidarOutBase + "_10m_rmn_pmfmccgrd_h.spd") + " -o " + os.path.join(dirRowColPath, lidarOutBase + "_rmn_pmfmccgrd_1m_CHM_height05_natnei.kea") + "\n")
                        jobFile.write("gdalcalcstats " + os.path.join(dirRowColPath, lidarOutBase + "_rmn_pmfmccgrd_1m_CHM_height05_natnei.kea") + " -ignore 0\n")
                        jobFile.write("###########################################\n\n\n")
                        jobFile.close()
                        cmd = "msub  " + jobFilePath
                        os.system(cmd)

    
    def run(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--dir", dest="inputdir", type=str, help="Input directory")
        parser.add_argument("-t", "--tiles", dest="tilesXMLfile", type=str, help="Input tile XML file.")

        args = parser.parse_args()    

        if args.inputdir is None:
            print("No input directory specified.")
            parser.print_help()
            sys.exit()
            
        if args.tilesXMLfile is None:
            print("No input tiles XML file specified.")
            parser.print_help()
            sys.exit()
            
        self.createBSubScripts(args.inputdir, args.tilesXMLfile)

if __name__ == '__main__':
    obj = BuildBSubCommands()
    obj.run()
