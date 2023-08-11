import globalUtilities as gu
import processingHiLoImages as hilo
import motionCorrection as mc
import registrationModule as reg
from rich import print
import time
import tifffile as tf
import numpy as np


if __name__ == "__main__":
    mainDirectory = "/Users/dcclab/Desktop/MainDir/"
    timeAcqSparqDir = "/Users/dcclab/Desktop/MainDir/timeAcqSparqDir/"
    resultDirectory = "/Users/dcclab/Desktop/MainDir/resultsOfProcessing/"
    gu.createDirectoryIfInexistant(resultDirectory)

    allProcessedPath, allUnifPath, allSpeckPath = gu.extractDataFromSparqDirectory(timeAcqSparqDir)
    globalConstants = gu.setParameterForHiLoComputation() 
    
    # Compute functionnal HiLo images
    functionnalHiLoImages = hilo.runWholeHiLoImageProcessingOnPaths(
        globalConstants, 
        resultDirectory, 
        allUnifPath, 
        allSpeckPath,
        f"functionnalHiLoImages.tif"
    )
  
    # Treatement of the time acquisition datas
    imagePerZstackForTimeAcquisition = 26
    timeAcqInOrderOfPlanes, timeAcqDirectoryToCorrect = gu.createTifFilesOfPlanesFromTimeAcquisition(
        resultDirectory,
        functionnalHiLoImages, 
        imagePerZstackForTimeAcquisition
    )
    #  mc.wholeMotionCorrectionOnDirectory(timeAcqDirectoryToCorrect) # Motion correction on the time acq data 
    HiLoFunctionnalZstack = gu.temporalMeanOfTimeAcquisitionNoMc(
        timeAcqDirectoryToCorrect,
        resultDirectory,
        "HiLoFunctionnalZstackTempMean.tif"
    )    
