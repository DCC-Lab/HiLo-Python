import globalUtilities as gu
import processingHiLoImages as hilo
import motionCorrection as mc
import registrationModule as reg
from rich import print
import time


if __name__ == "__main__":
    mainDirectory = "/Users/dcclab/Desktop/MainDir/"
    unifMainDirPath = "/Users/dcclab/Desktop/MainDir/Test_unif_all/"
    speckleMainDirPath = "/Users/dcclab/Desktop/MainDir/Test_speck_all/"
    timeAcquisitionPath = "/Users/dcclab/Desktop/MainDir/Time_Acquisition/"
    resultDirectory = "/Users/dcclab/Desktop/MainDir/Results_Of_Processing/"
    globalConstants = gu.setParameterForHiLoComputation()
    allData = gu.extractDataFromDirectoriesForHiLoProcessing(
        unifMainDirPath,
        speckleMainDirPath,
        timeAcquisitionPath
    )
    imagePerZstackForTimeAcquisition = 26
    
    # Compute anatomic HiLo images
    anatomicHiLoImages = hilo.runWholeHiLoImageProcessing(
        globalConstants, 
        mainDirectory, 
        unifMainDirPath, 
        speckleMainDirPath
    )
    
    # Treatement of the time acquisition datas
    # timeAcqInOrderOfPlanes, timeAcqDirectoryToCorrect = gu.createTifFilesOfPlanesFromTimeAcquisition(
    #     resultDirectory,
    #     allData[2], 
    #     imagePerZstackForTimeAcquisition
    # )
    # mc.wholeMotionCorrectionOnDirectory(timeAcqDirectoryToCorrect)
    # test = gu.temporalMeanOfTimeAcquisition(timeAcqDirectoryToCorrect, resultDirectory)
