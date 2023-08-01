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
    unifMainDirPath = "/Users/dcclab/Desktop/MainDir/Test_unif1/"
    speckleMainDirPath = "/Users/dcclab/Desktop/MainDir/Test_speck1/"
    timeAcquisitionPath = "/Users/dcclab/Desktop/MainDir/Time_Acquisition/"
    resultDirectory = "/Users/dcclab/Desktop/MainDir/Results_Of_Processing/"
    
    globalConstants = gu.setParameterForHiLoComputation()
    baseUniformZstack, baseSpeckleZstack, timeAcqNotOrdered = gu.extractDataFromDirectoriesForHiLoProcessing(
        unifMainDirPath,
        speckleMainDirPath,
        timeAcquisitionPath
    ) 
    imagePerZstackForTimeAcquisition = 26 
     
    # Compute anatomic HiLo images
    for i in np.arange(1, 15, 0.2):
        print(i)
        globalConstants["waveletGaussiansRatio"] = 2
        globalConstants["sigma"] = 1.5
        globalConstants["sigmaLP"] = 5.2 * globalConstants["sigma"] * (1 + globalConstants["waveletGaussiansRatio"] / 2)
        globalConstants["superGauss"] = 1
        globalConstants["eta"] = i
        # globalConstants["readoutNoiseVariance"] = 1.64
        # globalConstants["magnification"] = 40
        # globalConstants["illuminationAperture"] = 0.3
        # globalConstants["detectionAperture"] = 0.4
        # globalConstants["pixelSize"] = 22.2
        anatomicHiLoImages = hilo.runWholeHiLoImageProcessingOnDir(
            globalConstants, 
            mainDirectory, 
            unifMainDirPath, 
            speckleMainDirPath,
            f"{i}_anatomic_HiLo_images.tif"
        )
    
    # Treatement of the time acquisition datas
    # timeAcqInOrderOfPlanes, timeAcqDirectoryToCorrect = gu.createTifFilesOfPlanesFromTimeAcquisition(
    #     resultDirectory,
    #     timeAcqNotOrdered, 
    #     imagePerZstackForTimeAcquisition
    # )
    # mc.wholeMotionCorrectionOnDirectory(timeAcqDirectoryToCorrect)
    # uniformFunctionnalZstack = gu.temporalMeanOfTimeAcquisition(timeAcqDirectoryToCorrect, resultDirectory)

    # speckleWarpedOnFunctionnal = reg.applyRegistrationToTwoImageStacks(
    #     resultDirectory,
    #     baseSpeckleZstack,
    #     "baseSpeckle.nrrd",
    #     uniformFunctionnalZstack,
    #     "uniformFunctionnal.nrrd"
    # )

    # speckleWarpedOnFunctionnal = tf.imread(resultDirectory +  "Registration_Results/baseSpeckleWarpedOnuniformFunctionnal.tif")

    # Compute functionnal HiLo images for neuron positions
    # functionnalHiLoImagesNeuronSegm = hilo.runWholeHiLoImageProcessingOnArray(
    #     globalConstants, 
    #     mainDirectory,
    #     uniformFunctionnalZstack,
    #     speckleWarpedOnFunctionnal, 
    #     "functionnal_HiLo_images_neuron_segm.tif"
    # )
     
