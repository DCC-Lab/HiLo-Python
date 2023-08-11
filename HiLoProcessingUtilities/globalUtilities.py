"""
Module containing useful functions used in all the process flow of HiLo data processing
"""
import glob
import os
import numpy as np
import tifffile as tf
from rich import print
import shutil


def sortListOfImageFiles(imageFilePath: str) -> int:
    """DOCS
    """
    return int(imageFilePath.split("TimeLapse_")[1].split("-")[0])


def sortListOfTimeAcqPerPlaneFiles(imageFilePath: str) -> int:
    """DOCS
    """
    return int(imageFilePath.split("planeNumber")[1].split("UncorrectedImages")[0])


def sortListOfImageFilesSparq(imageFilePath: str) -> float:
    """DOCS
    """
    typeOfImage = imageFilePath.split("Sparq_")[1].split(".")[0]
    timeOfImage = int(imageFilePath.split("Frame_")[1].split("-")[0])
    if typeOfImage == "Processed":
        return 1 + timeOfImage
    elif typeOfImage == "Uniform":
        return 2 + timeOfImage
    elif typeOfImage == "Speckle":
        return 3 + timeOfImage


def readFilesInDirectory(directoryToRead: str, funcToSort) -> list:
    """DOCS
    """
    allFilesInDirectory = [files for files in glob.glob(directoryToRead + "*.tif")]
    return sorted(allFilesInDirectory, key = funcToSort)


def obtainAllImagePath(unifDirPath: str, speckDirPath: str, timeAcquisitionPath: str) -> list:
    """DOCS
    """
    return [readFilesInDirectory(directory, sortListOfImageFiles) 
            for directory in (unifDirPath, speckDirPath, timeAcquisitionPath)]


def obtainAllImageDataFromPath(allImagePathForHiLoProcessing: list) -> list:
    """DOCS
    """
    allImageDataForHiLoProcessing = [
        [tf.imread(path) for path in pathLists] for pathLists in allImagePathForHiLoProcessing
    ]
    return allImageDataForHiLoProcessing


def extractDataFromSparqDirectory(sparqDirectoryPath: str) -> list:
    """DOCS
    """
    pathsInDir = readFilesInDirectory(sparqDirectoryPath, sortListOfImageFilesSparq)
    allProcessedPaths = pathsInDir[0::3] 
    allUnifPaths = pathsInDir[1::3]
    allSpeckPaths = pathsInDir[2::3]
    return allProcessedPaths, allUnifPaths, allSpeckPaths


def extractDataFromDirectoriesForHiLoProcessing(unifDirPath: str, speckDirPath: str, timeAcquisitionPath: str) -> tuple:
    """DOCS
    """
    allImagePath = obtainAllImagePath(unifDirPath, speckDirPath, timeAcquisitionPath)
    allImageData = obtainAllImageDataFromPath(allImagePath)
    return (*allImageData,)


def removeWasteImagesFromTimeAcquisition(
        timeAcquisitionData: list, 
        imagesPerZstack: int
    ) -> list:
    """DOCS
    """
    indicesToRemove = [idx for idx in range(len(timeAcquisitionData)) if idx % imagesPerZstack == 0]
    return np.delete(timeAcquisitionData, indicesToRemove, axis = 0)


def mergePlanesOfDifferentZstacks(noWasteImageTimeAcquisitionData: list, imagesPerZstack: int) -> list:
    """DOCS
    """
    numberOfZstack = len(noWasteImageTimeAcquisitionData) / (imagesPerZstack - 1)
    splittedZstack = np.array_split(noWasteImageTimeAcquisitionData, numberOfZstack)
    allPlanesMatched = [np.array(i) for i  in zip(*splittedZstack)]
    return allPlanesMatched


def createTifFilesOfPlanesFromTimeAcquisition(
        directoryToInputResults: str, 
        timeAcquisitionData: list, 
        imagesPerZstack: int
    ) -> tuple:
    """DOCS
    """
    directoryToInputData = directoryToInputResults + "timeAcquisitionInOrderOfPlanes/"  
    noWasteImageData = removeWasteImagesFromTimeAcquisition(timeAcquisitionData, imagesPerZstack)
    mergedPlanesFromZstacks = mergePlanesOfDifferentZstacks(noWasteImageData, imagesPerZstack)
    createDirectoryIfInexistant(directoryToInputData)
    for index, plane in enumerate(mergedPlanesFromZstacks):
        tf.imwrite(directoryToInputData + f"planeNumber{index + 1}UncorrectedImages.tif", plane)
    return mergedPlanesFromZstacks, directoryToInputData


def createDirectoryIfInexistant(directoryName: str) -> None:
    """
    This function creates a directory if it is not yet been created.

    Parameters
    ----------
    directoryName : string
        The name of the directory to create

    """
    if os.path.exists(directoryName) == False:
        os.mkdir(directoryName)
    return


def setParameterForHiLoComputation() -> dict:
    """DOCS
    """
    parameters = {    
        "cameraGain" : 1,
        "readoutNoiseVariance" : 0.3508935,
        "sigma" : 2,
        "waveletGaussiansRatio" : 2,
        "windowValue" : 7,
        "illuminationAperture" : 1,
        "detectionAperture" : 1,
        "illuminationWavelength" : 488e-9,
        "detectionWavelength" : 520e-9,
        "pixelSize" : 4.5,
        "magnification" : 20,
        "superGauss": 1,
        "eta": 0
    }
    parameters["sigmaLP"] = 5.2 * parameters["sigma"] * (1 + parameters["waveletGaussiansRatio"] / 2)
    return parameters


def sortMotionCorrectedImageFiles(imageFilePath: str) -> int:
    """DOCS
    """
    return int(imageFilePath.split("planeNumber")[1].split("U")[0])


def readFilesInAllCorrectedDirectories(directoryWhereCorrectedAre: str) -> list:
    """DOCS
    """
    allFilesNotMergedInSubdirectory = [files for files in glob.glob(directoryWhereCorrectedAre + "*/corrected_0.tif")] 
    allFilesMergedInSubdirectory = [files for files in glob.glob(directoryWhereCorrectedAre + "*/corrected_and_*.tiff")]
    sortedListOfNotMergedFiles = sorted(allFilesNotMergedInSubdirectory, key = sortMotionCorrectedImageFiles)
    sortedListOfMergedFiles = sorted(allFilesMergedInSubdirectory, key = sortMotionCorrectedImageFiles)
    
    if len(allFilesMergedInSubdirectory) == 0:
        return getDataOfCorrectedFilesInCorrectedDirectories(sortedListOfNotMergedFiles)

    elif len(allFilesNotMergedInSubdirectory) == len(allFilesMergedInSubdirectory):
        return getDataOfCorrectedFilesInCorrectedDirectories(sortedListOfMergedFiles)
    
    else:
        sortedListOfMergedAndNotMergedFiles = keepOnlyCorrectedOrMergedFiles(
            sortedListOfNotMergedFiles, 
            sortedListOfMergedFiles
        )
        return getDataOfCorrectedFilesInCorrectedDirectories(sortedListOfMergedAndNotMergedFiles)


def keepOnlyCorrectedOrMergedFiles(listOfNotMergedFiles: list, listOfMergedFiles: list) -> list:
    """DOCS
    """
    listOfPlaneNumbersNotMerged = [
        int(file.split("planeNumber")[1].split("U")[0]) for file in listOfNotMergedFiles
    ]
    listOfPlaneNumbersMerged = [
        (int(file.split("planeNumber")[1].split("U")[0]), file) 
        for file in listOfMergedFiles
    ] 

    for planeNumbers, file in listOfPlaneNumbersMerged:
        listOfNotMergedFiles[planeNumbers - 1] = file

    return listOfNotMergedFiles


def getDataOfCorrectedFilesInCorrectedDirectories(listOfCorrectedFiles: list) -> list:
    """DOCS
    """
    return [tf.imread(file) for file in listOfCorrectedFiles]


def removeDirectory(directoryPath: str) -> None:
    """DOCS
    """
    return shutil.rmtree(directoryPath)


def temporalMeanOfTimeAcquisitionAfterMc(
        timeAcqDirectoryToCorrect: str, 
        directoryToInputResults: str,
        nameOfAveragedAndMotionCorrectedFile: str
    ) -> list:
    """DOCS
    """
    nameOfFile = directoryToInputResults + nameOfAveragedAndMotionCorrectedFile
    eachCorrectedPlanes = readFilesInAllCorrectedDirectories(timeAcqDirectoryToCorrect)
    averagedAndMotionCorrectedPlanes = [np.mean(plane, axis = 0) for plane in eachCorrectedPlanes]
    
    # removeDirectory(timeAcqDirectoryToCorrect)

    tf.imwrite(nameOfFile, averagedAndMotionCorrectedPlanes)
    return averagedAndMotionCorrectedPlanes


def temporalMeanOfTimeAcquisitionNoMc(
        timeAcqDirectoryToCorrect: str,
        directoryToInputResults: str,
        nameOfAveragedFile: str
    ) -> list:
    """DOCS
    """
    nameOfFile = directoryToInputResults + nameOfAveragedFile
    eachPlanes = readFilesInDirectory(timeAcqDirectoryToCorrect, sortListOfTimeAcqPerPlaneFiles)
    averagedPlanes = [np.mean(tf.imread(plane), axis = 0) for plane in eachPlanes]
    tf.imwrite(nameOfFile, averagedPlanes)
    return averagedPlanes 


if __name__ == "__main__":
    pass
