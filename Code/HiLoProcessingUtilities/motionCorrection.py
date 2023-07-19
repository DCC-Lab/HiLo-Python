"""
------------------------------------------------------------------------------------------------------------------------
Script that sends a whole directory to be motion corrected. There are a few word distinctions to make before reading
the code:

Directory : Main directory that contains single or multiple `.tiff` files to correct

File : Main `.tiff` file contained in the directory that need motion correction

Subfile : To correct the main file faster, we split it into multiple smaller files called subfiles that will all 
          undergo motion correction

Subdirectory : Sub directories containing the subfiles that will need motion correction
------------------------------------------------------------------------------------------------------------------------

Example on the main directory for only one file to correct after the correction :

                              Directory
                                  |
                Subdirectory 1 ___|___ File 1               
                    |                                                     
       Subfile 1 ___|... ___ Subfile N ___ CorrectedSubfile 1 ___... ___ CorrectedSubfile N ___ Correctedfile 1
        
------------------------------------------------------------------------------------------------------------------------
"""
import calimba.processing.caiman.CaimanPDK as cal
import caiman as cm
import tifffile as tf
import os
import ants 
import numpy as np
import globalUtilities as gu


def wholeMotionCorrectionOnDirectory(directoryToCorrect: str) -> None:
    """
    This function determines the files to correct and sends individual motion correction onto each one.  

    Parameters
    ----------
    directoryToCorrect : string
        Path of the main directory where all the files needing motion correction are located
    """
    uncorrectedFiles = cal.find_uncorrected_files(directoryToCorrect) # Find files to correct in directory
    print("Files to correct in directory: ", uncorrectedFiles) 
    for fileToCorrect in uncorrectedFiles:
        motionCorrectionOnFile(directoryToCorrect, fileToCorrect)
    return


def motionCorrectionOnFile(directoryToCorrect: str, fileToCorrect: str) -> None:
    """
    This function creates the subdirectories and the subfiles, apply motion correction on each one of those subfile and
    finally merges them back together using image registration to align them correctly. 

    Parameters
    ----------
    directoryToCorrect : string
        Path of the main directory where all the files needing motion correction are located

    fileToCorrect : string
        Name of the file to apply motion correction onto
    """
    subfilesDirectory = createSubdirectoryAppendSubfiles(directoryToCorrect, fileToCorrect)
    correctedSubfiles = correctTheSubfilesInSubdirectory(subfilesDirectory)
    if len(correctedSubfiles) > 1:
        applyRegistrationToFramesOfSubfileAndMerge(correctedSubfiles)    
    return

def createSubdirectoryAppendSubfiles(
        directoryToCorrect: str, 
        fileToCorrect: str
    ) -> str:
    """
    Function that creates the subdirectory and append the subfiles to it. These subfiles all come from the original file 
    to correct and will each need independant correction.

    Parameters
    ----------
    directoryToCorrect : string
        Path of the main directory where all the files needing motion correction are located

    fileToCorrect : string
        Name of the file to apply motion correction onto

    Returns
    -------
    nameOfSubdirectory : string
        Name of the subdirectory where all the subfiles, corrected subfiles and corrected file will be 
    """
    fileAsArray = tf.imread(directoryToCorrect + fileToCorrect)
    numberOfSubfilesToCreate = computeNumberOfSubfilesForFileToCorrect(fileAsArray)

    print(f"The file {fileToCorrect} will be splitted into", numberOfSubfilesToCreate, "subfiles to correct.")

    nameOfSubdirectory = splitFileCreateSubdirectoryAddSubfiles(
        fileAsArray, 
        numberOfSubfilesToCreate, 
        directoryToCorrect, 
        fileToCorrect
    )
    return nameOfSubdirectory


def correctTheSubfilesInSubdirectory(nameOfSubdirectory: str) -> list:
    """
    This function sends motion correction on each `.tiff` subfile and their corrected corresponding subfile will be 
    outputed as `.tiff` in the same directory.

    Parameters
    ----------
    nameOfSubdirectory : string
        Name of the subdirectory where all the subfiles, corrected subfiles and corrected file will be

    Returns
    -------
    nameOfCorrectedSubfiles : list of strings
        List that contains all the names of the corrected subfiles
    """
    applyMotionCorrectionWithCaiman(nameOfSubdirectory)
    os.chdir(nameOfSubdirectory)
    nameOfCorrectedSubfiles = sorted(getNamesOfCorrectedSubfiles(nameOfSubdirectory))
    return nameOfCorrectedSubfiles


def applyRegistrationToFramesOfSubfileAndMerge(nameOfCorrectedSubfiles: str) -> None:
    """
    This function applies image registration on the corrected subfiles to align them all together after motion 
    correction. They will then be merged together to create the final corrected file. The registration is done choosing
    a fixed referential subfile on which all the other subfiles will be maped. Here, the first subfile will be chosen as
    the fixed referential, and the second subfile will be the moving referential. Then the transformation computed with
    the registration will be applied individually onto all the frames of all the subfiles (except the first, chosen as 
    the fixed referential). It is also important to know that we will use the mean frame of both these fixed and moving 
    subfile.

    Parameters
    ----------
    nameOfCorrectedSubfiles : string
        List that contains all the names of the corrected subfiles
    """
    correctedSubfilesAsArray = [
            np.array(tf.imread(correctedSubfile), dtype = np.float32) 
            for correctedSubfile in nameOfCorrectedSubfiles
    ]
    fixedCorrectedSubfile = correctedSubfilesAsArray[0]

    # Already include the reference corrected subfile which doesnt need registration
    correctedAndRegisteredSubfilesIntoSingleArray = [frames.astype(np.float16) for frames in fixedCorrectedSubfile]

    registrationResult, meanFixedFrame = applyRegistrationOnMeanFixedAndMovingSubfiles(
            fixedCorrectedSubfile, 
            correctedSubfilesAsArray
    )

    applyTransformationToCorrectedSubfileFrames(
            correctedSubfilesAsArray, 
            registrationResult, 
            meanFixedFrame, 
            correctedAndRegisteredSubfilesIntoSingleArray   
    )

    mergeSubfilesIntoOneCorrectedFile("corrected_and_merged.tiff", correctedAndRegisteredSubfilesIntoSingleArray)
    return


def computeNumberOfSubfilesForFileToCorrect(fileAsArray: np.ndarray) -> int:
    """
    This function computes the number of subfiles the principal `.tiff` file will be splitted into to perform motion 
    correction. It is to be noted that each subfile as a maximum size of 2GB. 

    Parameters
    ----------
    fileAsArray : numpy array of numpy arrays
        Principal `.tiff` file as a single numpy array containing all image frames

    Returns
    -------
    numberOfSubfilesToCreate : integer
        The number of subfiles the principal file will be splitted into
    """
    sizeForWholeFile = fileAsArray.size
    numberOfImagesInFile = fileAsArray.shape[0]
    GbInBytes = 1e-9 # Conversion factor
    maximumSizeForOneSubfile = 2 # Size in GB

    sizePerImageInFile = (sizeForWholeFile / numberOfImagesInFile) * GbInBytes
    numberOfImagesInOneSubfile = maximumSizeForOneSubfile / sizePerImageInFile
    ratioImagesInFileToImagesInOneSubfile = numberOfImagesInFile / numberOfImagesInOneSubfile
    numberOfSubfilesToCreate = ratioImagesInFileToImagesInOneSubfile * maximumSizeForOneSubfile
    
    if numberOfSubfilesToCreate >= maximumSizeForOneSubfile:
        return int(numberOfSubfilesToCreate)
    else:
        return 1
    

def splitFileCreateSubdirectoryAddSubfiles(
        fileAsArray: np.ndarray, 
        numberOfSubfilesToCreate: int, 
        directoryToCorrect: str,
        fileToCorrect: str
    ) -> str:
    """
    This function split the main file as multiple subfiles given the `numberOfSubfilesToCreate` parameters, creates the
    subdirectory and appends the subfiles as `.tiff` to it.

    Parameters
    ----------
    fileAsArray : numpy array
        Principal `.tiff` file as a single numpy array containing all image frames

    numberOfSubfilesToCreate : integer
        Number of subfiles that will be create based on the size of the main file

    fileToCorrect : string
        Principal `.tiff` file as a single numpy array containing all image frames

    Returns
    -------
    nameOfSubdirectory : string
        The name of the subdirectory created that will contain the subfiles, the corrected subfiles and the corrected
        main file
    """
    nameOfSubdirectory = directoryToCorrect + fileToCorrect.split(".")[0] + "/"
    splittedFileAsArray = np.array_split(fileAsArray, numberOfSubfilesToCreate) 
    gu.createDirectoryIfInexistant(nameOfSubdirectory)
    buildSubfilesInSubdirectory(splittedFileAsArray, nameOfSubdirectory)
    return nameOfSubdirectory


def applyMotionCorrectionWithCaiman(nameOfDirectoryToCorrect: str) -> None:
    """
    This function applies motion correction to all the `.tiff` files in the specified directory. It uses the CaImAn 
    software tools via the calimba package. The parameters for the motion correction are all chosen for the particular 
    correction we want on zebrafishes and their effects are described in the CaImAn documentation.

    Parameters
    ----------
    nameOfDirectoryToCorrect : string
        Path of the directory the function will motion correct

    See also
    ----------
    CaImAn documentation : https://caiman.readthedocs.io/en/master/
    
    Calimba private github : https://github.com/PDKlab/Calimba
    """
    parametersForMotionCorrection = cal.get_custom_mc_params()
    parametersForMotionCorrection['strides'] = (144, 144)
    parametersForMotionCorrection['overlaps'] = (72, 72)

    cal.correct_motion_directory(nameOfDirectoryToCorrect, parameters = parametersForMotionCorrection)
    return


def getNamesOfCorrectedSubfiles(nameOfSubdirectory: str) -> list:
    """
    This function searches through the the subdirectory and finds all the files that have been corrected.

    Parameters
    ----------
    nameOfSubdirectory : string
        The name of the subdirectory created that will contain the subfiles, the corrected subfiles and the corrected
        main file

    Returns
    -------
    nameOfCorrectedSubfiles : string
        List that contains all the names of the corrected subfiles
    """
    nameOfCorrectedSubfiles = [
        nameOfSubdirectory + subfile 
        for subfile in os.listdir(nameOfSubdirectory) 
        if subfile.split("_")[0] == "corrected"
    ] 
    return nameOfCorrectedSubfiles


def applyRegistrationOnMeanFixedAndMovingSubfiles(
        fixedCorrectedSubfile: np.ndarray, 
        correctedSubfilesAsArray: list
    ) -> tuple:
    """
    This function applies the registration on the chosen mean fixed frame and the chosen mean moving frames.

    Parameters
    ----------
    fixedCorrectedSubfile : numpy array
        Array containing all the frames of the chosen fixed corrected subfile, in this case, the first corrected
        subfile

    correctedSubfilesAsArray : list of numpy arrays
        List containing all the numpy arrays corresponding to each corrected subfiles
    
    Returns
    -------
    registrationResult : dictionnary
        Dictionnary that contains the result of the registration (see the ANTsPy documentation here: 
        https://antspy.readthedocs.io/en/latest/index.html for the contents of the dictionnary)

    meanFixedFrame : numpy array
        Array containing the chosen fixed corrected subfile mean frame
    """
    meanFixedFrame, meanMovingFrame = computeMeanFixedAndMovingImages(fixedCorrectedSubfile, correctedSubfilesAsArray)
    
    registrationResult = ants.registration(
        fixed = ants.from_numpy(meanFixedFrame),
        moving = ants.from_numpy(meanMovingFrame),
        type_of_transform = "Affine"
    ) 
    return registrationResult, meanFixedFrame


def applyTransformationToCorrectedSubfileFrames(
        correctedSubfilesAsArray: list, 
        registrationResult: dict, 
        meanFixedFrame: np.ndarray, 
        correctedAndRegisteredSubfilesIntoSingleArray: list
    ) -> None:
    """
    This function applies the transformation obtained with the registration to all frames of each corrected subfiles and
    appends these corrected and transformed frames to a final list used for merging all the subfiles together.  

    Parameters
    ----------
    correctedSubfilesAsArray : list of numpy arrays
        List containing all the numpy arrays corresponding to each corrected subfiles

    registrationResult : dictionnary
        Dictionnary that contains the result of the registration (see the ANTsPy documentation here 
        https://antspy.readthedocs.io/en/latest/index.html for the contents of the dictionnary)

    meanFixedFrame : numpy array
        Array containing the chosen fixed corrected subfile mean frame

    correctedAndRegisteredSubfilesIntoSingleArray : list of numpy arrays
        List containing all the corrected and registered frames of each corrected subfiles used to create the final
        corrected file

    """
    for index, correctedSubfile in enumerate(correctedSubfilesAsArray):
        if index != 0: # Skip the first subfile because it is the fixed file
            temporaryTransformation =  appendTransformsToTemporaryList(
                    registrationResult, 
                    meanFixedFrame, 
                    correctedSubfile
            )

            appendFramesToCorrectedAndRegisteredSingleArray(
                    correctedAndRegisteredSubfilesIntoSingleArray, 
                    temporaryTransformation
            )
    return


def mergeSubfilesIntoOneCorrectedFile(
        nameOfCorrectedFile: str, 
        correctedSubfilesIntoSingleArray: list
    ) -> None:
    """
    This function merges all the corrected and registered frames into a single final corrected `.tiff` file.

    Parameters
    ----------
    nameOfCorrectedFile : string
        Name that will be given to the final corrected file

    correctedSubfilesIntoSingleArray : list of numpy arrays
        List containing all the corrected and registered frames of each corrected subfiles used to create the final
        corrected file
    """
    tf.imwrite(nameOfCorrectedFile, correctedSubfilesIntoSingleArray, bigtiff = True, dtype = np.float16)
    return


def buildSubfilesInSubdirectory(splittedFileAsArray: np.ndarray, nameOfSubdirectory: str) -> None:
    """
    This function creates all the `.tiff` subfiles into the subdirectory.

    Parameters
    ----------
    splittedFileAsArray : numpy array
        Numpy array of numpy arrays where each of them contain the information of one `.tiff` subfile  

    """
    os.chdir(nameOfSubdirectory)
    for index, subfileAsArray in enumerate(splittedFileAsArray):
        tf.imwrite(f"{index}.tif", subfileAsArray)
    os.chdir("..") 
    return


def computeMeanFixedAndMovingImages(
        fixedCorrectedSubfile: np.ndarray,
        correctedSubfilesAsArray: list
    ) -> tuple:
    """
    This function computes the fixed corrected subfile mean frame and the moving corrected subfile mean frame.

    Parameters
    ----------
    fixedCorrectedSubfile : numpy array
        Array containing all the frames of the chosen fixed corrected subfile, in this case, the first corrected
        subfile

    correctedSubfilesAsArray : list of numpy arrays
        List containing all the numpy arrays corresponding to each corrected subfiles
    
    Returns
    -------
    meanFixedFrame : numpy array
        Array containing the chosen fixed corrected subfile mean frame

    meanMovingFrame : numpy array
        Array containing the chosen moving corrected subfile mean frame
    """
    movingCorrectedSubfile = correctedSubfilesAsArray[1] # Chosen to be the second corrected image but could be other 
    meanFixedFrame = fixedCorrectedSubfile.mean(axis = 0)
    meanMovingFrame = movingCorrectedSubfile.mean(axis = 0)         
    return meanFixedFrame, meanMovingFrame


def appendTransformsToTemporaryList(
        registrationResult: dict,
        meanFixedFrame: np.ndarray, 
        correctedSubfile: list
    ) -> list:
    """
    This function applies the registration transformation to all the frames a corrected subfile and appends them to a
    temporary list.

    Parameters
    ----------
    registrationResult : dictionnary
        Dictionnary that contains the result of the registration (see the ANTsPy documentation here 
        https://antspy.readthedocs.io/en/latest/index.html for the contents of the dictionnary)

    meanFixedFrame : numpy array
        Array containing the chosen fixed corrected subfile mean frame

    correctedSubfile : numpy array
        Array corresponding to the frames of a corrected subfile
    
    Returns
    -------
    temporaryTransformation : list of ANTs images
        Temporary list containing the corrected and transformed frames of a subfile as ANTs images (see the ANTsPy 
        documentation here https://antspy.readthedocs.io/en/latest/index.html)
    """
    temporaryTransformation = [
        ants.apply_transforms(
            fixed = ants.from_numpy(meanFixedFrame), 
            moving = ants.from_numpy(frame), 
            transformlist = registrationResult["fwdtransforms"]
        ) 
        for frame in correctedSubfile
    ]
    return temporaryTransformation


def appendFramesToCorrectedAndRegisteredSingleArray(toAppendTo: list, temporaryTransformation: list) -> None:
    """
    This function appends the registered and transformed frames of a corrected subfile as a numpy array to the
    final list that contains the corrected and transformed frames of all corrected subfiles.

    Parameters
    ----------
    toAppendTo : list of numpy arrays
        final list that contains the corrected and transformed frames of all corrected subfiles

    temporaryTransformation : list of ANTs images
        Temporary list containing the corrected and transformed frames of a subfile as ANTs images (see the ANTsPy 
        documentation here https://antspy.readthedocs.io/en/latest/index.html)
    """
    for correctedAndRegisteredSubfileFrame in temporaryTransformation:
        toAppendTo.append(
            correctedAndRegisteredSubfileFrame.numpy().astype(np.float16)
        )
    return


if __name__ == "__main__":
    pass
