"""
Brief explanation on what the module does
"""
import numpy as np
import subprocess as sub
import nrrd
import tifffile as tf
import os
import glob
from rich import print


def registrationToConsole(
        whereToStoreResults: str, 
        movingReferentialStack: str, 
        fixedReferentialStack: str,
        alignedFilename: str
    ) -> str:
    """DOCS
    """
    alignedStackName = whereToStoreResults + alignedFilename
    antsPath = "/Users/dcclab/Documents/where_ants_is/install/bin"
    print("Registration starting!")
    os.system(
            f"export ANTSPATH={antsPath};"
            + f"export PATH=$ANTSPATH:$PATH;"
            + f"export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=18;"
            + f"antsRegistration -d 3 --float 1 -v --output [,{alignedStackName}] --interpolation gaussian" 
            + f" --use-histogram-matching 0 -r [{fixedReferentialStack},{movingReferentialStack},1]" 
            + f" -t rigid[0.05]"
            + f" -m MI[{fixedReferentialStack},{movingReferentialStack},1,32,Regular,0.25] -c [1000x500x250x125,1e-8,10]"
            + f" --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox" 
            + f" -t Affine[0.1] -m"
            + f" MI[{fixedReferentialStack},{movingReferentialStack},1,32,Regular,0.25] --convergence"
            + f" [1000x500x250x125,1e-8,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox" 
            + f" -t SyN[0.2,6,0] -m" 
            + f" CC[{fixedReferentialStack},{movingReferentialStack},1,2] -c [200x200x200x200x20,1e-8,10]" 
            + f" --shrink-factors 12x8x4x2x1 --smoothing-sigmas 4x3x2x1x0vox"
    )
    removeTransformsFilesAfterRegistration(os.getcwd())
    return


def applyRegistrationToTwoImageStacks(
        whereToStoreResults: str, 
        movingReferentialStackData: np.ndarray,
        movingReferentialStackName: str,
        fixedReferentialStackData: np.ndarray,
        fixedReferentialStackName: str
    ) -> np.ndarray:
    """DOCS
    """
    alignedStackName = (
        movingReferentialStackName.split(".")[0] 
        + "WarpedOn" 
        + fixedReferentialStackName.split(".")[0] 
        + ".nrrd"
    ) 
    saveStackAsNrrd(
        whereToStoreResults + "Registration_Results/", 
        movingReferentialStackData, 
        movingReferentialStackName
    )
    saveStackAsNrrd(
        whereToStoreResults + "Registration_Results/", 
        fixedReferentialStackData, 
        fixedReferentialStackName
    )
    registrationToConsole(
        whereToStoreResults + "Registration_Results/",
        whereToStoreResults + "Registration_Results/" + movingReferentialStackName,
        whereToStoreResults + "Registration_Results/" + fixedReferentialStackName,
        alignedStackName
    )
    movingWarpedOnFixed = readNrrdStackAndStaveAsTif(whereToStoreResults + "Registration_Results/", alignedStackName)
    return movingWarpedOnFixed 


def removeTransformsFilesAfterRegistration(directoryToRemove: str) -> None:
    """DOCS
    """
    filesInWorkingDir = glob.glob(directoryToRemove + "/*.*")
    filesInWorkingDirToRemove = [
        file for file in filesInWorkingDir if file.split(".")[1] == "mat" or file.split(".")[1] == "nii" 
    ]
    for file in filesInWorkingDirToRemove:
        os.remove(file)
    return


def saveStackAsNrrd(whereToSave: str, stackDataToSave: str, nameOfFileToSave: str) -> None:
    """DOCS
    """
    nrrd.write(whereToSave + nameOfFileToSave, np.array(stackDataToSave), index_order = "C")
    return


def readNrrdStackAndStaveAsTif(whereToSave: str, nameOfFileToRead: str) -> np.ndarray: 
    """DOCS
    """
    dataOfFile, headerOfFile = nrrd.read(whereToSave + nameOfFileToRead, index_order = "C")
    tf.imwrite(whereToSave + nameOfFileToRead.split(".")[0] + ".tif", dataOfFile)
    return dataOfFile


if __name__ == "__main__":
    pass
