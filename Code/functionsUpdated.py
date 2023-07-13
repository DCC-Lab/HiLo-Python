import tifffile as tf
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, fftfreq
from scipy.integrate import simpson
import math
from rich import print
import matplotlib.pyplot as plt
import time 
import multiprocess as mp
import glob
import os

# matplotlib params for figures
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.family': "serif"})
plt.rcParams.update({'figure.figsize': (6.4, 4.8)})
plt.rcParams.update({'figure.dpi': 150})


class FiltersForHiLo:
    """DOCS
    """

    def __init__(self, sizeOfFilter: tuple, filterType: str, sigma: float) -> None:
        """DOCS
        """
        self.sizeAlongKSpaceX = sizeOfFilter[0]
        self.sizeAlongKSpaceY = sizeOfFilter[1]
        self.kSpaceX, self.kSpaceY = np.meshgrid(
            fftshift(fftfreq(self.sizeAlongKSpaceX, 1)),
            fftshift(fftfreq(self.sizeAlongKSpaceY, 1)) 
        )
        self.sigma = sigma
        self.data = self.findFilterType(filterType)


    def simpleGaussianFilter(self) -> np.ndarray:
        """DOCS
        """
        sigmaFilter = (self.sigma * 0.18) / (2 * math.sqrt(2 * math.log(2)))
        kVectorDistance = self.kSpaceX**2 + self.kSpaceY**2
        simpleGaussFilter = (1 / (sigmaFilter * 2 * np.pi)) * np.exp(-(kVectorDistance / (2.0 * sigmaFilter**2)))
        normSimpleGaussFilter = simpleGaussFilter / np.max(simpleGaussFilter)
        return normSimpleGaussFilter
    

    def doubleGaussianFilter(self) -> np.ndarray:
        """DOCS
        """
        firstGaussianConstant = (
            -1.0 
            * ((2 * math.pi) / self.sizeAlongKSpaceX) 
            * ((2 * math.pi) / self.sizeAlongKSpaceY) 
            * self.sigma**2
        )
        secondGaussianConstant = (
            -1.0 
            * ((2 * math.pi) / self.sizeAlongKSpaceX) 
            * ((2 * math.pi) / self.sizeAlongKSpaceY) 
            * self.sigma**2 
            * globalConstants["waveletGaussiansRatio"]**2 
        )
        kVectorDistance = np.sqrt((self.kSpaceX)**2 + (self.kSpaceY)**2)
        doubleGaussFilter = (
            np.exp(kVectorDistance * firstGaussianConstant)
            - np.exp(kVectorDistance * secondGaussianConstant)
        )
        normDoubleGaussFilter = doubleGaussFilter / np.max(doubleGaussFilter)
        return normDoubleGaussFilter
    

    def findFilterType(self, filterType: str):
        """DOCS
        """
        filterContainer = {
            "lowpass": self.simpleGaussianFilter(),
            "highpass": 1 - self.simpleGaussianFilter(),
            "doublegauss": self.doubleGaussianFilter()
        }

        if filterType in filterContainer.keys():
            return filterContainer[filterType]
        else:
            raise Exception("The filter you want does not exist, maybe you can create it?")



class ImageForHiLo:
    """DOCS
    """

    def __init__(self, samplingWindow: int, **kwargs) -> None:
        """DOCS
        """
        if "imagePath" in kwargs: 
            self.data = (createImageFromPath(kwargs["imagePath"])).astype(np.float64)
        else:
            self.data = kwargs["imageArray"]

        self.sizeAlongXAxis = self.data.shape[0]
        self.sizeAlongYAxis = self.data.shape[1]
        self.samplingWindow = samplingWindow
        self.standardDev = np.zeros(shape = (self.sizeAlongXAxis, self.sizeAlongYAxis))
        self.mean = np.zeros(shape = (self.sizeAlongXAxis, self.sizeAlongYAxis))
        

    def fftOnImage(self) -> np.ndarray:
        """DOCS
        """
        fourierTransform = fft2(self.data)
        return fftshift(fourierTransform)


    def applyFilter(self, theFilter: FiltersForHiLo) -> np.ndarray:
        """DOCS
        """
        imageInKSpace = self.fftOnImage()
        imageFilteredInKSpace = imageInKSpace * theFilter.data
        return ifft2(imageFilteredInKSpace)


    def viewAllPixelsInSamplingWindow(self, pixelCoords: tuple, absoluteOn: bool):
        """DOCS 
        """
        n = self.samplingWindow // 2
        xPixel, yPixel = pixelCoords[0], pixelCoords[1]
        pixelValuesInSamplingWindow = []
        
        for x in range(xPixel - n, xPixel + n + 1):
            for y in range(yPixel - n, yPixel + n + 1):
                if not ((x < 0) or (y < 0) or (x > self.sizeAlongXAxis - 1) or (y > self.sizeAlongYAxis - 1)):
                    if absoluteOn == True:
                        pixelValuesInSamplingWindow.append(abs(self.data[x, y]))
                    else:
                        pixelValuesInSamplingWindow.append(self.data[x, y])
        return pixelValuesInSamplingWindow


    def stdDevOnePixelWithSamplingWindow(self, pixelCoords: tuple):
        """DOCS 
        """ 
        valuesOfPixelsInSamplingWindow = self.viewAllPixelsInSamplingWindow(pixelCoords, False)        
        return np.std(valuesOfPixelsInSamplingWindow)


    def stdDevOfWholeImage(self) -> np.ndarray:
        """DOCS
        """
        for x in range(self.sizeAlongXAxis):
            for y in range(self.sizeAlongYAxis):
                self.standardDev[x, y] = self.stdDevOnePixelWithSamplingWindow((x, y))
        return


    def meanOnePixelWithSamplingWindow(self, pixelCoords: tuple):
        """DOCS 
        """ 
        valuesOfPixelsInSamplingWindow = self.viewAllPixelsInSamplingWindow(pixelCoords, False) 
        return np.mean(valuesOfPixelsInSamplingWindow)


    def meanOfWholeImage(self) -> np.ndarray:
        """DOCS
        """
        for x in range(self.sizeAlongXAxis):
            for y in range(self.sizeAlongYAxis):
                self.mean[x, y] = self.meanOnePixelWithSamplingWindow((x, y))
        return 


    def showImageInRealSpace(self) -> None:
        """DOCS
        """
        xAxis, yAxis = range(self.sizeAlongXAxis), range(self.sizeAlongYAxis)
        xAxisGrid, yAxisGrid = np.meshgrid(xAxis, yAxis)
        fig, ax = plt.subplots()
        c = ax.pcolormesh(
            xAxisGrid, 
            yAxisGrid, 
            self.data, 
            cmap = "gray",
            vmin = np.min(self.data), 
            vmax = np.max(self.data)
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(c, ax = ax, label = "Intensity")
        plt.show()
        return
    

def contrastCalculation(
        uniformImage: ImageForHiLo, 
        speckleImage: ImageForHiLo, 
        bandpassFilter: FiltersForHiLo
    ) -> np.ndarray:
    """DOCS
    """
    differenceImage = createDifferenceImage(speckleImage, uniformImage)

    differenceImageWithDefocusIncrease = ImageForHiLo(
        globalConstants["windowValue"],
        imageArray = differenceImage.applyFilter(bandpassFilter)
    )

    differenceImageWithDefocusIncrease.stdDevOfWholeImage()
    speckleImage.meanOfWholeImage()

    noiseInducedBias = np.zeros(
        shape = (differenceImageWithDefocusIncrease.sizeAlongXAxis, differenceImageWithDefocusIncrease.sizeAlongYAxis)
    )

    # For now not working
    # noiseInducedBias = noiseInducedBiasComputation(
    #     speckleImage, 
    #     uniformImage, 
    #     doubleGaussianFilter
    # )

    constrastFunctionSquared = (
        (differenceImageWithDefocusIncrease.standardDev**2 - noiseInducedBias) / 
        np.mean(speckleImage.data)**2
    )

    return np.sqrt(constrastFunctionSquared)


def rescaleImage(image: np.ndarray) -> np.ndarray:
    """DOCS
    """
    minValue = np.min(image)
    if minValue < 0:
        return image + abs(minValue)
    else:
        return image   


def createImageFromPath(imagePath: str) -> np.ndarray:
    """
    This function is used to read an image and output the data related to that image. If the image as negative values, 
    it will shift the image to all positive values.

    Parameters
    ----------
    imagePath: string
        Path of the image to read and get the pixel values from.
    
    Returns
    ----------
    imageAsArray: np.ndarray
        The image pixel values as a numpy 2 dimensionnal array.
    """
    imageAsArray = tf.imread(imagePath)

    return rescaleImage(imageAsArray)


def createDifferenceImage(imageBeingSubstracted: ImageForHiLo, imageToSubstract: ImageForHiLo) -> ImageForHiLo:
    """DOCS
    """
    substractedImageData = rescaleImage(imageBeingSubstracted.data - imageToSubstract.data)
    return ImageForHiLo(globalConstants["windowValue"], imageArray = substractedImageData)


def noiseInducedBiasComputation(
        speckleImage: ImageForHiLo, 
        uniformImage: ImageForHiLo, 
        bandpassFilter: FiltersForHiLo
    ) -> np.ndarray:
    """DOCS
    """
    speckleImage.meanOfWholeImage()
    uniformImage.meanOfWholeImage()
    print(speckleImage.mean)
    print(speckleImage.mean)

    filterIntegration = integrate2DArray(
        bandpassFilter.kSpaceX[0], 
        bandpassFilter.kSpaceY.T[0], 
        np.abs(bandpassFilter.doubleGaussianFilter())**2
    )

    sizeX, sizeY = speckleImage.sizeAlongXAxis, speckleImage.sizeAlongYAxis

    noiseBias = np.zeros(shape = (sizeX, sizeY))
    for x in range(sizeX):
        for y in range(sizeY):
            noiseBias[x, y] = (
                    (
                        (
                            (globalConstants["cameraGain"] * uniformImage.mean[x, y]) 
                            + (globalConstants["cameraGain"] * speckleImage.mean[x, y]) 
                            + globalConstants["readoutNoiseVariance"]
                    ) * filterIntegration
                )
            )
    return noiseBias


def integrate2DArray(xDomain: np.ndarray, yDomain: np.ndarray, theArray: np.ndarray) -> float:
    """DOCS
    """
    firstDomainIntegration = simpson(theArray, yDomain)
    secondDomainIntegration = simpson(firstDomainIntegration, xDomain)
    return secondDomainIntegration


def cameraOTF(sizeAlongXAxis: int, sizeAlongYAxis: int) -> np.ndarray:
    """DOCS
    """
    pixelValues = np.zeros(shape = (sizeAlongXAxis, sizeAlongYAxis))
    for x in range(sizeAlongXAxis):
        for y in range(sizeAlongYAxis):
            newX = (2 * x - sizeAlongXAxis) * np.pi / sizeAlongXAxis
            newY = (2 * y - sizeAlongYAxis) * np.pi / sizeAlongYAxis
            if newX != 0 and newY != 0:
                pixelValues[x, y] = (
                    (math.sin(newX) * math.sin(newY)) / (newX * newY)
                )
            elif newX == 0 and newY != 0: 
                pixelValues[x, y] = math.sin(newY) / newY
            elif newX != 0 and newY == 0:
                pixelValues[x, y] = math.sin(newX) / newX
            elif newX == 0 and newY == 0:
                pixelValues[x, y] = 1
            if pixelValues[x, y] < 0:
                pixelValues[x, y] = 0
    return pixelValues


def imagingOTF(sizeAlongXAxis: int, sizeAlongYAxis: int, numAperture: float, wavelength: float) -> np.ndarray:
    """DOCS
    """
    bandwidth = 2 * numAperture / (wavelength * 1e-9)
    scaleUnits = globalConstants["magnification"] / (globalConstants["pixelSize"] * 1e-6) / bandwidth
    pixelValues = np.zeros(shape = (sizeAlongXAxis, sizeAlongYAxis))
    for x in range(sizeAlongXAxis):
        for y in range(sizeAlongYAxis):
            pixelValues[x, y] = scaleUnits * math.sqrt(
                (
                    (
                        (
                            (x + 0.5 - sizeAlongXAxis / 2)**2 
                            / (sizeAlongXAxis / 2 - 1)**2
                        ) 
                        + 
                        (
                            (y + 0.5 - sizeAlongYAxis / 2)**2 
                            / (sizeAlongYAxis / 2 - 1)**2
                        )
                    )
                )
            )
            if pixelValues[x, y] > 1:
                pixelValues[x, y] = 1
            pixelValues[x, y] = 0.6366197723675814 * (
                (
                    math.acos(pixelValues[x, y]) - pixelValues[x, y] 
                    * math.sqrt(1 - pixelValues[x, y] * pixelValues[x, y])
                )
            )
            if pixelValues[x, y] < 0:
                pixelValues[x, y] = 0
    return pixelValues


def estimateEta(bandpassFilter: FiltersForHiLo) -> float:
    """DOCS
    """
    camOTF = cameraOTF(bandpassFilter.sizeAlongKSpaceX, bandpassFilter.sizeAlongKSpaceY)
    illuminationOTF = imagingOTF(
        bandpassFilter.sizeAlongKSpaceX, 
        bandpassFilter.sizeAlongKSpaceY,
        globalConstants["illuminationAperture"],
        globalConstants["illuminationWavelength"]
    )
    detectionOTF = imagingOTF(
        bandpassFilter.sizeAlongKSpaceX,
        bandpassFilter.sizeAlongKSpaceY,
        globalConstants["detectionAperture"],
        globalConstants["detectionWavelength"]
    )
    bandpassFilterData = bandpassFilter.data
    fudgeFactor = 1.2
    numerator = 0
    denominator = 0
    for x in range(bandpassFilter.sizeAlongKSpaceX):
        for y in range(bandpassFilter.sizeAlongKSpaceY):
            denominator += (
                (bandpassFilterData[x, y] * detectionOTF[x, y] * camOTF[x, y])**2 * abs(illuminationOTF[x, y])
            )
            numerator += illuminationOTF[x, y]

    eta = math.sqrt(numerator / denominator) * fudgeFactor
    return eta


def computeLoImageOfHiLo(
        uniformImage: ImageForHiLo, 
        speckleImage: ImageForHiLo, 
        bandpassFilter: FiltersForHiLo
    ) -> ImageForHiLo:
    """DOCS
    """
    contrastFunction = contrastCalculation(uniformImage, speckleImage, bandpassFilter)
    contrastTimesUniform = ImageForHiLo(
        globalConstants["windowValue"], 
        imageArray = contrastFunction * uniformImage.data
    )
    lowpassFilter = FiltersForHiLo((sizeXForFilter, sizeYForFilter), "lowpass", globalConstants["sigma"])
    loImage = contrastTimesUniform.applyFilter(lowpassFilter)
    return ImageForHiLo(globalConstants["windowValue"], imageArray = loImage)


def computeHiImageOfHiLo(uniformImage: ImageForHiLo) -> ImageForHiLo:
    """DOCS
    """
    highpassFilter = FiltersForHiLo((sizeXForFilter, sizeYForFilter), "highpass", globalConstants["sigma"])
    hiImage = uniformImage.applyFilter(highpassFilter)
    return ImageForHiLo(globalConstants["windowValue"], imageArray = hiImage)


def checkIfSizeMatch(uniformImage: ImageForHiLo, speckleImage: ImageForHiLo) -> tuple:
    """DOCS
    """
    if (
        (uniformImage.sizeAlongXAxis != speckleImage.sizeAlongXAxis) or 
        (uniformImage.sizeAlongYAxis != uniformImage.sizeAlongYAxis)
    ):
        raise Exception("The size of the the speckle image and the uniform image do not match!")
    else:
        return uniformImage.sizeAlongXAxis, uniformImage.sizeAlongYAxis


def checkIfLengthMatch(allUniformPath: list, allSpecklePath: list) -> None:
    """DOCS
    """
    if len(allUniformPath) != len(allSpecklePath):
        raise Exception("The number of uniform and speckle images is not the same!")
    return


def createHiLoImage(uniformImage: ImageForHiLo, speckleImage: ImageForHiLo) -> ImageForHiLo:
    """DOCS
    """
    # Step to compute eta 
    bandpassFilter = FiltersForHiLo((sizeXForFilter, sizeYForFilter), "doublegauss", globalConstants["sigma"])
    eta = estimateEta(bandpassFilter)
    print("Eta estimated!")

    loImage = computeLoImageOfHiLo(uniformImage, speckleImage, bandpassFilter)
    print("Lo image done!")

    # Step to compute the Hi image of HiLo
    highpassFilter = FiltersForHiLo((sizeXForFilter, sizeYForFilter), "highpass", globalConstants["sigma"])
    hiImage = uniformImage.applyFilter(highpassFilter)
    print("Hi image done")

    # Build the HiLo final image
    hiLoImage = ImageForHiLo(
        globalConstants["windowValue"], 
        imageArray = eta * np.abs(loImage.data) + np.abs(hiImage.data)
    )
    print("HiLo image done!")
    return hiLoImage


def sendMultiprocessingUnits(functionToSplit, paramsToIterateOver: list) -> np.ndarray:
    """DOCS
    """
    processes = mp.Pool()
    resultingArray = processes.starmap(functionToSplit, paramsToIterateOver)
    processes.close()
    processes.join()
    return resultingArray


def sortListOfImageFiles(imageFilePath: str) -> int:
    """DOCS
    """
    return int(imageFilePath.split("TimeLapse_")[1].split("-")[0])


def readFilesInDirectory(directoryToRead: str) -> list:
    """DOCS
    """
    allFilesInDirectory = [files for files in glob.glob(directoryToRead + "*.tif")]
    return sorted(allFilesInDirectory, key = sortListOfImageFiles)


def obtainUniformAndSpecklesFromDir(uniformDirectory: str, speckleDirectory: str) -> tuple:
    """DOCS
    """
    uniformFiles, speckleFiles = readFilesInDirectory(uniformDirectory), readFilesInDirectory(speckleDirectory)
    return uniformFiles, speckleFiles


def buildUniformAndSpeckleObjects(allUniformPath: list, allSpecklePath: list) -> list:
    """DOCS
    """
    checkIfLengthMatch(allUniformPath, allSpecklePath)
    allImageForHiLo = [
            (
                ImageForHiLo(imagePath = unif, samplingWindow = globalConstants["windowValue"]), 
                ImageForHiLo(imagePath = speck, samplingWindow = globalConstants["windowValue"])
            )
            for unif, speck in zip(allUniformPath, allSpecklePath)
    ]
    sizeXForFilter, sizeYForFilter = checkIfSizeMatch(allImageForHiLo[0][0], allImageForHiLo[0][1])
    return allImageForHiLo, sizeXForFilter, sizeYForFilter


def buildHiLoImagesDirectory(principalDirectory: str, allHiLoImages: list) -> None:
    """DOCS
    """
    hiLoImagesData = [(image.data).astype(np.int16) for image in allHiLoImages]
    os.chdir(mainDirectory)
    return tf.imwrite("finalHiLoImages.tif", hiLoImagesData)


if __name__ == "__main__":  
    # Constant parameters
    globalConstants = {    
        "cameraGain" : 1,
        "readoutNoiseVariance" : 0.3508935,
        "sigma" : 2,
        "waveletGaussiansRatio" : 2,
        "windowValue" : 3,
        "illuminationAperture" : 1,
        "detectionAperture" : 1,
        "illuminationWavelength" : 488e-9,
        "detectionWavelength" : 520e-9,
        "pixelSize" : 4.5,
        "magnification" : 20
    }

    mainDirectory = "/Users/dcclab/Desktop/MainDir/"
    allUniformPath, allSpecklePath = obtainUniformAndSpecklesFromDir(
            "/Users/dcclab/Desktop/MainDir/Test_unif/", 
            "/Users/dcclab/Desktop/MainDir/Test_speck/"
    )

    allImageForHiLo, sizeXForFilter, sizeYForFilter = buildUniformAndSpeckleObjects(allUniformPath, allSpecklePath)
    print("All images initialized!")

    print("Let's go!")
    t1 = time.time()
    hiLoImages = sendMultiprocessingUnits(createHiLoImage, allImageForHiLo)
    print(time.time() - t1)

    buildHiLoImagesDirectory(mainDirectory, hiLoImages)

    pass
