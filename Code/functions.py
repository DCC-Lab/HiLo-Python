import tifffile as tiff
import scipy.ndimage as simg
import numpy as np
import math as math
import exceptions as exc
import cmath as cmath
import time
from rich import print
import matplotlib.pyplot as plt


def createImage(imagePath: str) -> np.ndarray:
    """
    This function is used to read an image and output the data related to that image. If the image as negative values,
    it will shift the image to all positif values.

    Parameters
    ----------
    imagePath: string
        Path of the image to read and get the pixel values from.
    
    Returns
    ----------
    imageAsArray: np.ndarray
        The image pixel values as a numpy 2 dimensionnal array.
    """
    exc.isTifOrTiff(imagePath)
    exc.isString(imagePath)

    imageAsArray = tiff.imread(imagePath)

    if np.min(imageAsArray) < 0:
        return imageAsArray + abs(np.min(imageAsArray))
    else:
        return imageAsArray


def createDifferenceImage(speckle, uniform):
	"""
	This function does image_speckle-image_unform to have an image that has the speckle pattern on the object without 
    the object. The resulting image is called the difference image. 
	"""
	exc.isANumpyArray(speckle)
	exc.isANumpyArray(uniform)
	exc.isSameShape(image1=speckle, image2=uniform)

	# Convert speckle and uniform images into np.float64 data type, so that the subtraction can be negative. 
	newSpeckle = speckle.astype(np.float64)
	newUniform = uniform.astype(np.float64)

	# Subtraction
	difference = newSpeckle - newUniform

	# Find the minimal value
	minPixel = np.min(difference)

	# Rescale the data to only have positive values. 
	difference = difference + abs(minPixel)

	# Convert data in uint16
	differenceInInt = difference.astype(np.uint16)

	return differenceInInt


def cameraOTF(image):
    """
    Optical transfer function is the fft of the PSF. The modulation transfer function is the magnitude of the
    complex OTF.
    """
    sizex, sizey = image.shape[0], image.shape[1]
    pixels = np.zeros(shape=(sizex, sizey), dtype=np.float64)

    for x in range(sizex):
        for y in range(sizey):
            x1 = (2 * x - sizex) * np.pi / sizex
            y1 = (2 * y - sizey) * np.pi / sizey
            if x1 != 0 and y1 != 0:
                pixels[x, y] = (math.sin(x1) * math.sin(y1)) / (x1 * y1)
            elif x1 == 0 and y1 != 0: 
                pixels[x, y] = math.sin(y1) / y1
            elif x1 != 0 and y1 == 0:
                pixels[x, y] = math.sin(x1) / x1
            elif x1 == 0 and y1 == 0:
                pixels[x, y] = 1
            if pixels[x, y] < 0:
                pixels[x, y] = 0
    return pixels


def imagingOTF(numAperture, wavelength, magnification, pixelSize, image):
    sizex, sizey = image.shape[0], image.shape[1]
    bandwidth = 2 * numAperture / (wavelength * 1e-9)
    scaleUnits = magnification / (pixelSize * 1e-6) / bandwidth
    pixel = np.zeros(shape=(sizex, sizey), dtype=np.float64)

    for x in range(sizex):
        for y in range(sizey):
            pixel[x, y] = scaleUnits * math.sqrt(
                    ((x + 0.5 - sizex / 2)**2 / (sizex / 2 - 1)**2) + ((y + 0.5 - sizey / 2)**2 / (sizey / 2 - 1)**2)
            )
            if pixel[x, y] > 1:
                pixel[x, y] = 1
            pixel[x, y] = 0.6366197723675814 * (
                    math.acos(pixel[x, y]) - pixel[x, y] * math.sqrt(1 - pixel[x, y] * pixel[x, y])
            )
            if pixel[x, y] < 0:
                pixel[x, y] = 0
    return pixel


def obtainFFTFitler(image, filteredImage):
	""""
	returns:
	2D numpy array of complex128 numbers. 
	"""
	exc.isANumpyArray(image)
	exc.isANumpyArray(filteredImage)
	exc.isSameShape(image1=image, image2=filteredImage)

	fftImage = np.fft.fft2(image)
	fftFilteredImage = np.fft.fft2(filteredImage)
	fftFilter = np.divide(fftFilteredImage, fftImage)
	normfftFilter = np.divide(fftFilter, np.amax(fftFilter))/fftFilter.shape[0]/fftFilter.shape[1]

	return normfftFilter


def applyGaussianFilter(sigma, image, truncate=4.0):
    """
    Apply a Gaussian filter on the imput image. The higher the value of sigma, the bigger the filter window, the more 
    intense the filter will be on the image. Another way of saying this is that a grater value of sigma will make the 
    image blurrier. 
    In the HiLo algorithm, we use a gaussian filter to increase the defocus dependence of the PSF, which leads to an 
    improved or "stronger" optical sectioning. Sigma is the parameter that drives the optical sectioning thickness in 
    the HiLo algorithm. 
    """
    exc.isANumpyArray(image)
    exc.isIntOrFloat(sigma)
    imgGaussian = simg.gaussian_filter(image, sigma=sigma, truncate=truncate)
    return imgGaussian


def valueOnePixel(image, pixelPosition):
	"""
	Simply extracts the value at a specific position in the image. The only reason why this is in a function is because 
    we also want to make sure that thing pixel value is not lower than 0. 
	"""
	value = image[pixelPosition[0]][pixelPosition[1]]
	if value < 0:
		value = 0
	return value


def valueAllPixelsInSW(image, pixel, size, absoluteOn):
    """
    Generates a list of the values of all elements in image at the center pixel position px in the area of the sampling 
    window samplingWindow. 
    """   
    n = size // 2
    xpixel, ypixel = pixel[0], pixel[1]
    values = []
    
    for xvalue in range(xpixel - n, xpixel + n + 1):
        for yvalue in range(ypixel - n, ypixel + n + 1):
            if not((xvalue < 0) or (yvalue < 0) or (xvalue > image.shape[0] - 1) or (yvalue > image.shape[1] - 1)):
                if absoluteOn == True:
                    values.append(abs(valueOnePixel(image, (xvalue, yvalue))))
                else:
                    values.append(valueOnePixel(image, (xvalue, yvalue)))
    return values


def absSum(array, samplingWindow): 
    """
    returns:
    numpy.nparray of the absolute sum of the values included in the sampling window samplingW of all pixels/elements in 
    array.
    """
    sizex, sizey = array.shape[0], array.shape[1]
    absSumImage = np.zeros(shape = (sizex, sizey))
    for x in range(sizex):
        for y in range(sizey):
            absSumImage[x, y] = sum(valueAllPixelsInSW(array, (x, y), samplingWindow, True)) 
    return absSumImage


def squared(array):
	"""
	returns:
	The squared value of the list, numpy.ndarray or number. The type of array is kept throughout the execution of the 
    function. 
	"""
	if type(array) is np.ndarray:
		newArray = array**2
	elif type(array) is list:
		newArray = [i**2 for i in array]
	else:
		newArray = array**2

	return newArray


def stDevOnePixel(image, samplingWindow, pixel):
    """ 
    """ 
    exc.isANumpyArray(image)
    exc.isInt(samplingWindow)
    exc.isPixelATuple(pixel)
    
    valuesOfPixelsInSamplingWindow = valueAllPixelsInSW(image, pixel, samplingWindow, False)
    stdDev = np.std(valuesOfPixelsInSamplingWindow) 
    
    return stdDev


def meanOnePixel(image, samplingWindow, pixel):
    """
    """ 
    exc.isANumpyArray(image)
    exc.isInt(samplingWindow)
    exc.isPixelATuple(pixel)
    
    valuesOfPixelsInSamplingWindow = valueAllPixelsInSW(image, pixel, samplingWindow, False)
    meanOfList = np.mean(valuesOfPixelsInSamplingWindow) 
    
    return meanOfList


def stdevWholeImage(image, samplingWindow):
    """
    """
    exc.isANumpyArray(image)
    exc.isInt(samplingWindow)

    sizex, sizey = image.shape[0], image.shape[1]
    stDevImage = np.zeros(shape=(sizex, sizey))
	
    for x in range(sizex):
        for y in range(sizey):
            stDevImage[x, y] = stDevOnePixel(image, samplingWindow, (x, y))
    return stDevImage


def meanWholeImage(image, samplingWindow):
    """
    """
    exc.isANumpyArray(image)
    exc.isInt(samplingWindow)

    sizex, sizey = image.shape[0], image.shape[1]
    meanImage = np.zeros(shape=(sizex, sizey))
	
    for x in range(sizex):
        for y in range(sizey):
            meanImage[x, y] = meanOnePixel(image, samplingWindow, (x, y))
    return meanImage


def noiseInducedBias(
        cameraGain, 
        readoutNoiseVariance, 
        imageSpeckle,
        imageUniform, 
        difference, 
        samplingWindowSize, 
        sigma
    ):
    """
    """
    exc.isIntOrFloat(cameraGain)
    exc.isIntOrFloat(readoutNoiseVariance)
    exc.isANumpyArray(imageSpeckle)
    exc.isANumpyArray(imageUniform)
    exc.isInt(samplingWindowSize)

    meanSpeckle = meanWholeImage(image=imageSpeckle, samplingWindow=samplingWindowSize)
    meanUniform = meanWholeImage(image=imageUniform, samplingWindow=samplingWindowSize)

    fftFilter = obtainFFTFitler(image=imageUniform, filteredImage=difference)
    sumAbsfftFilter = absSum(array=fftFilter, samplingWindow=samplingWindowSize)
    squaredSumAbsfftFilter = sumAbsfftFilter**2

    # Calculate the noise-induced biased function.
    sizex, sizey = imageSpeckle.shape[0], imageSpeckle.shape[1]
    bias = np.zeros(shape=(sizex, sizey))
    for x in range(sizex):
        for y in range(sizey):
            bias[x, y] = (
                    (np.sqrt(cameraGain * meanUniform[x, y]) + (cameraGain * meanSpeckle[x, y]) + readoutNoiseVariance)
                    * squaredSumAbsfftFilter[x, y]
            )
    return bias


def contrastCalculation(uniform, speckle, samplingWindow, sigma):
	"""
	This function generates the contrast weighting function that peaks at 1 for in-focus regions and goes to 0 for 
    out-of-focus regions of the difference image. 
	TODO : noise function calculation doesn't work
	Returns the contrast weighting function that is the 2D size of the raw input images. 
	"""
	exc.isANumpyArray(speckle)
	exc.isANumpyArray(uniform)
	exc.isSameShape(image1=uniform, image2=speckle)
	exc.isInt(samplingWindow)

	# create difference image and apply a gaussian filter on it
	differenceImage = createDifferenceImage(speckle=speckle, uniform=uniform)
	gaussianImage = applyGaussianFilter(sigma=sigma, image=differenceImage)

	# calculate the noise-induced bias of the image

	#TODO : 
	noiseFunction = np.zeros(shape=(uniform.shape[0], uniform.shape[1]))
	# noiseFunction = noiseInducedBias(
    #        cameraGain=1, 
    #        readoutNoiseVariance=0.3508935, 
    #        imageSpeckle=speckle, 
    #        imageUniform=uniform, 
    #        difference=gaussianImage,
    #        samplingWindowSize=samplingWindow, 
    #        sigma=sigma
    # )

	# calculate the stdev and the mean of the speckle and the uniform images
	stdevDifference = stdevWholeImage(image=gaussianImage, samplingWindow=samplingWindow)
	meanUniform = meanWholeImage(image=uniform, samplingWindow=samplingWindow)

	# subtract the noise-induced bias from the stdev of the filtered difference image
	reducedStdevDifference = np.subtract(stdevDifference, noiseFunction)

	# calculate the contrast function
	contrastFunction = reducedStdevDifference / meanUniform

	return contrastFunction


def lowpassFilter(image, sigmaFilter):
	"""
	Creates a 2D numpy array the size used as a low-pass Fourier filter.
	Sigma is used again in this function to evaluate the size of the center circle, aka the frequencies to get rid of. 
	"""
	exc.isANumpyArray(image)
	exc.isIntOrFloat(sigmaFilter)

	x, y = np.meshgrid(np.linspace(-1, 1, image.shape[0]), np.linspace(-1, 1, image.shape[1]))
	d = np.sqrt(x**2 + y**2)
	sigma = (sigmaFilter * 0.18) / (2 * math.sqrt(2 * math.log(2)))
	mu = 0.0
	gauss = (1 / (sigma * 2 * np.pi)) * np.exp(-((d - mu)**2 / (2.0 * sigma**2)))
	maxPixel = np.max(gauss)
	gaussNorm = gauss / maxPixel

	return gaussNorm


def highpassFilter(low):
	"""
	The high-pass Fourier filter is produced from the low-pass Fourier filter. It is litteraly its inverse. 
	"""
	exc.isANumpyArray(low)
	return 1 - low


def estimateEta(speckle, uniform, sigma, ffthi, fftlo, sig):
    """
    TODO : It is not clear how we should evaluate the eta.
    The LO image is always less intense than the HI image after treatments, so we need a scaling function eta to 
    readjust the intensities of the LO image. This scaling function eta compensates for the fact that the contrast 
    weighting function never reaches 1 even for the perfectly in-focus regions. It also prevents the discontinuities at 
    the cutoff frequency.
    Returns the function eta in numpy.ndarray. Calculations taken from the java code for the HiLo Fiji plugin and from 
    personal communication with Olivier Dupont-Therrien at Bliq Photonics
    """
    differenceImage = createDifferenceImage(speckle=speckle, uniform=uniform)
    gaussianImage = applyGaussianFilter(sigma=sigma, image=differenceImage)
    bandpassFilter = obtainFFTFitler(image=uniform, filteredImage=gaussianImage)

    illuminationOTF = imagingOTF(numAperture=1, wavelength=488e-9, magnification=20, pixelSize=6.5, image=uniform)
    detectionOTF = imagingOTF(numAperture=1, wavelength=520e-9, magnification=20, pixelSize=6.5, image=uniform)
    camOTF = cameraOTF(image=uniform)
    # print(f"Ill OTF : {illuminationOTF}{illuminationOTF.dtype}")
    # print(f"DET OTF {detectionOTF}{detectionOTF.dtype}")
	
    # Method 1 : Generate a function eta for each pixels.
    # eta = np.zeros(shape=(uniform.shape[0], uniform.shape[1]), dtype=np.complex128) 
    # x = 0
    # y = 0
    # while x<camOTF.shape[0]:
    #     etaList = []
    #     while y<camOTF.shape[1]:
    #         denominator = (bandpassFilter[x][y] * detectionOTF[x][y] * camOTF[x][y])**2 * np.absolute(illuminationOTF[x][y])
    #         numerator = illuminationOTF[x][y]
    #         if denominator == 0 or numerator == 0:
    #             result = 0
    #         else:
    #             result = cmath.sqrt(numerator / denominator)
    #         etaList.append(result)
    #         y += 1
    #     eta[x] = etaList
    #     y = 0
    #     x += 1

    # Method 2 : Generate one value for the whole image. 
    # numerator = 0
    # denominator = 0
    # x = 0
    # y = 0
    # while x<camOTF.shape[0]:
    #     while y<camOTF.shape[1]:
    #         firstStep = (bandpassFilter[x][y] * detectionOTF[x][y] * camOTF[x][y])**2
    #         secondStep = np.absolute(illuminationOTF[x][y])
    #         denominator += (bandpassFilter[x][y] * detectionOTF[x][y] * camOTF[x][y])**2 * np.absolute(illuminationOTF[x][y])
    #         numerator += illuminationOTF[x][y]
    #         y += 1
    #     y = 0
    #     x += 1
    # eta = cmath.sqrt(numerator / denominator) * 1.2

    #Method 3 : eta is obtained experimentally from the HI and the LO images comes
    # Comes from the articles Daryl Lim et al. 2008
    numerator = 0
    denominator = 0
    x = 0
    y = 0
    while x<ffthi.shape[0]:
        while y<ffthi.shape[1]:
            numerator += np.absolute(ffthi[x][y])
            denominator += np.absolute(fftlo[x][y])
            y += 1
        y = 0
        x += 1
    eta = numerator / denominator

    #Method 4 : Eta is obtained experimentally from the HI and the LO images at the cutoff frequency
    numerator = 0
    denominator = 0
    x = 0
    y = 0
    d = 0
    elementPosition = []
    cutoffhi = np.std(ffthi)*0.01
    cutoff = 0.18*sig

    while x<ffthi.shape[0]:
        while y<ffthi.shape[1]:
            if abs(cutoff-ffthi[x][y].real) < cutoffhi:
                #print(f"I'm adding this to numerator : {ffthi[x][y]}")
                numerator += math.sqrt(ffthi[x][y].real**2 + ffthi[x][y].imag**2)
                elementPosition.append([x,y])
            y += 1
        y = 0
        x += 1

    # print(len(elementPosition))
    for i in elementPosition:
        # print(i)
        denominator += math.sqrt(fftlo[i[0]][i[1]].real**2 + fftlo[i[0]][i[1]].imag**2)
        d += 1

    # print(f"N : {len(elementPosition)}")
    # print(f"D : {d}")
    # print(f"Num : {numerator}")
    # print(f"Den : {denominator}")
    # eta = numerator / denominator



    # Tried to normalize eta at some point to see what happened. Doesn't work. 
    #normEta = np.zeros(shape=(uniform.shape[0], uniform.shape[1]), dtype=np.float64)
    #x = 0
    #y = 0
    #while x < eta.shape[0]:
    #	normEtaList = []
    #	while y < eta.shape[1]:
    #		normEtaValue = eta[x][y]/np.absolute(eta[x][y])
    #		normEtaList.append(normEtaValue)
    #		y += 1
    #	normEta[x] = normEtaList
    #	y = 0
    #	x += 1

    print(f"ETA : {eta} + {type(eta)}")	

    return eta


def createHiLoImage(uniform, speckle, sigma, sWindow):

    # calculate the contrast weighting function
    contrast = contrastCalculation(uniform=uniform, speckle=speckle, samplingWindow=sWindow, sigma=sigma)
    print("Constrast done")

    # Create the filters
    lowFilter = lowpassFilter(image=uniform, sigmaFilter=sigma)
    highFilter = highpassFilter(low=lowFilter)
    print("Filters done")

    # Create fft of uniform image and normalize with max value
    fftuniform = np.fft.fftshift(np.fft.fft2(uniform))
    # normfftuniform = fftuniform/np.amax(fftuniform)

    # Apply the contrast weighting function on the uniform image. FFT of the result and then normalize with max value.
    cxu = contrast*uniform
    fftcxu = np.fft.fftshift(np.fft.fft2(cxu))
    print("Contrast applied")

    # Apply the low-pass frequency filter on the uniform image to create the LO portion
    # Ilp = LP[C*Iu]
    fftLO = lowFilter*fftcxu
    LO = np.fft.ifft2(np.fft.ifftshift(fftLO))
    print("LO created")

    # Apply the high-pass frequency filter to the uniform image to obtain the HI portion
    # Ihp = HP[Iu]
    fftHI = highFilter*fftuniform
    HI = np.fft.ifft2(np.fft.ifftshift(fftHI))
    print("HI created")

    # TODO: 
    # Estimate the function eta for scaling the frequencies of the low image. Generates a complex number. 
    eta = estimateEta(speckle=speckle, uniform=uniform, sigma=sigma, fftlo=fftLO, ffthi=fftHI, sig=sigma)
    print("Eta done")

    complexHiLo = eta * LO + HI

    # convert the complexHiLo image to obtain the modulus of each values. 
    HiLo = np.abs(complexHiLo)
    print("All done")

    return HiLo


def showDataSet(xArray, yArray, zArray):
    """DOCS
    """
    fig, ax = plt.subplots()
    c = ax.pcolormesh(xArray, yArray, zArray, cmap = "viridis_r", vmin = np.min(zArray), vmax = np.max(zArray))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(c, ax = ax, label = "z")
    plt.show()
    return


if __name__ == "__main__":  
    pass
