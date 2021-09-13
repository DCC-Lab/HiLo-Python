import tifffile as tiff
import scipy.ndimage as simg
import numpy as np
import math as math
import exceptions as exc
import matplotlib.pyplot as plt
import skimage as ski

def createImage(imagepath):
	exc.isTifOrTiff(imagepath)
	exc.isString(imagepath)

	image = tiff.imread(imagepath)
	imageRescale = ski.util.img_as_uint(image)

	return imageRescale

def createDifferenceImage(speckle, uniform):
	exc.isANumpyArray(speckle)
	exc.isANumpyArray(uniform)
	exc.isSameShape(image1=speckle, image2=uniform)

	# Convert speckle and uniform images into np.float64 data type, so that the subtraction can be negative. 
	newSpeckle = speckle.astype(np.float64)
	newUniform = uniform.astype(np.float64)

	# Subtraction
	i1 = 0
	i2 = 0
	typeOfData = type(speckle[0][0])
	difference = np.zeros(shape=(newSpeckle.shape[0], newSpeckle.shape[1]))
	while i1 < newSpeckle.shape[0]:
		while i2 < newSpeckle.shape[1]:
			difference[i1][i2] = newSpeckle[i1][i2] - newUniform[i1][i2]
			i2 += 1
		i2 = 0
		i1 += 1

	# Find the minimal value
	minPixel = np.amin(difference)

	# Rescale the data to only have positive values. 
	i1 = 0
	i2 = 0
	while i1 < difference.shape[0]:
		while i2 < difference.shape[1]:
			difference[i1][i2] = difference[i1][i2] + abs(minPixel)
			i2 += 1
		i2 = 0
		i1 += 1

	# Convert data in uint16
	differenceInInt = difference.astype(np.uint16)

	return differenceInInt

def createOTF(image):
	"""Optical transfer function is the fft of the PSF. The modulation transfer function is the magnitude of the complex OTF."""
	pixels = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint16)
	x = 0
	y = 0
	while x < image.shape[0]:
		while y < image.shape[1]:
			x1 = ((2*x-image.shape[0])*np.pi)/image.shape[0]
			y1 = ((2*y-image.shape[1])*np.pi)/image.shape[1]
			if x1 != 0 and y1 != 0:
				pixel[x][y] = (math.sin(x1)*math.sin(y1))/(x1*y1)
			elif x1 == 0 and y1 != 0: 
				pixel[x][y] = math.sin(y1)/y1
			elif x1 != 0 and y1 == 0:
				pixel[x][y] = math.sin(x1)/x1
			elif x1 != 0 and y1 != 0:
				pixel[x][y] = 1
			if pixel[x][y] < 0:
				pixel[x][y] = 0
			y += 1
		y = 0
		x += 1

	return pixels

def obtainFFTFitler(image, filteredImage):
	exc.isANumpyArray(image)
	exc.isANumpyArray(filteredImage)
	exc.isSameShape(image1=image, image2=filteredImage)

	fftImage = np.fft.fft2(image)
	fftFilteredImage = np.fft.fft2(filteredImage)
	fftFilter = np.divide(fftFilteredImage, fftImage)
	return fftFilter

def gaussianFilter(sigma, image, truncate=4.0):
	exc.isANumpyArray(image)
	exc.isIntOrFloat(sigma)

	imgGaussian = simg.gaussian_filter(image, sigma=sigma, truncate=truncate) 
	return imgGaussian

def valueOnePixel(image, pixelPosition):
	value = image[pixelPosition[0]][pixelPosition[1]]
	if value < 0:
		value = 0
	return value

def valueAllPixelsInImage(image):
	position = [0,0]
	values = []
	while position[0] < image.shape[0]:
		while position[1] < image.shape[1]:
			values.append(valueOnePixel(image=image, pixelPosition=position))
			position[1] += 1
				
		position[0] = position[0] + 1

	return values

def valueAllPixelsInSW(image, px, samplingWindow):
	n = int(samplingWindow/2)
	positionInSW = [px[0]-n, px[1]-n]

	values = []
	while positionInSW[0] <= px[0]+n and positionInSW[0] < image.shape[0]:
		if positionInSW[0] < 0:
			positionInSW[0] = 0
		while positionInSW[1] <= px[1]+n and positionInSW[1] < image.shape[1]:
			if positionInSW[1] < 0:  
				positionInSW[1] = 0
			values.append(valueOnePixel(image=image, pixelPosition=positionInSW))
			positionInSW[1] += 1
				
		positionInSW[1] = px[1] - n
		positionInSW[0] = positionInSW[0] + 1

	return values

def absSumAllPixels(array):
	i1 = 0
	i2 = 0
	if type(array) is np.ndarray:
		while i1 < array.shape[0]:
			while i2 < array.shape[1]: 
				if type(array[i1][i2]) is np.complex128:
					array[i1][i2] = np.absolute(array[i1][i2])
				else : 
					array[i1][i2] = abs(array[i1][i2])
				i2 += 1
			i2 = 0
			i1 += 1 
		sumValue = np.sum(array)

	elif type(array) is list:
		for i in range(len(array)):
			if type(array[i]) is np.complex128:
				array[i] = np.absolute(array[i])
			else : 
				array[i] = abs(array[i])
		sumValue = sum(array)

	else:
		if type(array) is np.complex128:
			array = np.absolute(array)
		else : 
			array = abs(array)
		sumValue = sum(array)

	return sumValue.real

def squaredFunction(array):
	if type(array) is np.ndarray:
		i1 = 0
		i2 = 0
		while i1 < array.shape[0] : 
			while i2 < array.shape[1]:
				value = array[i1][i2]
				array[i1][i2] = value**2
				i2 += 1
			i2 = 0
			i1 += 1
	elif type(array) is list:
		for element in array:
			array[element] = array[element]**2
	else:
		array = array**2

	return array

def stDevAndMeanOnePixel(image, sw, pixel): 
	exc.isANumpyArray(image)
	exc.isInt(sw)

	if type(pixel) is not list or len(pixel) != 2:
		raise Exception("pixel must be a list of two int.")
	 
	valuesOfPixelInSamplingWindow = valueAllPixelsInSW(image=image, px=pixel, samplingWindow=sw)
	std, mean = np.std(valuesOfPixelInSamplingWindow), np.mean(valuesOfPixelInSamplingWindow) 

	return std, mean

def stdevAndMeanWholeImage(image, samplingWindow):
	exc.isANumpyArray(image)
	exc.isInt(samplingWindow)

	stDevImage = np.zeros(shape=(image.shape[0], image.shape[1]))
	meanImage = np.zeros(shape=(image.shape[0], image.shape[1]))
	
	pixelPosition = [0,0]
	while pixelPosition[0] < image.shape[0]:
		stdevArray = []
		meanArray = []
		while pixelPosition[1] < image.shape[1]:
			# Calculations of one pixel
			stdev, mean = stDevAndMeanOnePixel(image=image, sw=samplingWindow, pixel=pixelPosition)
			stdevArray.append(stdev)
			meanArray.append(mean)
			pixelPosition[1] = pixelPosition[1] + 1

		stDevImage[pixelPosition[0]] = stdevArray
		meanImage[pixelPosition[0]] = meanArray
		pixelPosition[1] = 0
		pixelPosition[0] = pixelPosition[0] + 1

	return stDevImage, meanImage

def noiseInducedBias(cameraGain, readoutNoiseVariance, imageSpeckle, imageUniform, difference, samplingWindowSize, sigma):
	exc.isIntOrFloat(cameraGain)
	exc.isIntOrFloat(readoutNoiseVariance)
	exc.isANumpyArray(imageSpeckle)
	exc.isANumpyArray(imageUniform)
	exc.isInt(samplingWindowSize)

	stdevSpeckle, meanSpeckle = stdevAndMeanWholeImage(image=imageSpeckle, samplingWindow=samplingWindowSize)
	stdevUniform, meanUniform = stdevAndMeanWholeImage(image=imageUniform, samplingWindow=samplingWindowSize)
	# print("MEAN S : {}{}".format(meanSpeckle, meanSpeckle.dtype))
	# print("STDEV S : {}{}".format(stdevSpeckle, stdevSpeckle.dtype))
	# print("MEAN U : {}{}".format(meanUniform, meanUniform.dtype))
	# print("STDEV U : {}{}".format(stdevUniform, stdevUniform.dtype))

	fftFilter = obtainFFTFitler(image=imageUniform, filteredImage=difference)
	sumAbsfftFilter = absSumAllPixels(array=fftFilter)
	squaredSumAbsfftFilter = squaredFunction(array=sumAbsfftFilter)
	print("SQUARED : {}".format(squaredSumAbsfftFilter))
	# Calculate the noise-induced biased function. 
	i1 = 0
	i2 = 0
	bias = np.zeros(shape=(imageSpeckle.shape[0], imageSpeckle.shape[1]))
	while i1 < meanUniform.shape[0]:
		noiseArray = []
		while i2 < meanUniform.shape[1]:
			noiseArray.append(((np.sqrt(cameraGain*meanUniform[i1][i2]) + (cameraGain*meanSpeckle[i1][i2]) + readoutNoiseVariance) * squaredSumAbsfftFilter))
			i2 += 1
		bias[i1] = noiseArray	
		i2 = 0
		i1 += 1
	
	print("BIAS : {}{}".format(bias, type(bias[0][0])))
	return bias

def noiseInducedBiasReductionFromStdev(noise, stdev):
	i1 = 0
	i2 = 0
	while i1 < noise.shape[0]:
		while i2 < noise.shape[1]:
			stdev[i1][i2] = stdev[i1][i2] - noise[i1][i2]
			i2 += 1
		i2 = 0
		i1 += 1

	return stdev

def contrastCalculation(uniform, speckle, samplingWindow, sigma):
	exc.isANumpyArray(speckle)
	exc.isANumpyArray(uniform)
	exc.isSameShape(image1=uniform, image2=speckle)
	exc.isInt(samplingWindow)

	# create difference image and apply a gaussian filter on it
	differenceImage = createDifferenceImage(speckle=speckle, uniform=uniform)
	gaussianImage = gaussianFilter(sigma=sigma, image=differenceImage)

	# calculate the noise-induced bias of the image
	noiseFunction = noiseInducedBias(cameraGain=1, readoutNoiseVariance=0.3508935, imageSpeckle=speckle, imageUniform=uniform, difference=gaussianImage, samplingWindowSize=samplingWindow, sigma=sigma)

	# calculate the stdev and the mean of the speckle and the uniform image
	stdevDifference, meanDifference = stdevAndMeanWholeImage(image=gaussianImage, samplingWindow=samplingWindow)
	stdevUniform, meanUniform = stdevAndMeanWholeImage(image=uniform, samplingWindow=samplingWindow)
	print("STDEV DIFFERENCE : {}{}".format(stdevDifference, stdevDifference.dtype))

	# subtract the noise-induced bias from the stdev of the filtered difference image
	reducedStdevDifference = noiseInducedBiasReductionFromStdev(noise=noiseFunction, stdev=stdevDifference)
	print("REDUCED : {}{}".format(reducedStdevDifference, reducedStdevDifference.dtype))

	# calculate the contrast function
	i1 = 0
	i2 = 0
	contrastFunction = np.zeros(shape=(uniform.shape[0], uniform.shape[1]))
	while i1 < stdevDifference.shape[0]:
		while i2 < stdevDifference.shape[1]:
			contrastFunction[i1][i2] = reducedStdevDifference[i1][i2]/meanUniform[i1][i2]
			i2 += 1
		i2 = 0
		i1 += 1

	print("CONTRAST FUNCTION : {}{}".format(contrastFunction, contrastFunction.dtype))
	return contrastFunction

def highPassFilter(low):
	exc.isANumpyArray(low)

	hi = 1 - low
	return hi





