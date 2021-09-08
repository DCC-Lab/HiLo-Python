import tifffile as tiff
import scipy.ndimage as simg
import numpy as np
import math as math
import exceptions as exc

def image(imagepath):
	exc.isTifOrTiff(imagepath)
	exc.isString(imagepath)
	image = tiff.imread(imagepath) 
	return image

def differenceImage(speckle, uniform):
	exc.isANumpyArray(speckle)
	exc.isANumpyArray(uniform)
	exc.isSameShape(image1=speckle, image2=uniform)

	# Subtraction
	i1 = 0
	i2 = 0
	typeOfData = type(speckle[0][0])
	difference = np.zeros(shape=(speckle.shape[0], speckle.shape[1]), dtype=np.int16)
	while i1 < speckle.shape[0]:
		while i2 < speckle.shape[1]:
			difference[i1][i2] = speckle[i1][i2] - uniform[i1][i2]
			i2 += 1
		i2 = 0
		i1 += 1

	# Find the minimal value
	minPixel = np.amin(difference)

	# Rescale the data
	i1 = 0
	i2 = 0
	while i1 < difference.shape[0]:
		while i2 < difference.shape[1]:
			difference[i1][i2] = difference[i1][i2] + abs(minPixel)
			i2 += 1
		i2 = 0
		i1 += 1

	return difference

def gaussianImage(image, sigmaFilter, hilo=True):
	exc.isANumpyArray(image)
	exc.isIntOrFloat(sigmaFilter)

	# hilo is True by default. Only when the image produced is the LO filter. If this function is used to produce the convolution window of the gaussain filter (or any other filter), hilo=False.
	if hilo is True: 
		sigma = (sigmaFilter*0.18)/(2*math.sqrt(2*math.log(2)))
	else:
		sigma = sigmaFilter

	x, y = np.meshgrid(np.linspace(-1,1,image.shape[0], dtype=np.int8), np.linspace(-1,1,image.shape[1], dtype=np.int8))
	d = np.sqrt(x*x+y*y)
	mu = 0.0
	gauss = (1/(sigma*2*np.pi)) * np.exp(-((d-mu)**2/(2.0*sigma**2)))
	print("GAUSS : {}".format(gauss))

	return gauss

def gaussianFilter(sigma, image, truncate=4.0):
	exc.isANumpyArray(image)
	exc.isIntOrFloat(sigma)

	imgGaussian = simg.gaussian_filter(image, sigma=sigma, truncate=truncate) 
	# POURQUOI C'EST PAS DE LA TAILLE QUE JE VEUX rrrr
	windowSize = 2*int(truncate*sigma + 0.5) + 1
	print("window size : {}".format(windowSize))
	window = np.zeros(shape=(windowSize, windowSize), dtype=np.int16)
	convolutionWindow = gaussianImage(image=window, sigmaFilter=sigma, hilo=False)

	return imgGaussian, convolutionWindow

# def gaussianFilterOfImage(filteredImage, differenceImage):
# 	exc.isANumpyArray(filteredImage)
# 	exc.isANumpyArray(differenceImage)
# 	exc.isSameShape(image1=filteredImage, image2=differenceImage)

# 	bandpassFilter = filteredImage - differenceImage
# 	minValue = np.amin(bandpassFilter)

# 	i1 = 0
# 	i2 = 0
# 	while i1 < filteredImage.shape[0]:
# 		while i2 < filteredImage.shape[1]:
# 			bandpassFilter[i1][i2] = bandpassFilter[i1][i2] + abs(minValue)
# 			i2 += 1
# 		i2 = 0
# 		i1 += 1

	# return bandpassFilter

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

def absSumAllPixels(function):
	i1 = 0
	i2 = 0
	if type(function) is np.ndarray:
		while i1 < function.shape[0]:
			while i2 < function.shape[1]: 
				if type(function[i1][i2]) is np.complex128:
					function[i1][i2] = np.absolute(function[i1][i2])
				else : 
					function[i1][i2] = abs(function[i1][i2])
				i2 += 1
			i2 = 0
			i1 += 1 
		sumValue = np.sum(function)

	elif type(function) is list:
		for i in range(len(function)):
			if type(function[i]) is np.complex128:
				function[i] = np.absolute(function[i])
			else : 
				function[i] = abs(function[i])
		sumValue = sum(function)

	else:
		if type(function) is np.complex128:
			function = np.absolute(function)
		else : 
			function = abs(function)
		sumValue = sum(function)

	return sumValue

def squaredFunction(function):
	if type(function) is np.ndarray:
		i1 = 0
		i2 = 0
		while i1 < function.shape[0] : 
			while i2 < function.shape[1]:
				value = function[i1][i2]
				function[i1][i2] = value**2
				i2 += 1
			i2 = 0
			i1 += 1
	elif type(function) is list:
		for element in function:
			function[element] = function[element]**2
	else:
		function = function**2

	return function

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

	stDevImage = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.int16)
	meanImage = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.int16)
	
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

def noiseInducedBias(cameraGain, readoutNoiseVariance, imageSpeckle, imageUniform, fftFilter, samplingWindowSize):
	exc.isIntOrFloat(cameraGain)
	exc.isIntOrFloat(readoutNoiseVariance)
	exc.isANumpyArray(imageSpeckle)
	exc.isANumpyArray(imageUniform)
	exc.isANumpyArray(fftFilter)
	exc.isInt(samplingWindowSize)

	stdevSpeckle, meanSpeckle = stdevAndMeanWholeImage(image=imageSpeckle, samplingWindow=samplingWindowSize)
	stdevUniform, meanUniform = stdevAndMeanWholeImage(image=imageUniform, samplingWindow=samplingWindowSize)

	valueConv = valueAllPixelsInImage(image=fftFilter)
	absSumValueConv = absSumAllPixels(function=valueConv)
	squaredAbsSumValueConv = squaredFunction(function=absSumValueConv)
	print(squaredAbsSumValueConv)


	# 
	# pixelPosition = [0,0]
	# bandpassFilterSum = np.zeros(shape=(fftFilter.shape[0], fftFilter.shape[1]), dtype=np.int16)
	
	# while pixelPosition[0] < fftFilter.shape[0]:
	# 	noiseArray = []
	# 	while pixelPosition[1] < fftFilter.shape[1]:
	# 		# Values of all pixels contained in one sampling window
	# 		valuesFFTFilter = valueAllPixels(image=fftFilter, px=pixelPosition, samplingWindow=samplingWindowSize)
	# 		print("VALUE PIXEL : {}{}".format(valuesFFTFilter[0], type(valuesFFTFilter[0])))
	# 		# abs and squared of all of those values
	# 		valuesFFTFilterSum = absSumAllPixels(function=valuesFFTFilter)
	# 		print("VALUE PIXEL SUM ABS : {}{}".format(valuesFFTFilterSum, type(valuesFFTFilterSum)))
	# 		valuesFFTFilterSumSquared = squaredFunction(function=valuesFFTFilterSum)
	# 		print("VALUE PIXEL SUM ABS SQUARED : {}{}".format(valuesFFTFilterSumSquared, type(valuesFFTFilterSumSquared)))
	# 		noiseArray.append(valuesFFTFilterSum)
	# 		pixelPosition[1] += 1

	# 	bandpassFilterSum[pixelPosition[0]] = noiseArray
	# 	pixelPosition[1] = 0
	# 	pixelPosition[0] += 1

	# print("BANDPASSFILTER {}{}".format(bandpassFilterSum[0][0], type(bandpassFilterSum[0][0])))

	# Calculate the noise-induced biased function.  
	i1 = 0
	i2 = 0
	bias = np.zeros(shape=(imageSpeckle.shape[0], imageSpeckle.shape[1]), dtype=np.int16)
	while i1 < meanUniform.shape[0]:
		noiseArray = []
		while i2 < meanUniform.shape[1]:
			noiseArray.append((((cameraGain*meanUniform[i1][i2]) + (cameraGain*meanSpeckle[i1][i2]) + readoutNoiseVariance)) * squaredAbsSumValueConv)
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

def contrastCalculation(difference, uniform, speckle, samplingWindow, ffilter):
	exc.isANumpyArray(difference)
	exc.isANumpyArray(uniform)
	exc.isInt(samplingWindow)

	contrastFunction = np.zeros(shape=(difference.shape[0], difference.shape[1]), dtype=np.int16)
	noiseFunction = noiseInducedBias(cameraGain=(30000/65636), readoutNoiseVariance=0.3508935, imageSpeckle=speckle, imageUniform=uniform, fftFilter=ffilter, samplingWindowSize=samplingWindow)

	stdevDifference, meanDifference = stdevAndMeanWholeImage(image=difference, samplingWindow=samplingWindow)
	stdevUniform, meanUniform = stdevAndMeanWholeImage(image=uniform, samplingWindow=samplingWindow)
	reducedStdevDifference = noiseInducedBiasReductionFromStdev(noise=noiseFunction, stdev=stdevDifference)

	i1 = 0
	i2 = 0
	while i1 < stdevDifference.shape[0]:
		while i2 < stdevDifference.shape[1]:
			contrastFunction[i1][i2] = reducedStdevDifference[i1][i2]/meanUniform[i1][i2]
			i2 += 1
		i2 = 0
		i1 += 1

	print("CONTRAST FUNCTION : {}{}".format(contrastFunction[0][0], type(contrastFunction[0][0])))
	return contrastFunction

def highPassFilter(low):
	exc.isANumpyArray(low)

	hi = 1 - low
	return hi





