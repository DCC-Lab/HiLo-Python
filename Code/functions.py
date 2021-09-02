import tifffile as tiff
import scipy.ndimage as simg
import numpy as np
import exceptions as exc

def image(imagepath):
	exc.isTifOrTiff(imagepath)
	exc.isString(imagepath)
	image = tiff.imread(imagepath) 
	return image

def gaussianFilter(sigma, image):
	exc.isANumpyArray(image)
	exc.isDefined(sigma)
	exc.isDefined(image)
	exc.isIntOrFloat(sigma)

	imgGaussian = simg.gaussian_filter(image, sigma=sigma)
	return imgGaussian

def valueOnePixel(image, pixelPosition):
	value = image[pixelPosition[0]][pixelPosition[1]]
	if value < 0:
		value = 0
	return value

def valueAllPixels(image):
	pixel = [0,0]
	values = []
	while pixel[0] < image.shape[0]:
		while pixel[1] < image.shape[1]:
			valuePixel = valueOnePixel(image=image, pixelPosition=pixel)
			values.append(valuePixel)
			pixel[1] = pixel[1] + 1
		pixel[1] = 0
		pixel[0] = pixel[0] + 1
	return values

def absSumAllPixels(function):
	for element in function:
		function[element] = abs(function[element]) # list indices must be integers or slices, not numpy.complex128
	sumValue = sum(function)
	return sumValue

def squaredFunction(function):
	if type(function) is np.ndarray:
		element1 = 0
		element2 = 0
		while element1 < function.shape[0] : 
			while element2 < function.shape[1]:
				value = function[element1][element2]
				function[element1][element2] = value**2
				element2 += 1
			element1 += 1
	else:
		for element in function:
			function[element] = function[element]**2

	return function

def stDevAndMeanOnePixel(image1, halfSamplingWindow, pixel, position, image2=None): 
	exc.isDefined(image1)
	exc.isDefined(halfSamplingWindow)
	exc.isDefined(pixel)
	exc.isDefined(position)
	exc.isANumpyArray(image1)
	if image2 is not None : 
		exc.isANumpyArray(image2)
	exc.isInt(halfSamplingWindow)

	if type(pixel) is not list or len(pixel) != 2:
		raise Exception("pixel must be a list of two int.")
	 
	valuesImg1 = []
	valuesImg2 = []
	while position[0] <= pixel[0]+halfSamplingWindow and position[0] < image1.shape[0]:
				if position[0] < 0:
					position[0] = 0
				while position[1] <= pixel[1]+halfSamplingWindow and position[1] < image1.shape[1]:
					valuePixel1 = valueOnePixel(image = image1, pixelPosition = position)
					valuesImg1.append(valuePixel1)
						
					if image2 is not None : 	
						valuePixel2 = valueOnePixel(image = image2, pixelPosition = position)
						valuesImg2.append(valuePixel2)
				
					position[1] += 1
				
				position[1] = pixel[1] - halfSamplingWindow
				if position[1] < 0: 
					position[1] = 0

				position[0] = position[0] + 1

	if image2 is not None : 
		std, mean = np.std(valuesImg1), np.mean(valuesImg2) 
	else: 
		std, mean = np.std(valuesImg1), np.mean(valuesImg1)
		
	return std, mean

def stdevAndMeanWholeImage(image, samplingWindow):
	exc.isANumpyArray(image)
	exc.isInt(samplingWindow)

	n = int(samplingWindow/2)
	pixelPosition = [0,0]
	stDevImage = np.zeros(shape=(image.shape[0], image.shape[1]))
	meanImage = np.zeros(shape=(image.shape[0], image.shape[1]))

	while pixelPosition[0] < image.shape[0]:
		stDevArray = []
		meanArray = []

		while pixelPosition[1] < image.shape[1]:
			positionInSamplingWindow = [pixelPosition[0]-n, pixelPosition[1]-n]
			if positionInSamplingWindow[0] < 0:
				positionInSamplingWindow[0] = 0
			if positionInSamplingWindow[1] < 0: 
				positionInSamplingWindow[1] = 0
	
			# Calculations of one pixel
			stdev, mean = stDevAndMeanOnePixel(image1=image, halfSamplingWindow=n, pixel=pixelPosition, position=positionInSamplingWindow)
				
			stDevArray.append(stdev)
			meanArray.append(mean)
			pixelPosition[1] = pixelPosition[1] + 1

		stDevImage[pixelPosition[0]] = stDevArray
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
	valuesFFTFilter = valueAllPixels(image=fftFilter)
	absfftFilter = absSumAllPixels(function=valuesFFTFilter)
	squaredAbsFilter = squaredFunction(function=absfftFilter)

	element1 = 0
	element2 = 0
	bias = np.zeros(shape=(imageSpeckle.shape[0], imageSpeckle.shape[1]))
	while element1 < meanUniform.shape[0]:
		noiseArray = []
		while element2 < meanUniform.shape[1]:
			noiseArray.append((((cameraGain*meanUniform[element1][element2]) + (cameraGain*meanSpeckle[element1][element2]) + readoutNoiseVariance)) * squaredAbsFilter)
			element2 += 1
		bias.append(noiseArray)	
		element2 = 0
		element1 += 1
		
	return bias

def noiseInducedBiasReductionFromStdev(noise, stdev, pixel):
	stdev = stdev - noise[pixel[0]][pixel[1]]
	return stdev

def contrastCalculation(difference, uniform, speckle, samplingWindow, ffilter):
	exc.isANumpyArray(difference)
	exc.isANumpyArray(uniform)
	exc.isInt(samplingWindow)

	n = int(samplingWindow/2)
	pixelPosition = [0,0]
	contrastFunction = np.zeros(shape=(difference.shape[0], difference.shape[1]))
	noiseFunction = noiseInducedBias(cameraGain=(30000/65636), readoutNoiseVariance=0.3508935, imageSpeckle=speckle, imageUniform=uniform, fftFilter=ffilter, samplingWindowSize=samplingWindow)

	while pixelPosition[0] < difference.shape[0]:
		contrastArray = []

		while pixelPosition[1] < difference.shape[1]:
			positionInSamplingWindow = [pixelPosition[0]-n, pixelPosition[1]-n]
			if positionInSamplingWindow[0] < 0:
				positionInSamplingWindow[0] = 0
			if positionInSamplingWindow[1] < 0: 
				positionInSamplingWindow[1] = 0
	
			# Calculations of one pixel
			stdevDifference, meanUniform = stDevAndMeanOnePixel(image1=difference, image2=uniform, halfSamplingWindow=n, pixel=pixelPosition, position=positionInSamplingWindow)
			reducedStdevDifference = noiseInducedBiasReductionFromStdev(noise=noiseFunction, stdev=stdevDifference, pixel=pixelPosition)

			contrastInSamplingWindow = reducedStdevDifference/meanUniform
			contrastArray.append(contrastInSamplingWindow)
			pixelPosition[1] = pixelPosition[1] + 1

		contrastFunction[pixelPosition[0]] = contrastArray
		pixelPosition[1] = 0
		pixelPosition[0] = pixelPosition[0] + 1

	return contrastFunction








