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

def stDevAndMeanContrast(image1, image2, halfSamplingWindow, pixel, position): 
	exc.isANumpyArray(image1)
	exc.isANumpyArray(image2)
	exc.isDefined(image1)
	exc.isDefined(image2)
	exc.isDefined(halfSamplingWindow)
	exc.isDefined(pixel)
	exc.isInt(halfSamplingWindow)

	if type(pixel) is not list or len(pixel) != 2:
		raise Exception("pixel must be a list of two int.")
	else: 
		valuesImg1 = []
		valuesImg2 = []
		while position[0] <= pixel[0]+halfSamplingWindow and position[0] < image1.shape[0]:
					if position[0] < 0:
						position[0] = 0
					while position[1] <= pixel[1]+halfSamplingWindow and position[1] < image1.shape[1]:
						valuePixel1 = image1[position[0]][position[1]]
						if valuePixel1 < 0:
							valuePixel1 = 0
						valuesImg1.append(valuePixel1)
						
						valuePixel2 = image2[position[0]][position[1]]
						valuesImg2.append(valuePixelUniform)
				
						position[1] += 1
				
					position[1] = pixel[1] - halfSamplingWindow
					if position[1] < 0: 
						position[1] = 0

					position[0] = position[0] + 1

		std, mean = np.std(valuesImg1), np.mean(valuesImg2) 
		return std, mean

def contrastCalculation(difference, uniform, samplingWindow):
	exc.isANumpyArray(difference)
	exc.isANumpyArray(uniform)
	exc.isInt(samplingWindow)

	n = int(samplingWindow/2)
	pixelPosition = [0,0]
	contrastFunction = np.zeros(shape=(difference.shape[0], difference.shape[1]))

	while pixelPosition[0] < difference.shape[0]:
		contrastArray = []

		while pixelPosition[1] < difference.shape[1]:
			positionInSamplingWindow = [pixelPosition[0]-n, pixelPosition[1]-n]
			if positionInSamplingWindow[0] < 0:
				positionInSamplingWindow[0] = 0
			if positionInSamplingWindow[1] < 0: 
				positionInSamplingWindow[1] = 0
	
			# Calculations of one pixel
			stDevDifference, meanUniform = stDevAndMeanContrast(imageDifference=difference, imageUniform=uniform, halfSamplingWindow=n, pixel=pixelPosition, position=positionInSamplingWindow)
				
			contrastInSamplingWindow = stDevDifference/meanUniform
			contrastArray.append(contrastInSamplingWindow)
			pixelPosition[1] = pixelPosition[1] + 1

		contrastFunction[pixelPosition[0]] = contrastArray
		pixelPosition[1] = 0
		pixelPosition[0] = pixelPosition[0] + 1

	return contrastFunction

def contrastCalculationSquared(function):
	exc.isANumpyArray(function)
	
	element1 = 0
	element2 = 0
	while element1 < function.shape[0] : 
		while element2 < function.shape[1]:
			value = function[element1][element2]
			function[element1][element2] = value**2
			element2 += 1
		element1 += 1

	return function















