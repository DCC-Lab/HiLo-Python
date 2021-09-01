import tifffile as tiff
import scipy.ndimage as simg
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

def stDevAndMeanContrast(imageDifference, imageUniform, halfSamplingWindow, pixel): 
	exc.isANumpyArray(imageDifference)
	exc.isANumpyArray(imageUniform)
	exc.isDefined(imageDifference)
	exc.isDefined(imageUniform)
	exc.isDefined(halfSamplingWindow)
	exc.isDefined(pixel)
	exc.isInt(halfSamplingwindow)

	if type(pixel) is not list or len(pixel) != 2:
		raise Exception("pixel must be a list of two int.")
	else: 
		valuesImgDiff = []
		valuesImgUniform = []
		while positionInSamplingWindow[0] <= pixel[0]+halfSamplingWondow and positionInSamplingWindow[0] < imageDifference.shape[0]:
					if positionInSamplingWindow[0] < 0:
						positionInSamplingWindow[0] = 0
					while positionInSamplingWindow[1] <= pixel[1]+halfSamplingWondow and positionInSamplingWindow[1] < imageDifference.shape[1]:
						valuePixelDiff = imageDifference[positionInSamplingWindow[0]][positionInSamplingWindow[1]]
						if valuePixelDiff < 0:
							valuePixelDiff = 0
				
						valuesImgDiff.append(valuePixelDiff)
						valuePixelUniform = imageUniform[positionInSamplingWindow[0]][positionInSamplingWindow[1]]
						valuesImgUniform.append(valuePixelUniform)
				
						positionInSamplingWindow[1] += 1
				
					positionInSamplingWindow[1] = pixel[1] - halfSamplingWondow
					if positionInSamplingWindow[1] < 0: 
						positionInSamplingWindow[1] = 0

					positionInSamplingWindow[0] = positionInSamplingWindow[0] + 1

		std, mean = np.std(valuesImgDiff), np.mean(valuesImgUniform) 
		return std, mean

def contrastCalculation(difference, uniform, samplingWindow):
	exc.isANumpyArray(difference)
	exc.isANumpyArray(uniform)
	exc.isInt(samplingWindow)

	n = int(samplingWindow/2)
	pixelPosition = [0,0]
	contrastFunction = np.zeros(shape=(imageDifference.shape[0], imageDifference.shape[1]))

	while pixelPosition[0] < image.shape[0]:
		contrastArray = []

		while pixelPosition[1] < imageDifference.shape[1]:
			positionInSamplingWindow = [pixelPosition[0]-n, pixelPosition[1]-n]
			if positionInSamplingWindow[0] < 0:
				positionInSamplingWindow[0] = 0
			if positionInSamplingWindow[1] < 0: 
				positionInSamplingWindow[1] = 0
	
			# Calculations of one pixel
			stDevDifference, meanUniform = stDevAndMeanContrast(imageDifference=difference, imageUniform=uniform, halfSamplingWindow=n, pixel=pixelPosition)
				
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















