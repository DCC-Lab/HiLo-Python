import tifffile as tiff
import scipy.ndimage as simg
import numpy as np
import math as math
import exceptions as exc
import matplotlib.pyplot as plt
import skimage as ski
import cmath as cmath

def createImage(imagepath):
	exc.isTifOrTiff(imagepath)
	exc.isString(imagepath)

	image = tiff.imread(imagepath)
	imageRescale = ski.util.img_as_uint(image)

	return imageRescale

def rescaleImage(image):
	# Find the minimal value
	minPixel = np.amin(image)

	if minPixel < 0:
		# Rescale the data to only have positive values. 
		i1 = 0
		i2 = 0
		while i1 < image.shape[0]:
			while i2 < image.shape[1]:
				image[i1][i2] = image[i1][i2] + abs(minPixel)
				i2 += 1
			i2 = 0
			i1 += 1

	return image


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

	# rescale the values to get rid of the negative values. 
	rescaledDiff = rescaleImage(difference)

	# Convert data in uint16
	differenceInInt = rescaledDiff.astype(np.uint16)

	return differenceInInt

def cameraOTF(image):
	"""Optical transfer function is the fft of the PSF. The modulation transfer function is the magnitude of the complex OTF."""
	pixels = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.float64)
	x = 0
	y = 0
	while x < image.shape[0]:
		while y < image.shape[1]:
			x1 = (2*x-image.shape[0])*np.pi/image.shape[0]
			y1 = (2*y-image.shape[1])*np.pi/image.shape[1]
			if x1 != 0 and y1 != 0:
				pixels[x][y] = (math.sin(x1)*math.sin(y1))/(x1*y1)
			elif x1 == 0 and y1 != 0: 
				pixels[x][y] = math.sin(y1)/y1
			elif x1 != 0 and y1 == 0:
				pixels[x][y] = math.sin(x1)/x1
			elif x1 != 0 and y1 != 0:
				pixels[x][y] = 1
			if pixels[x][y] < 0:
				pixels[x][y] = 0
			y += 1
		y = 0
		x += 1

	return pixels

def imagingOTF(numAperture, wavelength, magnification, pixelSize, image):
	sizex = image.shape[0]
	sizey = image.shape[1]
	bandwidth = 2 * numAperture / (wavelength * 1e-9)
	scaleUnits = magnification / (pixelSize * 1e-6) / bandwidth
	pixel = np.zeros(shape=(sizex, sizey), dtype=np.float64)

	x = 0
	y = 0
	while x<sizex:
		while y<sizey:
			pixel[x][y] = math.sqrt(((x+0.5-sizex/2)**2 / (sizex/2-1)**2) + ((y+0.5-sizey/2)**2 / (sizey/2-1)**2))*scaleUnits
			if pixel[x][y] > 1:
				pixel[x][y] = 1
			pixel[x][y] = 0.6366197723675814 * math.acos(pixel[x][y]) - pixel[x][y] * math.sqrt(1-pixel[x][y]*pixel[x][y])
			if pixel[x][y] < 0:
				pixel[x][y] = 0
			y += 1
		y = 0
		x += 1

	return pixel

def obtainFFTFitler(image, filteredImage):
	""""
	returns:
	2D numpy array of complex128 numbers. 
	"""
	exc.isANumpyArray(image)
	exc.isANumpyArray(filteredImage)
	exc.isSameShape(image1=image, image2=filteredImage)

	fftImage = np.fft.fftshift(np.fft.fft2(image))
	fftFilteredImage = np.fft.fftshift(np.fft.fft2(filteredImage))
	fftFilter = np.divide(fftFilteredImage, fftImage) # convolution in the space domain is a multiplication in the Fourier space and vice versa
	normfftFilter = np.divide(fftFilter, np.amax(fftFilter))/fftFilter.shape[0]/fftFilter.shape[1]

	return normfftFilter

def gaussianFilter(sigma, image):
	exc.isANumpyArray(image)
	exc.isIntOrFloat(sigma)

	imgGaussian = simg.gaussian_filter(image, sigma=sigma, mode="nearest") 
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

		position[1] = 0
		position[0] = position[0] + 1

	return values

def valueAllPixelsInSW(image, px, samplingWindow):
	"""
	returns:
	list of the values of all elements in image at the pixel position px in the area of the sampling window samplingWindow. 
	"""
	if samplingWindow == 0:
		return []

	else:
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

def absSum(array, samplingW):
	"""
	returns:
	numpy.nparray of the absolute sum of the values included in the sampling window samplingW of all pixels/elements in array.
	"""
	absSumImage = np.zeros(shape=(array.shape[0], array.shape[1]))
	pixelPosition = [0,0]

	while pixelPosition[0] < array.shape[0]:
		absSumValues = []
		while pixelPosition[1] < array.shape[1]:
			# Calculations of one pixel
			valuesInSW = valueAllPixelsInSW(image=array, px=pixelPosition, samplingWindow=samplingW)
			if type(array[pixelPosition[0]][pixelPosition[1]]) is np.complex128:
				absSumValues.append(np.sum(np.absolute(valuesInSW)))
			else:
				absSumValues.append(np.sum(np.abs(valuesInSW)))

			pixelPosition[1] = pixelPosition[1] + 1

		absSumImage[pixelPosition[0]] = absSumValues
		pixelPosition[1] = 0
		pixelPosition[0] = pixelPosition[0] + 1

	return absSumImage.real

def squared(array):
	"""
	returns:
	The squared value of the list, numpy.ndarray or number. The type of array is kept throughout the execution of the function. 
	"""
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
		for i in array:
			array[i] = array[i]**2
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

	fftFilter = obtainFFTFitler(image=imageUniform, filteredImage=difference)
	sumAbsfftFilter = absSum(array=fftFilter, samplingW=samplingWindowSize)
	squaredSumAbsfftFilter = squared(array=sumAbsfftFilter)

	# Calculate the noise-induced biased function. 
	i1 = 0
	i2 = 0
	bias = np.zeros(shape=(imageSpeckle.shape[0], imageSpeckle.shape[1]))
	while i1 < meanUniform.shape[0]:
		noiseArray = []
		while i2 < meanUniform.shape[1]:
			noiseArray.append(((np.sqrt(cameraGain*meanUniform[i1][i2]) + (cameraGain*meanSpeckle[i1][i2]) + readoutNoiseVariance) * squaredSumAbsfftFilter[i1][i2]))

			i2 += 1
		bias[i1] = noiseArray	
		i2 = 0
		i1 += 1
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

	noiseFunction = np.zeros(shape=(uniform.shape[0], uniform.shape[1]))
	# noiseFunction = noiseInducedBias(cameraGain=1, readoutNoiseVariance=0.3508935, imageSpeckle=speckle, imageUniform=uniform, difference=gaussianImage, samplingWindowSize=samplingWindow, sigma=sigma)

	# calculate the stdev and the mean of the speckle and the uniform image
	stdevDifference, meanDifference = stdevAndMeanWholeImage(image=gaussianImage, samplingWindow=samplingWindow)
	stdevUniform, meanUniform = stdevAndMeanWholeImage(image=uniform, samplingWindow=samplingWindow)

	# subtract the noise-induced bias from the stdev of the filtered difference image
	reducedStdevDifference = noiseInducedBiasReductionFromStdev(noise=noiseFunction, stdev=stdevDifference)

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

	#print("CONTRAST FUNCTION : {}{}".format(contrastFunction, contrastFunction.dtype))
	return contrastFunction

def lowpassFilter(image, sigmaFilter):

	exc.isANumpyArray(image)
	exc.isIntOrFloat(sigmaFilter)

	x, y = np.meshgrid(np.linspace(-1,1,image.shape[0]), np.linspace(-1,1,image.shape[1]))
	d = np.sqrt(x*x+y*y)
	sigma = (sigmaFilter*0.18)/(2*math.sqrt(2*math.log(2)))
	mu = 0.0
	gauss = (1/(sigma*2*np.pi)) * np.exp(-((d-mu)**2/(2.0*sigma**2)))
	maxPixel = np.amax(gauss)
	gaussNorm = gauss/maxPixel

	return gaussNorm

def highpassFilter(low):
	exc.isANumpyArray(low)

	hi = 1 - low
	return hi

def estimateEta(speckle, uniform, sigma, ffthi, fftlo, sig):
	"""
	returns:
	the function eta in numpy.ndarray. Calculations taken from the java code for the HiLo Fiji plugin and from personal communication with Olivier Dupont-Therrien at Bliq Photonics
	"""
	differenceImage = createDifferenceImage(speckle=speckle, uniform=uniform)
	gaussianImage = gaussianFilter(sigma=sigma, image=differenceImage)
	bandpassFilter = obtainFFTFitler(image=uniform, filteredImage=gaussianImage)

	illuminationOTF = imagingOTF(numAperture=1, wavelength=488e-9, magnification=20, pixelSize=6.5, image=uniform)
	detectionOTF = imagingOTF(numAperture=1, wavelength=520e-9, magnification=20, pixelSize=6.5, image=uniform)
	camOTF = cameraOTF(image=uniform)
	print(f"Ill OTF : {illuminationOTF}{illuminationOTF.dtype}")
	print(f"DET OTF {detectionOTF}{detectionOTF.dtype}")
	
	# Method 1 : Generate a function eta for each pixels.
	#eta = np.zeros(shape=(uniform.shape[0], uniform.shape[1]), dtype=np.complex128) 
	#x = 0
	#y = 0
	#while x<camOTF.shape[0]:
	#	etaList = []
	#	while y<camOTF.shape[1]:
	#		denominator = (bandpassFilter[x][y] * detectionOTF[x][y] * camOTF[x][y])**2 * np.absolute(illuminationOTF[x][y])
	#		numerator = illuminationOTF[x][y]
	#		if denominator == 0 or numerator == 0:
	#			result = 0
	#		else:
	#			result = cmath.sqrt(numerator / denominator)
	#		etaList.append(result)
	#		y += 1
	#	eta[x] = etaList
	#	y = 0
	#	x += 1

	# Method 2 : Generate one value for the whole image. 
	#numerator = 0
	#denominator = 0
	#x = 0
	#y = 0
	#while x<camOTF.shape[0]:
	#	while y<camOTF.shape[1]:
	#		firstStep = (bandpassFilter[x][y] * detectionOTF[x][y] * camOTF[x][y])**2
	#		secondStep = np.absolute(illuminationOTF[x][y])
	#		denominator += (bandpassFilter[x][y] * detectionOTF[x][y] * camOTF[x][y])**2 * np.absolute(illuminationOTF[x][y])
	#		numerator += illuminationOTF[x][y]
	#		y += 1
	#	y = 0
	#	x += 1
	#eta = cmath.sqrt(numerator / denominator) * 1.2

	#Method 3 : eta is obtained experimentally from the HI and the LO images
	#numerator = 0
	#denominator = 0
	#x = 0
	#y = 0
	#while x<ffthi.shape[0]:
	#	while y<ffthi.shape[1]:
	#		numerator += np.absolute(ffthi[x][y])
	#		denominator += np.absolute(fftlo[x][y])
	#		y += 1
	#	y = 0
	#	x += 1
	#eta = numerator/denominator

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

	print(len(elementPosition))
	for i in elementPosition:
		print(i)
		denominator += math.sqrt(fftlo[i[0]][i[1]].real**2 + fftlo[i[0]][i[1]].imag**2)
		d += 1

	print(f"N : {len(elementPosition)}")
	print(f"D : {d}")
	print(f"Num : {numerator}")
	print(f"Den : {denominator}")
	eta = numerator/denominator



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

	print(f"ETA : {eta}{type(eta)}")	

	return eta


def createHiLoImage(uniform, speckle, sigma, sWindow):

	# calculate the contrast weighting function
	contrast = contrastCalculation(uniform=uniform, speckle=speckle, samplingWindow=sWindow, sigma=sigma)

	# Create the filters
	lowFilter = lowpassFilter(image=uniform, sigmaFilter=sigma)
	highFilter = highpassFilter(low=lowFilter)

	# Create fft of uniform image and normalize with max value
	fftuniform = np.fft.fftshift(np.fft.fft2(uniform))
	normfftuniform = fftuniform/np.amax(fftuniform)

	# Apply the contrast weighting function on the uniform image. FFT of the result and then normalize with max value.
	cxu = contrast*uniform
	fftcxu = np.fft.fftshift(np.fft.fft2(cxu))

	# Apply the low-pass frequency filter on the uniform image to create the LO portion
	# Ilp = LP[C*Iu]
	fftLO = lowFilter*fftcxu
	LO = np.fft.ifft2(np.fft.ifftshift(fftLO))

	# Apply the high-pass frequency filter to the uniform image to obtain the HI portion
	# Ihp = HP[Iu]
	fftHI = highFilter*fftuniform
	HI = np.fft.ifft2(np.fft.ifftshift(fftHI))

	# Estimate the function eta for scaling the frequencies of the low image. Generates a complex number. 
	eta = estimateEta(speckle=speckle, uniform=uniform, sigma=sigma, fftlo=fftLO, ffthi=fftHI, sig=sigma)

	#print(f"LO : {LO}{type(LO)}{LO.dtype}")
	#print(f"HI : {HI}{type(HI)}{HI.dtype}")

	complexHiLo = eta*LO + HI

	# convert the complexHiLo image to obtain the modulus of each values. 
	HiLo = np.zeros(shape=(complexHiLo.shape[0], complexHiLo.shape[1]), dtype=np.uint16)
	x = 0 
	y = 0
	while x < complexHiLo.shape[0]:
		while y < complexHiLo.shape[1]:
			print(complexHiLo[x][y], complexHiLo[x][y].real, complexHiLo[x][y].imag)
			HiLo[x][y] = cmath.sqrt(complexHiLo[x][y].real**2 + complexHiLo[x][y].imag**2)
			y += 1
		y = 0	
		x += 1

	#tiff.imshow(HiLo)
	#plt.show()
	tiff.imsave("/Users/valeriepineaunoel/Documents/HiLo-Python/Data/HiLoExp_2sigma_2.tiff", HiLo)
	print(f"complexHILO : {complexHiLo}{type(complexHiLo)}{complexHiLo.dtype}")
	print(f"HILO : {HiLo}{type(HiLo)}{HiLo.dtype}")

	return HiLo



