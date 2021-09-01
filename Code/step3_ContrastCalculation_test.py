import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import scipy.ndimage as simg


# Écentuellement, ce sera une fonction (ou classe?) qui va calculer le contraste. Filtre gaussien se fera avant. 
# arguments : imagediff, imageuniforme, samplingwindow, 

sigma = 2
imgDiff = tiff.imread(r"/Users/valeriepineaunoel/Desktop/testImage.tif")
imgDiffBandpass = simg.gaussian_filter(imgDiff, sigma=sigma)

samplingWindow = 7 # in pixel^2
n = int(samplingWindow/2)
pixel = [0,0]
contrastFunction = np.zeros(shape=(imgDiff.shape[0], imgDiff.shape[1])) # is it better to create an np array? 
print(contrastFunction)

while pixel[0] < imgDiffBandpass.shape[0]:
	contrastArray = []

	while pixel[1] < imgDiffBandpass.shape[1]:
		valuesImgDiff = []
		valuesImgUniform = []
		positionInSamplingWindow = [pixel[0]-n, pixel[1]-n]
		if positionInSamplingWindow[0] < 0:
			positionInSamplingWindow[0] = 0
		if positionInSamplingWindow[1] < 0: 
			positionInSamplingWindow[1] = 0
	
		#Contrast calculation for one pixel
		while positionInSamplingWindow[0] <= pixel[0]+n and positionInSamplingWindow[0] < imgDiffBandpass.shape[0]:
			if positionInSamplingWindow[0] < 0:
				positionInSamplingWindow[0] = 0
			while positionInSamplingWindow[1] <= pixel[1]+n and positionInSamplingWindow[1] < imgDiffBandpass.shape[1]:
				valuePixelDiff = imgDiffBandpass[positionInSamplingWindow[0]][positionInSamplingWindow[1]]
				if valuePixelDiff < 0:
					valuePixelDiff = 0
				
				valuesImgDiff.append(valuePixelDiff)
				valuePixelUniform = imgUniform[positionInSamplingWindow[0]][positionInSamplingWindow[1]]
				valuesImgUniform.append(valuePixelUniform)
				
				positionInSamplingWindow[1] += 1
				
			positionInSamplingWindow[1] = pixel[1] - n
			if positionInSamplingWindow[1] < 0: 
				positionInSamplingWindow[1] = 0

			positionInSamplingWindow[0] = positionInSamplingWindow[0] + 1
		
		contrastInSamplingWindow = np.std(valuesImgDiff)/np.mean(valuesImgUniform)
		contrastArray.append(contrastInSamplingWindow)
		pixel[1] = pixel[1] + 1

	contrastFunction[pixel[0]] = contrastArray
	pixel[1] = 0
	pixel[0] = pixel[0] + 1

## Obtain the contrast^2
element1 = 0
element2 = 0
while element1 < contrastFunction.shape[0] : 
	while element2 < contrastFunction.shape[1]:
		value = contrastFunction[element1][element2]
		contrastFunction[element1][element2] = value**2
		element2 += 1
	element1 += 1
















