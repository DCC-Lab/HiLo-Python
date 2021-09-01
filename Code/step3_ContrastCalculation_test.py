import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import scipy.ndimage as simg

sigma = 2
imgDiff = tiff.imread(r"/Users/valeriepineaunoel/Desktop/testImage.tif")
imgDiffBandpass = simg.gaussian_filter(imgDiff, sigma=sigma)

samplingWindow = 7 # in pixel^2
n = int(samplingWindow/2)
pixel = [0,0]
contrastFunction = [] # is it better to create an np array? 

while pixel[0] < imgDiffBandpass.shape[0]:
	contrastArray = []

	while pixel[1] < imgDiffBandpass.shape[1]:
		print("Valeur du pixel : {}".format(pixel))
		valuesImgDiff = []
		valuesImgUniform = []
		positionInSamplingWindow = [pixel[0]-n, pixel[1]-n]
		print("Ouséquejesuis : {}".format(positionInSamplingWindow))
		if positionInSamplingWindow[0] < 0:
			positionInSamplingWindow[0] = 0
		if positionInSamplingWindow[1] < 0: 
			positionInSamplingWindow[1] = 0
		print("Ouséquejesuis2 : {}".format(positionInSamplingWindow))
	
		#Contrast calculation for one pixel
		while positionInSamplingWindow[0] <= pixel[0]+n and positionInSamplingWindow[0] < imgDiffBandpass.shape[0]:
			if positionInSamplingWindow[0] < 0:
				positionInSamplingWindow[0] = 0
			while positionInSamplingWindow[1] <= pixel[1]+n and positionInSamplingWindow[1] < imgDiffBandpass.shape[1]:
				valuePixelDiff = imgDiffBandpass[positionInSamplingWindow[0]][positionInSamplingWindow[1]]
				if valuePixelDiff < 0:
					valuePixelDiff = 0
				
				print("VALEUR {}".format(valuePixelDiff))
				print("Position {}".format(positionInSamplingWindow[1]))
				valuesImgDiff.append(valuePixelDiff)
				valuePixelUniform = imgUniform[positionInSamplingWindow[0]][positionInSamplingWindow[1]]
				valuesImgUniform.append(valuePixelUniform)
				positionInSamplingWindow[1] += 1
				
			positionInSamplingWindow[1] = pixel[1] - n
			if positionInSamplingWindow[1] < 0: 
				positionInSamplingWindow[1] = 0

			print("Position of the samplig window {}".format(positionInSamplingWindow[0]))
			positionInSamplingWindow[0] = positionInSamplingWindow[0] + 1
		
		contrastInSamplingWindow = np.std(valuesImgDiff)/np.mean(valuesImgUniform)
		print("Contrast in one sampling window : {}".format(contrastInSamplingWindow))
		contrastArray.append(contrastInSamplingWindow)
		pixel[1] = pixel[1] + 1

	contrastFunction.append(contrastArray)
	pixel[1] = 0
	pixel[0] = pixel[0] + 1

## Obtain the contrast^2
print(contrastFunction)