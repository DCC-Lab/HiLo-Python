import functions as fun
import exceptions as exc
import matplotlib.pyplot as plt
import numpy as np

"""This file tests multiple functions used in the algorithm."""

def testStdevAndMeanOfWholeImage(image,area):
	'''Calculates the mean and the standard deviation of an image in a sampling window from 0 to the value entered as area.
	The results are printed for each matrix to verify is they are the same whatever the value of the sampling window.

	If the entered image is a black image, then the standard deviation and the mean is 0, whatever the size of the sampling window is.
	If the entered image is white, then the standard deviation is 0 for all pixel and the mean is 255, whatever the size of the sampling window.'''

	testImage = fun.image(image)
	i = 0
	while i <= area:
		stdev, mean = fun.stdevAndMeanWholeImage(image=testImage, samplingWindow=i)
		print(i)
		print("STDEV : {}".format(stdev))
		print("MEAN : {}".format(mean))
		i += 1
	return 

def testAbsAndSum(shape1, shape2):
	# 1 : Produce an image with values of -1 everywhere
	testImage = np.zeros(shape=(shape1, shape2))
	element1 = 0
	element2 = 0
	while element1 < testImage.shape[0]:
		while element2 < testImage.shape[1]: 
			testImage[element1][element2] = -1
			element2 += 1
		element2 =0 
		element1 += 1

	# Step 2 : Make a manual sum and use the function. Compare the two values. 
	manualSum = np.sum(np.absolute(testImage))
	functionSum = fun.absSumAllPixels(testImage)
	exc.areValuesEqual(value1=manualSum, value2=functionSum)
	return



if __name__ == "__main__":
	#testStdevAndMeanOfWholeImage(image="/Users/valeriepineaunoel/Documents/HiLo-Python/Data/testWhiteImage.tif", area=7)
	testAbsAndSum(10,10)
