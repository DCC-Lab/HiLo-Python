import unittest 
import numpy as np
import functions as fun

class TestHiLo(unittest.TestCase):
	def testRescaling(self):
		with self.subTest("Test subtraction of two images that results in only positive values"): # Message that I was the user to see while the tests are running
			image1 = np.asarray([[20, 20, 20],[20, 20, 20], [20, 20, 20]])
			image2 = np.asarray([[3, 3, 3],[3, 3, 3], [3, 3, 3]])
			differenceImage = fun.createDifferenceImage(speckle=image1, uniform=image2)

			imgGroundTruth = np.asarray([[17, 17, 17],[17, 17, 17], [17, 17, 17]])
			self.assertTrue(np.all(np.equal(differenceImage, imgGroundTruth))) # np.all True or False of the overall array

		with self.subTest("Test subtraction two images that results with some negative values"):
			image1 = np.asarray([[1, 1, 1],[1, 1, 1],[1, 1, 1]])
			image2 = np.asarray([[10, 10, 10],[10, 10, 10],[10, 10, 10]])
			differenceImage = fun.createDifferenceImage(speckle=image1, uniform=image2)

			imgGroundTruth = np.asarray([[0, 0, 0],[0, 0, 0], [0, 0, 0]])
			self.assertTrue(np.all(np.equal(differenceImage, imgGroundTruth)))
	
		with self.subTest("Test subtraction two images that results with some values equal to 0"):
			image1 = np.asarray([[1, 1, 1],[1, 1, 1],[1, 1, 1]])
			image2 = np.asarray([[1, 1, 1],[1, 1, 1],[1, 1, 1]])
			differenceImage = fun.createDifferenceImage(speckle=image1, uniform=image2)

			imgGroundTruth = np.asarray([[0, 0, 0],[0, 0, 0], [0, 0, 0]])
			self.assertTrue(np.all(np.equal(differenceImage, imgGroundTruth)))

	def testSizeFFTFilter(self):
		with self.subTest("Test the size of the numpy array that is returned after producing the fft of an image."):
			image = np.asarray([[255, 0, 255],[255, 0, 255],[255, 0, 255]], dtype=np.uint8)
			imageWithFilter = np.asarray([[193, 153, 193],[193, 153, 193],[193, 153, 193]], dtype=np.uint8)
			filterOnly = fun.obtainFFTFitler(image=image, filteredImage=imageWithFilter)
			
			sizeFilter = filterOnly.shape
			sizeImage = image.shape
			self.assertEqual(sizeFilter, sizeImage)

	def testGaussianFilter(self):
		with self.subTest("Test the gaussian filter"):
			image = np.asarray([[255, 0, 255],[255, 0, 255],[255, 0, 255]], dtype=np.uint8)
			filteredImage = fun.gaussianFilter(sigma=1, image=image)

			trueFilteredImage = np.asarray([[193, 153, 193],[193, 153, 193],[193, 153, 193]], dtype=np.uint8)
			self.assertTrue(np.all(np.equal(filteredImage, trueFilteredImage)))

	def testValueOnePixel(self):
		with self.subTest("Test if valueOnePixel() actually returns the value one element in a numpy array of int."):
			image = np.asarray([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
			pixel = [1,1]
			value = fun.valueOnePixel(image=image, pixelPosition=pixel)

			trueValueOfPixel = 5
			self.assertEqual(value, trueValueOfPixel)

		with self.subTest("Test if valueOnePixel() actually returns the value one element in a numpy array of float."):
			image = np.asarray([[1.0, 2.1, 3.2],[4.3, 5.4, 6.5],[7.6, 8.7, 9.8]])
			pixel = [2,1]
			value = fun.valueOnePixel(image=image, pixelPosition=pixel)

			trueValueOfPixel = 8.7
			self.assertEqual(value, trueValueOfPixel)

		with self.subTest("Test if the type of value returned by valueOnePixel() fits the actual type of value in the initial numpy array"):
			image = np.asarray([[1, 2, 3],[4, 5, 6],[7, 8, 9]], dtype=np.float32)
			pixel = [2,2]
			typeValue = type(fun.valueOnePixel(image=image, pixelPosition=pixel))

			trueTypeValueOfPixel = type(np.float32(1))
			self.assertEqual(typeValue, trueTypeValueOfPixel)

	def testValueAllPixelsInImage(self):
		with self.subTest("Test if returns all value of pixels in an image."):
			image = np.asarray([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
			valueAllPixels = fun.valueAllPixelsInImage(image=image)

			trueValueAllPixels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
			self.assertTrue(np.all(np.equal(valueAllPixels, trueValueAllPixels)))

		with self.subTest("Test if the number of elements in the final list matches the number of elements in the input numpy array."):
			image = np.asarray([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
			sizeOfValueAllPixels = len(fun.valueAllPixelsInImage(image=image))

			trueSizeOfValueAllPixels = len([1, 2, 3, 4, 5, 6, 7, 8, 9])
			self.assertTrue(np.all(np.equal(sizeOfValueAllPixels, trueSizeOfValueAllPixels)))

	def testValueAllPixelsInSW(self):
		with self.subTest("Test if returns the list of values of all pixels in sampling window"):
			image = np.asarray([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
			valuesInSW = fun.valueAllPixelsInSW(image=image, px=[1,1], samplingWindow=3)

			trueValuesInSW = [1, 2, 3, 4, 5, 6, 7, 8, 9]
			self.assertTrue(np.all(np.equal(valuesInSW, trueValuesInSW)))

		with self.subTest("Test if returns the list of values of all pixels in sampling window if sampling window is 0"):
			image = np.asarray([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
			valuesInSW = fun.valueAllPixelsInSW(image=image, px=[1,1], samplingWindow=0)

			trueValuesInSW = []
			self.assertTrue(np.all(np.equal(valuesInSW, trueValuesInSW)))

		with self.subTest("Test if returns the list of values of all pixels in sampling window if sampling window is 1"):
			image = np.asarray([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
			valuesInSW = fun.valueAllPixelsInSW(image=image, px=[1,1], samplingWindow=1)

			trueValuesInSW = [5]
			self.assertTrue(np.all(np.equal(valuesInSW, trueValuesInSW)))

		with self.subTest("Test if returns the list of values of all pixels when numpy array has negative numbers"):
			image = np.asarray([[-1, -2, -3],[-4, -5, -6],[-7, -8, -9]])
			valuesInSW = fun.valueAllPixelsInSW(image=image, px=[1,1], samplingWindow=3)

			trueValuesInSW = [-1, -2, -3, -4, -5, -6, -7, -8, -9]
			self.assertTrue(np.all(np.equal(valuesInSW, trueValuesInSW)))

	def testAbsSumInSW(self):
		with self.subTest("Test if the function absSum() returns the right value in sampling window."):
			image = np.asarray([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
			absoluteSum = fun.absSum(array=image, samplingW=3)

			trueAbsSum = [[12., 21., 16.],[27., 45., 33.],[24., 39., 28.]]
			self.assertTrue(np.all(np.equal(absoluteSum, trueAbsSum)))

		with self.subTest("Test if the function absSum() returns the right value in sampling window is sampling window equals 1."):
			image = np.asarray([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
			absoluteSum = fun.absSum(array=image, samplingW=1)

			trueAbsSum = [[1., 2., 3.],[4., 5., 6.], [7., 8., 9.]]
			self.assertTrue(np.all(np.equal(absoluteSum, trueAbsSum)))

		with self.subTest("Test if the function absSum() returns the right value is sampling window is sampling window equals 0."):
			image = np.asarray([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
			absoluteSum = fun.absSum(array=image, samplingW=0)

			trueAbsSum = [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]
			self.assertTrue(np.all(np.equal(absoluteSum, trueAbsSum)))

		with self.subTest("Test if the function absSum() returns the right value is sampling window if complex values in numpy array."):
			image = np.asarray([[1, 2, 3],[4, 5, 6],[7, 8, 9]], dtype=np.complex128)
			absoluteSum = fun.absSum(array=image, samplingW=1)

			trueAbsSum = [[1., 2., 3.],[4., 5., 6.],[7., 8., 9.]]
			self.assertTrue(np.all(np.equal(absoluteSum, trueAbsSum)))

		with self.subTest("Test sum and abs if negative value"):
			image = image = np.asarray([[-1, -2, -3],[-4, -5, -6],[-7, -8, -9]])
			absoluteSum = fun.absSum(array=image, samplingW=3)

			trueAbsSum = [[12., 21., 16.],[27., 45., 33.],[24., 39., 28.]]

	def testSquared(self):
		with self.subTest("Test if the function squared() returns the right value is values in numpy array are int."):
			image = np.asarray([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
			stdMean = fun.stDevAndMeanOnePixel(image=image, sw=3, pixel=[1,1])

			trueStdMean = [2.581988897471611, 5]
			self.assertTrue(np.all(np.equal(stdMean, trueStdMean)))

		with self.subTest("Test if returns the right values of stdev and mean for one pixel when negative values."):
			image = np.asarray([[-1, -2, -3],[-4, -5, -6],[-7, -8, -9]])
			stdMean = fun.stDevAndMeanOnePixel(image=image, sw=3, pixel=[1,1])

			trueStdMean = [2.581988897471611, -5]
			self.assertTrue(np.all(np.equal(stdMean, trueStdMean)))

		with self.subTest("Test if returns the right values of stdev and mean for one pixel when values are 0."):
			image = np.asarray([[0, 0, 0],[0, 0, 0],[0, 0, 0]])
			stdMean = fun.stDevAndMeanOnePixel(image=image, sw=3, pixel=[1,1])

			trueStdMean = [0, 0]
			self.assertTrue(np.all(np.equal(stdMean, trueStdMean)))

		with self.subTest("Test if returns the right values of stdev and mean of one pixel when values are complex."):
			image = np.asarray([[0+1j, 0+1j, 0+1j],[1+1j, 1+1j, 1+1j],[0+0j, 0+0j, 0+0j]])
			stdMean = fun.stDevAndMeanOnePixel(image=image, sw=3, pixel=[1,1])

			trueStdMean = [0.6666666666666666, 0.3333333333333333+0.6666666666666666j]
			self.assertTrue(np.all(np.equal(stdMean, trueStdMean)))

if __name__ == "__main__":
     unittest.main()



