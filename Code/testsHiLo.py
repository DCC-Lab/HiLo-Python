import unittest 
import numpy as np
import functions as fun

class TestHiLo(unittest.TestCase):
	def testPositiveRescaling(self):
		with self.subTest("Test subtraction of two images that results in only positive values"): # Message that I was the user to see while the tests are running
			image1 = np.asarray([[20, 20, 20],[20, 20, 20], [20, 20, 20]])
			image2 = np.asarray([[3, 3, 3],[3, 3, 3], [3, 3, 3]])
			differenceImage = fun.createDifferenceImage(speckle=image1, uniform=image2)
			imgGroundTruth = np.asarray([[17, 17, 17],[17, 17, 17], [17, 17, 17]])
			self.assertTrue(np.all(np.equal(differenceImage, imgGroundTruth))) # np.all True or False of the overall array

	def testNegativeRescaling(self):
		with self.subTest("Test subtraction two images that results with some negative values"):
			image1 = np.asarray([[1, 1, 1],[1, 1, 1],[1, 1, 1]])
			image2 = np.asarray([[10, 10, 10],[10, 10, 10],[10, 10, 10]])
			differenceImage = fun.createDifferenceImage(speckle=image1, uniform=image2)
			imgGroundTruth = np.asarray([[0, 0, 0],[0, 0, 0], [0, 0, 0]])
			self.assertTrue(np.all(np.equal(differenceImage, imgGroundTruth)))
	
	def testZeroRescaling(self):
		with self.subTest("Test subtraction two images that results with some values equal to 0"):
			image1 = np.asarray([[1, 1, 1],[1, 1, 1],[1, 1, 1]])
			image2 = np.asarray([[1, 1, 1],[1, 1, 1],[1, 1, 1]])
			differenceImage = fun.createDifferenceImage(speckle=image1, uniform=image2)
			imgGroundTruth = np.asarray([[0, 0, 0],[0, 0, 0], [0, 0, 0]])
			self.assertTrue(np.all(np.equal(differenceImage, imgGroundTruth)))

if __name__ == "__main__":
     unittest.main()



