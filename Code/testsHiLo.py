import unittest 
import numpy as np
import functions as fun

class TestHiLo(unittest.TestCase):
	def testDifferenceImage(self):
		with self.subTest("Test subtraction of two images"): # Message that I was the user to see while the tests are running
			image1 = np.asarray([[20, 20, 20],[20, 20, 20], [20, 20, 20]])
			image2 = np.asarray([[3, 3, 3],[3, 3, 3], [3, 3, 3]])
			differenceImage = fun.createDifferenceImage(speckle=image1, uniform=image2)
			print(f"Diff image : {differenceImage}")
			imgGroundTruth = np.asarray([[17, 17, 17],[17, 17, 17], [17, 17, 17]])
			self.assertTrue(np.all(np.equal(differenceImage, imgGroundTruth))) # np.all True or False of the overall 


if __name__ == "__main__":
     unittest.main()




