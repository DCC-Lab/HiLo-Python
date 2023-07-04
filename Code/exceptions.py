import numpy as np

def isANumpyArray(parameter):
	if type(parameter) is not np.ndarray:
		raise Exception(f"{parameter} does not result in a numpy.ndarray.")

def isInt(parameter):
	if type(parameter) is not int:
		raise Exception(f"{parameter} must be an int.")

def isIntOrFloat(parameter):
	if type(parameter) is not int and type(parameter) is not float:
		raise Exception(f"{parameter} must be an int or a float.")

def isString(parameter):
	if type(parameter) is not str:
		raise Exception(f"{parameter} must be a string.")

def isTifOrTiff(path):
    pathExtention = path.split(".")[-1]
    if pathExtention != "tif" and pathExtention != "tiff":
        raise Exception("Image must be a .tif of .tiff")

def areValuesEqual(value1, value2):
	if value1 != value2:
		raise Exception("Values are different.")

def isSameShape(image1, image2):
	if image1.shape[0] != image2.shape[0]:
		raise Exception("Images don't have the same size.")
	elif image1.shape[1] != image2.shape[1]:
		raise Exception("Images don't have the same size.")

def isPixelATuple(pixel):
    if type(pixel) is not tuple and len(pixel) != 2:
        raise Exception("Pixel must be a tuple of two integers (starting from 0)!")
