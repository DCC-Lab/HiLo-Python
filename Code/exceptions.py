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
	tifEnd = path[-3] + path[-2] + path[-1]
	tiffEnd = path[-4] + path[-3] + path[-2] + path[-1]
	if tifEnd != "tif" and tiffEnd != "tiff":
		raise Exception("Image must be a .tif of .tiff")

def areValuesEqual(value1, value2):
	if value1 != value2:
		raise Exception("Values are different.")

def isSameShape(image1, image2):
	if image1.shape[0] != image2.shape[0]:
		raise Exception("Images don't have the same size.")
	elif image1.shape[1] != image2.shape[1]:
		raise Exception("Images don't have the same size.")