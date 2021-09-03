import numpy as np

def isANumpyArray(parameter):
	if type(parameter) is not np.ndarray:
		raise Exception(f"{parameter} does not result in a numpy.ndarray.")

def isDefined(parameter):
	if parameter is None:
		raise Exception(f"{parameter} must be defined.")

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