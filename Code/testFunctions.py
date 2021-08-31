import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

def isANumpyArray(image):
	if isinstance(image,(list, np.ndarray)):
		pass
	else:
		raise Exception("Image does not result in a numpy array.")