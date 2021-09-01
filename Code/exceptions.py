def isANumpyArray(parameter):
	if isinstance(parameter,(list, np.ndarray)):
		pass
	else:
		raise Exception("{parameter} does not result in a numpy.ndarray.")