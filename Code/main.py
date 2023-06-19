import functions as fun
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import skimage as ski
import time
import cProfile
import re

#cProfile.run('re.compile("foo|bar")')

start = time.time()

# Sigma defines the width of the filter. A greater sigmaValue decreases the optical sectioning in the resultant HiLo image. 
sigmaValue = 2

# Step 1 : Subtract uniform from speckled image to for the difference image
imgSpeckle = fun.createImage("/Users/valeriepineaunoel/Documents/HiLo-Python/Data/20210306-SpeckleRhodamineETL-NoETL-S-6-Cropped.tif")
imgUniform = fun.createImage("/Users/valeriepineaunoel/Documents/HiLo-Python/Data/20210306-SpeckleRhodamineETL-NoETL-U-6-Cropped.tif")
imgDiff = fun.createDifferenceImage(speckle=imgSpeckle, uniform=imgUniform)

imgHiLo = fun.createHiLoImage(uniform=imgUniform, speckle=imgSpeckle, sigma=sigmaValue, sWindow=3)
#tiff.imshow(imgHiLo)
#plt.show()

end = time.time()
print(f"Execution time is {end-start}")
