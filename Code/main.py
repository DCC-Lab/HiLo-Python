import functions as fun
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

# Sigma defines the width of the filter.
sigmaValue = 1

# Step 1 : Subtract uniform from speckled image to for the difference image
imgSpeckle = fun.createImage("/Users/valeriepineaunoel/Documents/HiLo-Python/Data/20210306-SpeckleRhodamineETL-NoETL-S-6-Cropped.tif")
imgUniform = fun.createImage("/Users/valeriepineaunoel/Documents/HiLo-Python/Data/20210306-SpeckleRhodamineETL-NoETL-U-6-Cropped.tif")
imgDiff = fun.createDifferenceImage(speckle=imgSpeckle, uniform=imgUniform)


# Step 2 : Frequency bandpass on the difference image. Adjusting its with to tune the width of the sectioning strength
## Image in frequency space.
# imgDiffBP = fun.gaussianFilter(sigma=sigmaValue, image=imgDiff)
# gaussImage = fun.obtainFFTFitler(image=imgUniform, filteredImage=imgDiffBP)

# Step 3 : Evaluate the weigthing function (squared) according to equation StDev_diff/MeanIntensity_s
# Step 4 : Removing noise-induced bias from C^2 by subtracting 
print("I'm calculating the contrast")
contrast = fun.contrastCalculation(uniform=imgUniform, speckle=imgSpeckle, samplingWindow=3, sigma=sigmaValue) # va être utilisé pour produire le LP

# Create the filters
lowFilter = fun.lowPassFilter(image=imgUniform, sigmaFilter=sigmaValue)
highFilter = fun.highPassFilter(low=lowFilter)

cxu = contrast*imgUniform
print("CXU : {}{}".format(cxu, cxu.dtype))
cxuuint16 = cxu.astype(np.uint16)
print("CXU UINT16 : {}{}".format(cxuuint16, cxuuint16.dtype))

# Apply the low-pass frequency filter on the uniform image to create the LO portion
# Ilp = LP[C*Iu]
LO = np.absolute(np.fft.ifft2(lowFilter*(np.fft.fft2(cxuuint16))))
print("LO : {}{}".format(LO, LO.dtype))

# Apply the high-pass frequency filter to the uniform image to obtain the HI portion
# Ihp = HP[Iu]
HI = np.absolute(np.fft.ifft2(highFilter*np.fft.fft2(imgUniform)))
print("HI : {}{}".format(HI, HI.dtype))

# Step 6 : Evaluate the scaling function (seamless transition from low to high spatial frequencies)
## Évaluer n en fonction de l'équation 8 (C^2)
etaValue = fun.estimateEta(speckle=imgSpeckle, uniform=imgUniform, sigma=sigmaValue)
print("ETA : {etaValue}")

# Step 7 : Evaluate I_{HiLo}
imageHiLo = eta*LO + HI
print("HILO : {}{}".format(imageHiLo, imageHiLo.dtype))
tiff.imshow(imageHiLo)
plt.show()




