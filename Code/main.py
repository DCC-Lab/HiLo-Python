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
# contrastSquared = fun.squaredFunction(contrast) # va être utilisé pour évaluer n
# print("TYPE DIFF contrast : {}".format(type(contrast[0][0])))
# print("TYPE DIFF contrast : {}".format(type(contrast)))
# print("TYPE OF contrastSquared : {}".format(type(contrastSquared[0][0])))
# print("TYPE OF contrastSquared : {}".format(type(contrastSquared)))

# # Step 5 : construct LP and HP filters according to the W defined at step 2
# # imgDiffFrequency = np.fft.fft2(imgDiff) # produit des nombres complexes. Ne peux pas être affiché à moins d'utiliser np.abs(fft)
# # cutoff frequency des filtres = 0.18sigma du gaussian bandpass filter en step 3
# # HP(k) = 1 - LP(k)
# # Ilp = LP[C*Iu]
# # Ihp = HP[Iu]

# # Create the filters
# lowFilter = fun.lowPassFilter(image=imgUniform, sigmaFilter=sigmaValue)
# highFilter = fun.highPassFilter(low=lowFilter)

# # Apply the low-pass frequency filter on the uniform image to create the LO portion
# # Ilp = LP[C*Iu]
# LO = np.fft.ifft2(lowFilter*(np.fft.fft2(contrast*imgUniform)))

# # Apply the high-pass frequency filter to the uniform image to obtain the HI portion
# # Ihp = HP[Iu]
# HI = np.fft.fft2(highFilter*fftUniform)

# tiff.imshow(imgDiff)
# plt.show()
# tiff.imshow(imgUniform)
# plt.show()
# tiff.imshow(lowFilter)
# plt.show()
# tiff.imshow(highFilter)
# plt.show()
# tiff.imshow(contrast*imgUniform)
# plt.show()
# tiff.imshow(LO)
# # the LO image is weird and the HI image in reversed. À faire demain! 
# plt.show()
# tiff.imshow(HI)
# plt.show()


# # Step 6 : Evaluate the scaling function (seamless transition from low to high spatial frequencies)
# ## Évaluer n en fonction de l'équation 8 (C^2)

# # Step 7 : Evaluate I_{HiLo}