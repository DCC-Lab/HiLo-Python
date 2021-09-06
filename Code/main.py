import functions as fun
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

sigmaValue = 2

# Step 1 : Subtract uniform from speckled image to for the difference image

imgSpeckle = fun.image("/Users/valeriepineaunoel/Documents/HiLo-Python/Data/testImage.tif")
imgUniform = fun.image("/Users/valeriepineaunoel/Documents/HiLo-Python/Data/testImage2.tif")
imgDiff = imgSpeckle - imgUniform

imgSpeckle = fun.image("/Users/valeriepineaunoel/Documents/HiLo-Python/Data/samplespeckle.tif")
imgUniform = fun.image("/Users/valeriepineaunoel/Documents/HiLo-Python/Data/sampleuniform.tif")
fftUniform = np.fft.fft2(imgUniform)
imgDiff = fun.differenceImage(speckle=imgSpeckle, uniform=imgUniform)


# Step 2 : Frequency bandpass on the difference image. Adjusting its with to tune the width of the sectioning strength
## Image in frequency space. Sigma defines the width of the filter. 
imgDiffBP = fun.gaussianFilter(sigma=sigmaValue, image=imgDiff)
bandpassFilter = fun.gaussianFilterOfImage(filteredImage=imgDiffBP, differenceImage=imgDiff)
fftBandpassFilter = np.fft.fft2(bandpassFilter)

# # Step 3 : Evaluate the weigthing function (squared) according to equation StDev_diff/MeanIntensity_s
contrast = fun.contrastCalculation(difference=imgDiffBP, uniform=imgUniform, speckle=imgSpeckle, samplingWindow=7, ffilter=fftBandpassFilter) # va être utilisé pour produire le LP

# Step 4 : Removing noise-induced bias from C^2 by subtracting 
# stdevSpeckle, meanSpeckle = fun.stdevAndMeanWholeImage(image=imgSpeckle, samplingWindow=7)
# stdevUniform, meanUniform = fun.stdevAndMeanWholeImage(image=imgUniform, samplingWindow=7)
contrastSquared = fun.squaredFunction(contrast) # va être utilisé pour évaluer n


# Step 5 : construct LP and HP filters according to the W defined at step 2
# imgDiffFrequency = np.fft.fft2(imgDiff) # produit des nombres complexes. Ne peux pas être affiché à moins d'utiliser np.abs(fft)
# cutoff frequency des filtres = 0.18sigma du gaussian bandpass filter en step 3
# HP(k) = 1 - LP(k)
# Ilp = LP[C*Iu]
# Ihp = HP[Iu]

# Create the filters
lowFilter = fun.lowPassFilter(image=imgUniform, sigmaFilter=sigmaValue)
highFilter = fun.highPassFilter(low=lowFilter)

contrastUniform = contrast*imgUniform
fftContrastUniform = np.fft.fft2(contrastUniform)
fftLO = lowFilter*fftContrastUniform
LO = np.fft.ifft2(fftLO)

fftHI = highFilter*fftUniform
HI = np.fft.fft2(fftHI)

tiff.imshow(imgDiff)
plt.show()
tiff.imshow(imgUniform)
plt.show()
tiff.imshow(lowFilter)
plt.show()
tiff.imshow(highFilter)
plt.show()
tiff.imshow(contrastUniform)
plt.show()
tiff.imshow(LO)
# the LO image is weird and the HI image in reversed. À faire demain! 
plt.show()
tiff.imshow(HI)
plt.show()


# Step 6 : Evaluate the scaling function (seamless transition from low to high spatial frequencies)
## Évaluer n en fonction de l'équation 8 (C^2)

# Step 7 : Evaluate I_{HiLo}