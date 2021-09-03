import functions as fun
import matplotlib.pyplot as plt
import numpy as np

# Step 1 : Subtract uniform from speckled image to for the difference image
imgSpeckle = fun.image("/Users/valeriepineaunoel/Documents/HiLo-Python/Data/testImage.tif")
imgUniform = fun.image("/Users/valeriepineaunoel/Documents/HiLo-Python/Data/testImage2.tif")
imgDiff = imgSpeckle - imgUniform

# Step 2 : Frequency bandpass on the difference image. Adjusting its with to tune the width of the sectioning strength
## Image in frequency space. Sigma defines the width of the filter. 
imgDiffBP = fun.gaussianFilter(sigma=2, image=imgDiff)
bandpassFilter = imgDiffBP - imgDiff
fftBandpassFilter = np.fft.fft2(bandpassFilter)

# Step 3 : Evaluate the weigthing function (squared) according to equation StDev_diff/MeanIntensity_s
contrast = fun.contrastCalculation(difference=imgDiffBP, uniform=imgUniform, speckle=imgSpeckle, samplingWindow=7, ffilter=fftBandpassFilter) # va être utilisé pour produire le LP
print("ALLO : {}".format(contrast))
contrastWeightingFunction = fun.squaredFunction(contrast) # va être utilisé pour évaluer n
print(contrastWeightingFunction)

# Step 4 : Removing noise-induced bias from C^2 by subtracting 
# stdevSpeckle, meanSpeckle = fun.stdevAndMeanWholeImage(image=imgSpeckle, samplingWindow=7)
# stdevUniform, meanUniform = fun.stdevAndMeanWholeImage(image=imgUniform, samplingWindow=7)

# Step 5 : construct LP and HP filters according to the W defined at step 2
## imgDiffFrequency = np.fft.fft2(imgDiff) # produit des nombres complexes. Ne peux pas être affiché à moins d'utiliser np.abs(fft)
## cutoff frequency des filtres = 0.18sigma du gaussian bandpass filter en step 3
## HP(k) = 1 - LP(k)
## Ilp = LP[C*Iu]
## Ihp = HP[Iu]

# Step 6 : Evaluate the scaling function (seamless transition from low to high spatial frequencies)
## Évaluer n en fonction de l'équation 8 (C^2)

# Step 7 : Evaluate I_{HiLo}