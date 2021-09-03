import functions as fun
import matplotlib.pyplot as plt

# Step 1 : Subtract uniform from speckled image to for the difference image
imgSpeckle = fun.image("/Users/valeriepineaunoel/Desktop/samplestructured(forspeckleplugin).tif")
imgUniform = fun.image("/Users/valeriepineaunoel/Desktop/sampleuniform(forspeckleplugin).tif")
imgDiff = imgSpeckle - imgUniform

# Step 2 : Frequency bandpass on the difference image. Adjusting its with to tune the width of the sectioning strength
## Image in frequency space. Sigma defines the width of the filter. 
imgDiffBP = fun.gaussianFilter(sigma=2, image=imgDiff)

# Step 3 : Evaluate the weigthing function (squared) according to equation StDev_diff/MeanIntensity_s
contrast = fun.contrastCalculation(difference=imgDiffBP, uniform=imgUniform, samplingWindow=7)
contrastWeightingFunction = fun.contrastCalculationSquared(contrast)

# Step 4 : Removing noise-induced bias from C^2 by subtracting 
## Calculating stDev of the uniform image


# Step 5 : construct LP and HP filters according to the W defined at step 2
## imgDiffFrequency = np.fft.fft2(imgDiff) # produit des nombres complexes. Ne peux pas être affiché à moins d'utiliser np.abs(fft)

# Step 6 : Evaluate the scaling function (seamless transition from low to high spatial frequencies)
# Step 7 : Evaluate I_{HiLo}








