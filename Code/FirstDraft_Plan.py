import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import scipy.ndimage as simg

# Step 1 : Subtract uniform from speckled image to for the difference image
imgSpeckle = tiff.imread(r"/Users/valeriepineaunoel/Desktop/samplestructured(forspeckleplugin).tif") 
imgUniform = tiff.imread(r"/Users/valeriepineaunoel/Desktop/sampleuniform(forspeckleplugin).tif") 

# plt.imshow(imgSpeckle)
# plt.show()
# plt.imshow(imgUniform)
# plt.show()

## Verify the size of both images for fun
# heightS, weightS = imgSpeckle.shape
# heightU, weightU = imgUniform.shape 
# print(weightS, heightS)
# print(weightU, heightU)
# print(imgSpeckle.shape[1])


## Difference image
imgDiff = imgSpeckle - imgUniform
# print(type(imgDiff))
# plt.imshow(imgDiff)
# plt.show()

# Step 2 : Frequency bandpass on the difference image. Adjusting its with to tune the width of the sectioning strength
## Image in frequency space. Sigma defines the width of the filter. 
imgDiffFrequency = np.fft.fft2(imgDiff) # produit des nombres complexes. Ne peux pas être affiché à moins d'utiliser np.abs(fft)
imgDiffBandpass = simg.gaussian_filter(imgDiff, sigma=2)

# imgDiffFrequencyAbs = np.abs(imgDiffFrequency)
# plt.imshow(imgDiffFrequencyAbs) # display des nombres complexes
# plt.show()
# plt.imshow(imgDiffFrequencyAbs**2) # display le power spectrum
# plt.show()
# plt.imshow(imgDiffBandpass)
# plt.show()


# Step 3 : Evaluate the weigthing function (squared) according to equation StDev_diff/MeanIntensity_s
samplingWindow = 7 # in pixel^2
n = int(samplingWindow/2)
pixel = [0,0]
positionInSamplingWindow = [pixel[0]-n, pixel[1]-n]
if positionInSamplingWindow[0] < 0:
	positionInSamplingWindow[0] = 0
if positionInSamplingWindow[1] < 0: 
	positionInSamplingWindow[1] = 0

# Calculations in one sampling window
valuesImgDiff = []
valuesImgUniform = []

while positionInSamplingWindow[0] <= pixel[0]+n and positionInSamplingWindow[0] <= imgDiffBandpass.shape[0]:
	while positionInSamplingWindow[1] <= pixel[1]+n and positionInSamplingWindow[1] <= imgDiffBandpass.shape[1]:
		valuePixelDiff = imgDiffBandpass[positionInSamplingWindow[0]][positionInSamplingWindow[1]]
		
		if valuePixelDiff < 0:
			valuePixelDiff = 0
		valuesImgDiff.append(valuePixelDiff)
		valuePixelUniform = imgUniform[positionInSamplingWindow[0]][positionInSamplingWindow[1]]
		valuesImgUniform.append(valuePixelUniform)
		positionInSamplingWindow[1] += 1
		
	positionInSamplingWindow[1] = 0
	positionInSamplingWindow[0] = positionInSamplingWindow[0] + 1

contrastInSamplingWindow = np.std(valuesImgDiff)/np.mean(valuesImgUniform)
print("ALLO {}".format(contrastInSamplingWindow))

# Step 4 : Removing noise-induced bias from C^2 by subtracting 
# Step 5 : construct LP and HP filters according to the W defined at step 2
# Step 6 : Evaluate the scaling function (seamless transition from low to high spatial frequencies)
# Step 7 : Evaluate I_{HiLo}








