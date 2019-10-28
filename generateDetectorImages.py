import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import sys
from tqdm import tqdm
from detector_simulation import *


#M is the inverse of contrast
M = 1
contrast = 1/float(M)
gridSize = [2048,2048]
speckleSize = 4
detectorSize=[64,64]

title = speckleNameBatcher(gridSize,speckleSize,detectorSize,contrast)
with h5.File('speckles/'+str(title)+'.hdf5','r') as h:
    speckle = h["speckle"][:]
    gridSize = h["gridSize"][:][0]
    speckleSize = h["speckleSize"][:][0]
    detectorSize = h["detectorSize"][:][0]


# Parameters for the images
kbar = 0.01
chargeCloudSize= 0.385
photonEnergy=8200
readoutNoise= 3.0
chargeCloudShape = "gaussian"
gain = 0.005

nframes = 10
imgs = np.zeros((nframes,detectorSize[0],detectorSize[1]))
for i in tqdm(np.arange(nframes)):
    imgs[i],_ = makeWeakSpeckle(speckle, kbar = kbar, gridSize=gridSize, detectorSize=detectorSize, chargeCloudSize=chargeCloudSize, photonEnergy=8200, readoutNoise=readoutNoise,gain = gain,chargeCloudShape = chargeCloudShape)


title = dropletNameBatcher(kbar,gridSize,speckleSize,\
                               detectorSize,chargeCloudSize,photonEnergy,readoutNoise,gain,contrast,chargeCloudShape)
with h5.File(title,'w') as f:
    f.create_dataset('imgs',data = imgs)
    f.create_dataset('kbar',data = kbar)
    f.create_dataset('photonEnergy',data = photonEnergy)
    f.create_dataset('gain',data = gain)
    f.create_dataset('chargeCloudShape',data = chargeCloudShape)
    f.create_dataset('chargeCloudSize',data = chargeCloudSize)
    f.create_dataset('readoutNoise',data = readoutNoise)









    
