import numpy as np
import h5py
import os
from detector_simulation import *

directory = "speckles/"
if not os.path.exists(directory):
    os.makedirs(directory)
# Parameters for the images
gridSize=[2048,2048]
speckleSize = 4
detectorSize=[64,64]
Ms = np.power(2,np.arange(8))
#contrast is 1/M
for M in Ms:
    for i in np.arange(M):
        if i == 0:
            speckle =  makeSpeckle(size = [2048,2048], speckleSize = speckleSize, gridSize = gridSize, detectorSize = detectorSize)
        else:
            speckle +=  makeSpeckle(size = [2048,2048], speckleSize = speckleSize, gridSize = gridSize, detectorSize = detectorSize)
    contrast = 1/float(M)
    title = speckleNameBatcher(gridSize,speckleSize,detectorSize,contrast)
    print (title)
    with h5py.File(directory+str(title)+'.hdf5','w') as h:
        h.create_dataset("speckle", data = speckle)
        h.create_dataset("contrast",data = contrast)
        h.create_dataset("gridSize", data = np.asarray([gridSize]))
        h.create_dataset("speckleSize", data = np.asarray([speckleSize]))
        h.create_dataset("detectorSize", data = np.asarray([detectorSize]))

