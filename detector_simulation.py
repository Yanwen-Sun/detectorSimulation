from ast import literal_eval as make_tuple
from scipy import ndimage
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
import h5py as h5
import scipy
import math
import multiprocessing as mp
from time import time

"""
when photons are absorbed in silicon sensors and turns into electrons, there is a conversion rate.
Every 3.6 eV turns into 1 electron. For example, 360 eV photon will turn into 100 electrons and 7.2 keV
photon will turn into about 2000 electrons. The process of course is not fully deterministic and there
are many other sources of uncertainties and electron/charge loss mechanisms.
"""

# def makeSpeckle(size, speckleSize = 4, gridSize = [2048,2048], detectorSize = [64,64]):
#     '''
#     Output: a speckle pattern of size "size" with the given speckle size
#     '''
#     image = np.zeros(size)
#     speckleSize_grid = speckleSize*gridSize[0]/detectorSize[0]
#     xPhasors = int(np.round(size[0]/speckleSize_grid))
#     yPhasors = int(np.round(size[1]/speckleSize_grid))
#     rndPhasor = np.zeros(size)
#     rndPhasor[0:xPhasors,0:yPhasors] = np.exp(np.random.random([xPhasors,yPhasors])*2.j*np.pi)
#     speckleField = np.fft.fft2(rndPhasor)
#     speckleIntensity = np.abs(speckleField)**2
#     return speckleIntensity/np.mean(speckleIntensity.flatten())

def addIntensityVariation(speckle):
    x,y = np.indices((speckle.shape))
    b = y*1e-4+np.ones_like(x)
    return speckle*b

def makeSpeckle(size, speckleSize = 4, gridSize = [2048,2048], detectorSize = [64,64]):
    '''
    Output: a speckle pattern of size "size" with the given speckle size
    '''
    image = np.zeros(size)
    speckleSize_grid = speckleSize*gridSize[0]/detectorSize[0]
    xPhasors = int(np.round(size[0]/speckleSize_grid))
    yPhasors = int(np.round(size[1]/speckleSize_grid))
    rndPhasor = np.zeros(size)
    rndPhasor[0:xPhasors,0:yPhasors] = np.exp(np.random.random([xPhasors,yPhasors])*2.j*np.pi)
    speckleField = np.fft.fft2(rndPhasor)
    speckleIntensity = np.abs(speckleField)**2
    #speckleIntensity = addIntensityVariation(speckleIntensity)
    return speckleIntensity/np.mean(speckleIntensity.flatten())


def AddShotNoise(speckle, kbar):
    """
    generate speckle pattern as discrete photons by introducing shot noise.
    variable kbar is the average photon density over the field of view.
    
    Output: image of individual photon hits with probabilities given by the speckle
    """
    speckle = speckle/np.mean(speckle.flatten())*kbar;
    return np.random.poisson(speckle)



def lorentzianKernel2D(gamma,truncate,width):
    """
    generate a lorentzian kernel
    """
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * gamma + 0.5)
    x,y = np.indices((2*lw,2*lw))-lw
    weights = gamma/np.pi/2/(x**2+y**2+gamma**2)**1.5 
    weights = weights/weights.sum()
    return np.pad(weights,int((width-2*lw)/2),'constant')


def superGaussianKernel2D(sigma,truncate,width, n= 0.5):
    """
    generate a superGaussian kernel
    """
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sigma+0.5)
    x,y = np.indices((2*lw,2*lw))-lw
    x = np.abs(x)
    y = np.abs(y)
    weights = np.exp(-(x**n+y**n)/2/sigma**n)
    weights = weights/weights.sum()
    return np.pad(weights,int((width-2*lw)/2),'constant')

def mixedGaussianKernel2D(sigma,truncate,width):
    """
    generate a superGaussian kernel
    """
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sigma+0.5)
    x,y = np.indices((2*lw,2*lw))-lw
    x = np.abs(x)
    y = np.abs(y)
    n = 0.5
    weights = np.exp(-(x**n+y**n)/2/sigma**n)
    n = 2
    weights += np.exp(-(x**n+y**n)/2/sigma**n)
    weights = weights/weights.sum()
    return np.pad(weights,int((width-2*lw)/2),'constant')

def ApplyChargeCloud(photonPattern, cloudSize, photonEnergy,chargeCloudShape = "gaussian"):
    """
    this step blurs the photon map by the charge cloud size, assuming
    every electron in silicon takes 3.6 eV to generate.
    """
    
    if chargeCloudShape == 'gaussian':
        return ndimage.filters.gaussian_filter(photonPattern*photonEnergy/3.6, cloudSize/2.35, mode='wrap', truncate=7)
    
    if chargeCloudShape == 'lorentzian':
        kernel = lorentzianKernel2D(cloudSize/1.5,5
                                    ,np.shape(photonPattern)[0])
        fftKernel = np.fft.fft2(kernel)
        fftImage = np.fft.fft2(photonPattern*photonEnergy/3.6)
        fftProduct = np.multiply(fftKernel,fftImage)
        return np.abs(np.fft.ifft2(fftProduct))
    if chargeCloudShape == 'superGaussian':
        kernel = superGaussianKernel2D(cloudSize/2.35,7,np.shape(photonPattern)[0])
        fftKernel = np.fft.fft2(kernel)
        fftImage = np.fft.fft2(photonPattern*photonEnergy/3.6)
        fftProduct = np.multiply(fftKernel,fftImage)
        return np.abs(np.fft.ifft2(fftProduct))   
    if chargeCloudShape == 'mixedGaussian':
        kernel = mixedGaussianKernel2D(cloudSize/2.35,7,np.shape(photonPattern)[0])
        fftKernel = np.fft.fft2(kernel)
        fftImage = np.fft.fft2(photonPattern*photonEnergy/3.6)
        fftProduct = np.multiply(fftKernel,fftImage)
        return np.abs(np.fft.ifft2(fftProduct))   

def rebin(a, shape):
    """
    pattern rebinning down to smaller sizes
    by adding MXN blocks together. IDL type, pretty quick.
    """
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)
            

def digitizeCharge(chargePattern, gain, chargePerADU=15, gainMap=None):
    """
    digitize the charge pattern to ADUs, with gain variation
    """
    if gainMap is None:
        gainMap = np.random.randn(chargePattern.shape[0], chargePattern.shape[1])*gain+1
#     if gain == 0.05:
#         gainMap = np.load('gainMap0p05.npy')
#     if gain == 0.03:
#         gainMap = np.load('gainMap0p03.npy')
#     if gain == 0.02:
#         gainMap = np.load('gainMap0p02.npy')
#     if gain == 0.06:
#         gainMap = np.load('gainMap0p06.npy')
    return chargePattern*gainMap/chargePerADU

    
def AddReadoutNoise(chargePattern, pixelReadoutNoise):
    """
    add readout noise to the detector.
    """
    return chargePattern + np.random.randn(chargePattern.shape[0], chargePattern.shape[1])*pixelReadoutNoise

def photonPositions(photons):
    '''
    Get the positions of each photon in the photon image and return a list of [i,j] positions
    '''
    positions = []
    
    for i,row in enumerate(photons):
        array = np.asarray(row)
        for j in np.where(array > 0)[0]:
            elt = array[j]
            while elt != 0:
                positions.append([i,j])
                elt -= 1
    
    return positions

def makeWeakSpeckle(speckle, kbar = 0.01, gridSize=[1024,1024], detectorSize=[128,128], 
                    chargeCloudSize=0.2, photonEnergy=9500, readoutNoise=10,gain = 0.0,chargeCloudShape = "gaussian"):
    """
    use the functions in the earlier section to produce a single detector image of
    a weak speckle pattern with average signal rate of kbar.
    """
    #add gamma distribution to the scattering intensity to simulate the flux after mono
    #kbar = np.random.gamma(shape = 1.3, scale = kbar/1.3)
    kbar_grid = kbar/gridSize[0]/gridSize[1]*detectorSize[0]*detectorSize[1]
#     ccsholder = chargeCloudSize
#     chargeCloudSize = np.random.normal(loc=chargeCloudSize, scale=chargeCloudSize/5.)
#     if chargeCloudSize < 0: chargeCloudSize = ccsholder
    chargeCloudSize_grid = chargeCloudSize*gridSize[0]/detectorSize[0]
    speckleN = AddShotNoise(speckle, kbar_grid)
    position = photonPositions(speckleN)
    speckleNE = ApplyChargeCloud(speckleN, chargeCloudSize_grid, photonEnergy,chargeCloudShape)
    speckleShift = ndimage.interpolation.shift(speckleNE,[np.random.rand(),np.random.rand()],order = 5, mode='wrap')
    speckleD = rebin(speckleShift, detectorSize)
    speckleDD = digitizeCharge(speckleD, gain)
    return AddReadoutNoise(speckleDD, readoutNoise), position

def dropletNameBatcher(kbar,gridSize,speckleSize,detectorSize,chargeCloudSize,photonEnergy,readoutNoise,gain,contrast,chargeCloudShape = "gaussian"):
    '''
    Given all the parameters of the simulation it generates a name for the file that includes those parameters.
    This way it's easier to know what parameters you are using.
    '''
    return "k" + str(kbar) + "gs" + str(gridSize[0]) + "ss" + str(speckleSize) + "ds" + str(detectorSize[0]) + "ccs"+ str(chargeCloudSize) + "ccsh"+ chargeCloudShape.capitalize() +"pE" + str(photonEnergy) + "rn" + str(readoutNoise)+"gain"+str(gain)+"beta"+str(contrast)


def speckleNameBatcher(gridSize,speckleSize,detectorSize,contrast):
    '''
    Given all the parameters of the simulation it generates a name for the file that includes those parameters.
    This way it's easier to know what parameters you are using.
    '''
    return "gs" + str(gridSize[0]) + "ss" + str(speckleSize) + "ds" + str(detectorSize[0])+"beta"+str(contrast)

def PhotonHist(images,bins,PRINT = False):
    """generate a histogram from images"""
    histFinal = np.zeros(np.shape(bins)[0]-1)

    for i,image in enumerate(images):
        if np.mod(i,10) == 0 and PRINT:
            print(i)
        speckle = image
        histHere = np.histogram(speckle.flatten(), bins)
        histFinal += histHere[0]      
    return histFinal    

def DetectorImageBatch3Step(kbar,gridSize,detectorSize,chargeCloudSize,
                            photonEnergy,readoutNoise,gain,N,speckle,chargeCloudShape = "gaussian"):
    """generate N detector images"""
    tic = time()
    images = []
    positions = []
    
    np.random.seed() # Otherwise each thread gets the same pseudorandom number and you get the same plot 4 times

    for i in range(N):
        if np.mod(i,20) == 0:
            print(i)
        detectorOutput, position = makeWeakSpeckle(speckle,kbar,gridSize,detectorSize,chargeCloudSize,photonEnergy,readoutNoise,gain,chargeCloudShape)
        images.append(detectorOutput)
        positions.append(position)
    toc = time()
    print(toc-tic)
    return images, positions

def threadImages3Step(kbar,gridSize,detectorSize,chargeCloudSize,photonEnergy,readoutNoise, gain,N, cores,speckle,chargeCloudShape = "gaussian"):
    '''
    Distributes N images (imagesPerCore) to each of the cores
    '''
    pool = mp.Pool(cores)
    data = (kbar,gridSize,detectorSize,chargeCloudSize,photonEnergy,readoutNoise,gain,N,speckle,chargeCloudShape)
    result = pool.starmap(DetectorImageBatch3Step, [data for i in range(cores)])
    return result
