# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import cv2
import pyamg
from skimage.filters import laplace
# Read source, target and mask for a given id
def Read(id, path = ""):
    source = plt.imread(path + "source_" + id + ".jpg")
    # source = plt.imread(path + "source_01.jpg")
    info = np.iinfo(source.dtype) # get information about the image type (min max values)
    source = source.astype(np.float32) / info.max # normalize the image into range 0 and 1
    target = plt.imread(path + "target_" + id + ".jpg")
    # target = plt.imread(path + "target_01.jpg")
    info = np.iinfo(target.dtype) # get information about the image type (min max values)
    target = target.astype(np.float32) / info.max # normalize the image into range 0 and 1
    mask   = plt.imread(path + "mask_" + id + ".jpg")
    # mask   = plt.imread(path + "mask_01.jpg")
    info = np.iinfo(mask.dtype) # get information about the image type (min max values)
    mask = mask.astype(np.float32) / info.max # normalize the image into range 0 and 1

    return source, mask, target

# Adjust parameters, source and mask for negative offsets or out of bounds of offsets
def AlignImages(mask, source, target, offset):
    sourceHeight, sourceWidth, _ = source.shape
    targetHeight, targetWidth, _ = target.shape
    xOffset, yOffset = offset
    
    if (xOffset < 0):
        mask    = mask[abs(xOffset):, :]
        source  = source[abs(xOffset):, :]
        sourceHeight -= abs(xOffset)
        xOffset = 0
    if (yOffset < 0):
        mask    = mask[:, abs(yOffset):]
        source  = source[:, abs(yOffset):]
        sourceWidth -= abs(yOffset)
        yOffset = 0
    # Source image outside target image after applying offset
    if (targetHeight < (sourceHeight + xOffset)):
        sourceHeight = targetHeight - xOffset
        mask    = mask[:sourceHeight, :]
        source  = source[:sourceHeight, :]
    if (targetWidth < (sourceWidth + yOffset)):
        sourceWidth = targetWidth - yOffset
        mask    = mask[:, :sourceWidth]
        source  = source[:, :sourceWidth]
    
    maskLocal = np.zeros_like(target)
    maskLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = mask
    sourceLocal = np.zeros_like(target)
    sourceLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = source

    return sourceLocal, maskLocal

# Pyramid Blend
def PyramidBlend(source, mask, target):
    
    return source * mask + target * (1 - mask)

def maskCheck(mask):
    if mask.ndim == 3:
        mask = np.any(mask, axis=-1).astype(np.uint8)
    return mask

def sparseMatrix(sizeImg, mask):
    sparseMatrix = scipy.sparse.identity(np.prod(sizeImg), format='lil')
    for y in range(sizeImg[0]):
        for x in range(sizeImg[1]):
            if mask[y,x]:
                index = x+y*sizeImg[1]
                sparseMatrix[index, index] = 4
                if index+1 < np.prod(sizeImg):
                    sparseMatrix[index, index+1] = -1
                if index-1 >= 0:
                    sparseMatrix[index, index-1] = -1
                if index+sizeImg[1] < np.prod(sizeImg):
                    sparseMatrix[index, index+sizeImg[1]] = -1
                if index-sizeImg[1] >= 0:
                    sparseMatrix[index, index-sizeImg[1]] = -1

    sparseMatrix = sparseMatrix.tocsr()

    return sparseMatrix
# PoissonBlend(source, mask, target, offsets[index])
def PoissonBlend(source, mask, target):
    # do I need this? (not sure)
    # have images up so that the blend can occur correctly 
    # ex. doesnt get blended incorrectly since one will be larger than the other
    sourceImg = (min(target.shape[0], source.shape[0]), min(target.shape[1], source.shape[1]))
    targetImg = (min(target.shape[0], source.shape[0]), min(target.shape[1], source.shape[1]))

    sizeImg = (sourceImg[0], sourceImg[1])

    mask = mask[0:sourceImg[0], 0:sourceImg[1]]
    mask = maskCheck(mask)
    mask[mask==0] = False
    mask[mask!=False] = True

    # create sparse matrix to use later
    sparMatrix = sparseMatrix(sizeImg, mask)
   
    # poisson matrix in order to findthe value of p / b
    P = pyamg.gallery.poisson(mask.shape)

    for i in range(target.shape[2]):
        targetFlat = target[0:targetImg[0],0:targetImg[1],i]
        sourceFlat = source[0:sourceImg[0], 0:sourceImg[1],i]
        targetFlat = targetFlat.flatten()
        sourceFlat = sourceFlat.flatten()

        # finds b in order to solve for Ax = b 
        b = P * sourceFlat
        for y in range(sizeImg[0]):
            for x in range(sizeImg[1]):
                if not mask[y,x]:
                    currVal = x+y*sizeImg[1]
                    b[currVal] = targetFlat[currVal]

        # use pyamg to solve the Ax = b stuff
        # couldnt get other method to work - this does?
        newShape = pyamg.solve(sparMatrix,b,verb=False)

        # newShape to target image in order for the size to be continuous 
        newShape = np.reshape(newShape , sizeImg)
        newShape = np.clip(newShape, 0, 255)
        target[0:targetImg[0],0:targetImg[1],i] = newShape

    return target

if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'
    
    # False for source gradient, true for mixing gradients
    isMix = False

    # Source offsets in target
    offsets = [[210, 10], [10, 28], [140, 80], [-40, 90], [60, 100], [20, 20], [-28, 88]]

    # main area to specify files and display blended image
    for index in range(len(offsets)):
        # Read data and clean mask
        source, maskOriginal, target = Read(str(index+1).zfill(2), inputDir)

        # Cleaning up the mask
        mask = np.ones_like(maskOriginal)
        mask[maskOriginal < 0.5] = 0

        # Align the source and mask using the provided offest
        source, mask = AlignImages(mask, source, target, offsets[index])

        
        ### The main part of the code ###
    
        # Implement the PoissonBlend function
        poissonOutput = PoissonBlend(source, mask, target)
        poissonOutput = np.clip(poissonOutput, 0, 1)

        
        # Writing the result
                
        if not isMix:
            plt.imsave("{}poisson_{}.jpg".format(outputDir, str(index+1).zfill(2)), poissonOutput)
        else:
            plt.imsave("{}poisson_{}_Mixing.jpg".format(outputDir, str(index+1).zfill(2)), poissonOutput)
