import sys
import cv2 as cv
import numpy as np
import time

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from sklearn import svm
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import json
import pickle

# Input: RGB Image, binary classifier, window parameters
# The classifier has to give a class output of 1 for the positives 
# windowSize: the number of px on the side of the square window. The window's surface is thus windowSize^2.
# windowStepX,Y: the amount of px the window shifts horiz between two shots. And vertical shift when reaching the right side of the image 
# Output: A binary image (gray value 0 or 255) of all windows recognised as positive by the SVM classifier
# The windows are sliding from top left to bottom right, horizontally first, then vertically.
def slidingWindows(image, classifier, windowSize, windowStepX, windowStepY, HOGCellSize, HOGHistoSize, HOGBlockCellSide):
    greyImg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    [imgHeight, imgWidth] = greyImg.shape
    outputMask = np.zeros(greyImg.shape)
    heatmap = np.zeros(greyImg.shape)

    # First iterating to get all the image windows
    # The position of the window is the coordinate of the upper left corner
    for windowYPos in range(1, imgHeight-windowSize+1, windowStepY):
        for windowXPos in range(1, imgWidth-windowSize+1, windowStepX):
            # Attribute a class to the window with classifier model
            window = greyImg[windowYPos:windowYPos+windowSize,windowXPos:windowXPos+windowSize]

            # Get HOG features vector
            features = getHOGFeatures(window, HOGCellSize, HOGHistoSize, HOGBlockCellSide)
            windowClass = classifier.predict([features])
            if windowClass == 1:
                print("Larva detected at " + str([windowYPos,windowXPos]))
                
                bottomLeftCorner = [min(windowYPos+windowSize, imgWidth), windowXPos]
                bottomRightCorner = [min(windowYPos+windowSize, imgHeight), min(windowXPos+windowSize, imgWidth)]
                topRightCorner = [windowYPos, min(windowXPos+windowSize, imgWidth)]
                topLeftCorner = [windowYPos,windowXPos]
                # Filled in rectangle
                outputMask[windowYPos:bottomRightCorner[0], windowXPos:bottomRightCorner[0]] = 255
                heatmap[windowYPos:bottomRightCorner[0], windowXPos:bottomRightCorner[0]] = heatmap[windowYPos:bottomRightCorner[0], windowXPos:bottomRightCorner[0]]+1

    # Normalize heatmap from 0 to max to 0 to 255
    # normalizedHeatmap = cv.normalize(heatmap, None, 0, 255, cv.NORM_MINMAX, dtype=np.uint8)
    # return normalizedHeatmap
    return outputMask.astype(np.uint8)

# Takes an gray image in input, and return the hog features vector
def getHOGFeatures(img, cellSize, histoSize, blockCellSide):
    # Compute HOG from SKImage
    features,hog_image = hog(img, orientations=histoSize, pixels_per_cell=(cellSize,cellSize),cells_per_block=(blockCellSide, blockCellSide),block_norm= 'L2',visualize=True)
    return features

def main():
    # Get config file name
    if len(sys.argv) >= 2:
        config_file = sys.argv[1]
    else:
        config_file = "config_green_slider_dataset.json"

    # Load config file
    with open(config_file) as f:
        config = json.load(f)
        imageToProcessPath = config['image_to_process']
        classifierFilePath = config['classifier_model_file']
        windowSize = config['parameters']['window_size']
        wndwStepX = config['parameters']['window_step_x']
        wndwStepY = config['parameters']['window_step_y']
        cellSize = config['hog_algo_params']['cell_size']
        blockCellSide = config['hog_algo_params']['block_cell_side']
        histogramSize = config['hog_algo_params']['histogram_size']

    # ------Load classifier model------
    classifier = pickle.load(open(classifierFilePath, 'rb'))

    # Load image
    image = cv.imread(imageToProcessPath)
    print("Image shape: " + str(image.shape))

    # ------Run sliding windows------
    startTime = time.time()
    windowsMask = slidingWindows(image, classifier, windowSize, wndwStepX, wndwStepY
        , cellSize, histogramSize, blockCellSide)
    duration = time.time() - startTime
    print("Processing time: " + str(duration))

    # Display image with windows
    # Display the initial image with half brightness except where the larvae were identified
    # Get the initial image masked by the mask where the larvae were identified
    larvaeHighlights = cv.bitwise_and(image,image, mask=windowsMask)
    # Mask the initial image with half brightness with the opposite mask
    maskOpposite = cv.bitwise_not(windowsMask)
    halfBrightnessImage = (image*0.5).astype(np.uint8)
    opposedMaskedImage = cv.bitwise_and(halfBrightnessImage,halfBrightnessImage, mask=maskOpposite)
    # Combine both masked images
    larvaeHighligthedImage = cv.bitwise_or(opposedMaskedImage, larvaeHighlights)

    cv.namedWindow("Display", cv.WINDOW_NORMAL)
    cv.setWindowProperty('Display ', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow("Display", windowsMask)


# main
main()
cv.waitKey(0)
