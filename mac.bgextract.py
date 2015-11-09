#!/usr/bin/python
### dependencies
import os, sys
# may need to remove for windows systems
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2, numpy as np
from matplotlib import pyplot

class BackgroundExtractor:
    pathToVideo = None
    pathToResult = None
    def __init__(self, videoPath, outputPath):
        self.pathToVideo = videoPath
        self.pathToResult = oututPath
    def run(self):
        cap = cv2.VideoCapture(self.pathToVideo)
        fcount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        _, img = cap.read()
        avgImg = np.float32(img)
        for fr in range(1, fcount):
            _, img = cap.read()
            img = np.float32(img)
            avgImg = np.add(np.multiply(avgImg, fr/(fr+1.0)), np.divide(img, fr))
        normImg = cv2.convertScaleAbs(avgImg) # convert into uint8 image
        cap.release()
        cv2.imwrite(pathToResult, normImg)

driver = BackgroundExtractor('input.avi', 'bg.png')
driver.run()
