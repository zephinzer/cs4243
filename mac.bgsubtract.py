#!/usr/bin/python
### dependencies
import os, sys
# may need to remove for windows systems
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2, numpy as np
from matplotlib import pyplot

class BackgroundSubtractor:
    imageBackground = None
    pathToVideo = None
    pathToResult = None
    def __init__(self, inputPath, imageBackground, outputPath):
        self.pathToVideo = inputPath
        self.imageBackground = imageBackground
        self.pathToResult = outputPath

    def readVideo(self, inputPath):
        cap = cv2.VideoCapture(inputPath)
        frameWidth = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        frameRate = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
        frameCount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        frameFourCC = int(cap.get(cv2.cv.CV_CAP_PROP_FOURCC))
        return cap, frameWidth, frameHeight, frameRate, frameCount, frameFourCC
    def run(self, fourCC):
        """Use \"MP42\" for .avi generation or \"mp4v\" for .mov files"""
        bgimage = cv2.imread(self.imageBackground)
        cap,w,h,r,fc,fcc = self.readVideo(self.pathToVideo)
        wrt = cv2.VideoWriter(self.pathToResult, cv2.cv.CV_FOURCC(*fourCC), r, (w,h))
        for fr in range(1, fc):
            print fr/float(fc)*100,'% completed'
            _, img = cap.read()
            frame = cv2.subtract(img, bgimage)
            wrt.write(frame)
        cap.release()
        wrt.release()

driver = BackgroundSubtractor('input-v-eq-final.avi','bg.png','input-no-bg.avi')
driver.run("MP42")
