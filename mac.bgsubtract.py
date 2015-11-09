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
    def equalizeValues(self, img):
        h = np.zeros([len(img), len(img[0])])
        s = np.zeros([len(img), len(img[0])])
        v = np.zeros([len(img), len(img[0])])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.split(hsv, [h,s,v])
        v = cv2.equalizeHist(cv2.convertScaleAbs(v)).astype('float32')
        s = s.astype('float32')
        h = h.astype('float32')
        cv2.merge([h,s,v], hsv)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img
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
        for fr in range(1, 100):
            _, img = cap.read()
            #mask = fgmask.apply(img, mask, 1)
            frame = cv2.subtract(img, bgimage)
            self.equalizeValues(frame)
            frame = cv2.filter2D(frame, -1, np.ones((3,3),np.float32)/8)
            wrt.write(frame)
        cap.release()
        wrt.release()

driver = BackgroundSubtractor('input.avi','bg.png','input-no-bg.mov')
driver.run("mp4v")
