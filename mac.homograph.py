#!/usr/bin/python
### dependencies
import os, sys
# may need to remove for windows systems
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2, numpy as np
from matplotlib import pyplot

class ClicksCaptor:
    FIELD_DISPLAY_NAME = 'Field'
    coords = []
    nClicks = 0

    ## get the videoCoords with Clicks
    def getClick(self, e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            print x, y
            self.coords.append([float(x), float(y)])
            print self.nClicks
            self.nClicks += 1
            if(self.nClicks == 4):
                cv2.destroyWindow(self.FIELD_DISPLAY_NAME)

    def getCoordsByClick(self, image):
        cv2.imshow(self.FIELD_DISPLAY_NAME, image)
        cv2.setMouseCallback(self.FIELD_DISPLAY_NAME, self.getClick, 0)
        #  Press "Escape button" to exit
        while True:
            key = cv2.waitKey(10) & 0xff
            if key == 27 or self.nClicks >= 4:
                break

def readVideo(inputPath):
    cap = cv2.VideoCapture(inputPath)
    frameWidth = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    frameRate = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    frameCount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    frameFourCC = int(cap.get(cv2.cv.CV_CAP_PROP_FOURCC))
    return cap, frameWidth, frameHeight, frameRate, frameCount, frameFourCC

i,w,h,r,c,cc = readVideo('input.avi')
print '[WIDTH] ', w
print '[HEIGHT]', h
print '[RATE]  ', r
print '[FRAMES]', c
print '[FOURCC]', cc

o = extractBackground(i,c)

class ClicksCaptor:
    FIELD_DISPLAY_NAME = 'Field'
    coords = []
    nClicks = 0
    nMaxClicks = None

    def __init__(self, nMaxClicks):
        self.nMaxClicks = nMaxClicks

    ## get the videoCoords with Clicks
    def getClick(self, e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            print x, y
            self.coords.append([float(x), float(y)])
            print self.nClicks
            self.nClicks += 1
            if(self.nClicks == 4):
                cv2.destroyWindow(self.FIELD_DISPLAY_NAME)

    def getCoordsByClick(self, image):
        cv2.imshow(self.FIELD_DISPLAY_NAME, image)
        cv2.setMouseCallback(self.FIELD_DISPLAY_NAME, self.getClick, 0)
        #  Press "Escape button" to exit
        while True:
            key = cv2.waitKey(10) & 0xff
            if key == 27 or self.nClicks >= 4:
                break

## create array to store image of warped football field
## 1112 x 745
field = np.zeros([1112,745])
TL = (float(0), float(0))
TR = (float(1112), float(0))
BL = (float(0), float(745))
BR = (float(1112), float(745))
fieldCoords = [TL, TR, BR, BL]
#clicksCaptor = ClicksCaptor()
#clicksCaptor.getCoordsByClick(o)
#birdEyeCoords = clicksCaptor.coords
#print birdEyeCoords
aTL = (float(643),float(52))
aTR = (float(1143), float(50))
aBR = (float(1880),float(223))
aBL = (float(20), float(226))
birdEyeCoords = [aTL, aTR, aBR, aBL]
print fieldCoords
print birdEyeCoords

Hmatrix, status = cv2.findHomography(np.array(birdEyeCoords), np.array(fieldCoords), 0)
print Hmatrix
#M = cv2.getPerspectiveTransform(fieldCoords, birdEyeCoords)
#print M
o2 = cv2.warpPerspective(o, Hmatrix, field.shape)
cv2.imwrite('outpu2.jpg', o2)
print len(o[0])
i.release()
