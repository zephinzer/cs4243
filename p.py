#!/usr/bin/python
### dependencies
import os, sys
# may need to remove for windows systems
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2, numpy as np
from matplotlib import pyplot

def readVideo(inputPath):
    cap = cv2.VideoCapture(inputPath)
    frameWidth = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    frameRate = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    frameCount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    frameFourCC = int(cap.get(cv2.cv.CV_CAP_PROP_FOURCC))
    return cap, frameWidth, frameHeight, frameRate, frameCount, frameFourCC

def extractBackground(videoStream, frameCount):
    _,img = videoStream.read()
    avgImg = np.float32(img)
    for fr in range(1,100): # TODO: change back to frameCount
        _,img = videoStream.read()
        avgImg = avgImg * ((fr)/float(fr+1)) + ( np.float32(img) / (fr + 1) );
        normImg = cv2.convertScaleAbs(avgImg) # convert into uint8 image
        #cv2.imshow('img',img)
        #cv2.imshow('normImg', normImg)
    return cv2.convertScaleAbs(avgImg)

i,w,h,r,c,cc = readVideo('input.avi')
print '[WIDTH] ', w
print '[HEIGHT]', h
print '[RATE]  ', r
print '[FRAMES]', c
print '[FOURCC]', cc

o = extractBackground(i,c)
cv2.imwrite('output.jpg', o)
print len(o), len(o[0])
i.release()


print r, h, w
vr = cv2.VideoWriter('output.avi', cv2.cv.CV_FOURCC("M","P","4","2"), r, (w,h))
for i in range(1,20):
    vr.write(o)
vr.release()
