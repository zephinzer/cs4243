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

fgmask = cv2.BackgroundSubtractorMOG()
bgimage = cv2.imread('bg.png')
cap,w,h,r,fc,fcc = readVideo('input.avi')
#wrt = cv2.VideoWriter('output.avi', cv2.cv.CV_FOURCC(*"MP42"), r, (w,h))
wrt = cv2.VideoWriter('output.mov', cv2.cv.CV_FOURCC(*"mp4v"), r, (w,h))
for fr in range(1, fc):
    _, img = cap.read()
    #mask = fgmask.apply(img, mask, 1)
    frame = cv2.subtract(img, bgimage)
    wrt.write(frame)
cap.release()
wrt.release()
