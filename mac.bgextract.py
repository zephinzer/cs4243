#!/usr/bin/python
### dependencies
import os, sys
# may need to remove for windows systems
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2, numpy as np
from matplotlib import pyplot

def bg_extract(video):
    cap = cv2.VideoCapture(video)
    fcount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    _, img = cap.read()
    avgImg = np.float32(img)
    for fr in range(1, fcount):
        _, img = cap.read()
        img = np.float32(img)
        avgImg = np.add(np.multiply(avgImg, fr/(fr+1.0)), np.divide(img, fr))
    normImg = cv2.convertScaleAbs(avgImg) # convert into uint8 image
    cap.release()
    return normImg

def main():
    bg = bg_extract(os.getcwd() + "/input.avi")
    cv2.imwrite(os.getcwd() + "/bg.png", bg)

if __name__ == "__main__":
    main()
