#!/usr/bin/python
import os, sys
# may need to remove for windows systems
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2, numpy as np
from matplotlib import pyplot

class ClicksCaptor:
    FIELD_DISPLAY_NAME = 'Field'
    coords = []
    nClicks = 0
    nMaxClicks = None

    def __init__(self, maxClicks):
        self.nMaxClicks = maxClicks

    ## get the videoCoords with Clicks
    def getClick(self, e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            print x, y
            self.coords.append([[np.float32(x), np.float32(y)]])
            print self.nClicks
            self.nClicks += 1
            if(self.nClicks == self.nMaxClicks):
                cv2.destroyWindow(self.FIELD_DISPLAY_NAME)

    def getCoordsByClick(self, image):
        cv2.imshow(self.FIELD_DISPLAY_NAME, image)
        cv2.setMouseCallback(self.FIELD_DISPLAY_NAME, self.getClick, 0)
        #  Press "Escape button" to exit
        while True:
            key = cv2.waitKey(10) & 0xff
            if key == 27 or self.nClicks >= self.nMaxClicks:
                break

    def getResults(self):
        return self.coords
