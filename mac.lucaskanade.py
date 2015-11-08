#!/usr/bin/python
### Lucas Kanade

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

cap = cv2.VideoCapture('input.avi');

frameWidth = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
frameRate = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
frameCount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
print '[WIDTH] ', frameWidth
print '[HEIGHT]', frameHeight
print '[RATE]  ', frameRate
print '[FRAMES]', frameCount
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 1,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#blue players first 11 = 11
#red players second 11 = 22
#referee last = 23
#football last = 24
clicksCaptor = ClicksCaptor(2)
clicksCaptor.getCoordsByClick(old_frame)
p0 = np.array(clicksCaptor.getResults())
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

for fr in range(1, frameCount):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5,5), 0)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1 #[st==1]
    good_old = p0 #[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(   mask, (a,b),    (c,d),  [0, 0, 255],    2)
        cv2.circle( frame,(a,b),    5,      [0, 0, 255],    -1)
    img = cv2.add(frame, mask)
    if img != None:
        cv2.imshow('frame',img)
        while True:
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()
