#!/usr/bin/python
import os, sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import cv2.cv as cv
import numpy as np


def track(video, bg):
    # mask to remove everything outside the field
    mask = np.zeros(bg.shape, np.uint8)
    corners = np.array([[1925, 159], [3423, 150],[5612, 664], [100, 673]])
    cv2.fillPoly(mask, [corners.reshape((-1,1,2))], (255, 255, 255))

    reader = cv2.VideoCapture(video)
    height, width, _ = bg.shape
    codec = cv.CV_FOURCC('M', 'P', '4', '2')
    writer = cv2.VideoWriter("test.avi", codec, 24, (width, height), True)

    kernel = np.ones((5,5),np.float32)/9
    for i in range(0, 120):
        _, frame = reader.read()

        #modified_frame = cv2.filter2D(frame,-1,kernel)
        modified_frame = cv2.GaussianBlur(frame, (3,3), 0);
        masked = cv2.bitwise_and(mask, modified_frame)


        # convert bgr to hsv
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([0,100,100])
        upper_blue = np.array([10,255,255])

        # Threshold the HSV image to get only blue colors
        bmask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND blue mask and original image
        res = cv2.bitwise_and(masked, masked, mask=bmask)

        h, s, v = cv2.split(res)
        countours, hier = cv2.findContours(h, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        threshold = 40
        markedPointsX = []
        markedPointsY = []
        for c in countours:
            moment = cv2.moments(c)
            if moment['m00'] == 0:
                continue
            # get mass center
            x = int(moment['m10'] / moment['m00'])
            y = int(moment['m01'] / moment['m00'])
            duplicate = False
            for j in range(len(markedPointsX)):
                if np.abs(x - markedPointsX[j]) < threshold:
                    for k in range(len(markedPointsY)):
                        if np.abs(y - markedPointsY[k]) < threshold:
                            duplicate = True
            if duplicate == False:
                cv2.circle(modified_frame, (x, y), 20, (255, 0, 0), 2)
        writer.write(modified_frame)

    reader.release()
    writer.release()


def main():
    video = os.getcwd() + "/input.avi"
    bg = cv2.imread(os.getcwd() + "/bg.png")
    track(video, bg)

if __name__ == "__main__":
    main()
