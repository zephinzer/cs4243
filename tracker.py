import cv2
import cv2.cv as cv
import numpy as np
import os

def extract(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    masked = cv2.bitwise_and(hsv, hsv, mask=mask)
    h,_,_ = cv2.split(masked)
    h = cv2.GaussianBlur(h, (15,15), 1)
    kernel = np.ones([5,3], np.uint8)
    h = cv2.dilate(h, kernel, iterations=3)

    result = []
    contours, hierarchy = cv2.findContours(h, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 250:
            continue
        moment = cv2.moments(c)
        if moment['m00'] == 0:
            continue
        x = int(moment['m10'] / moment['m00'])
        y = int(moment['m01'] / moment['m00'])
        #result.append((x,y))
        #cv2.circle(frame, (x,y), 20, (0, 0, 255), 2)
        #rx, ry, rw, rh = cv2.boundingRect(c)
        #cv2.rectangle(frame, (rx,ry), (rx+rw, ry+rh), c, 2)
        x,y,w,h = cv2.boundingRect(c)
        result.append((x,y,w,h,cv2.contourArea(c)))
    return result

def draw(frame, coord, color):
    for points in coord:
        #cv2.circle(frame, points, 20, (0, 0, 255), 2)
        rx, ry, rw, rh, a = points
        cv2.putText(frame, str(a), (rx,ry+rh), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255))
        cv2.rectangle(frame, (rx,ry), (rx+rw, ry+rh), color, 2)
    return frame

def track(video, bg):
    # mask to remove everything outside the field
    mask = np.zeros(bg.shape, np.uint8)
    corners = np.array([[1870, 159], [3500, 150],[5750, 650], [100, 650]])
    cv2.fillPoly(mask, [corners.reshape((-1,1,2))], (255, 255, 255))

    reader = cv2.VideoCapture(video)
    height, width, _ = bg.shape
    codec = cv.CV_FOURCC('M', 'P', '4', '2')
    writer = cv2.VideoWriter("test.avi", codec, 24, (width, height), True)

    red1 = np.array([0, 80, 80])
    red2 = np.array([10, 255, 255])
    blue1 = np.array([100, 30, 10])
    blue2 = np.array([130, 255, 255])
    yellow1 = np.array([30, 100, 210])
    yellow2 = np.array([50, 140, 255])
    fc = 120
    for i in range(0, fc):
        print "Processing frame ", i, " of ", fc
        _, frame = reader.read()
        # remove background and everything outside the field
        diff = cv2.absdiff(frame, bg)
        field_bgs = cv2.bitwise_and(mask, diff)
        field = cv2.bitwise_and(mask, frame)

        hsv = cv2.cvtColor(field_bgs, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        _, v = cv2.threshold(v, 60, 255, cv2.THRESH_BINARY)
        #v = cv2.GaussianBlur(v, (33, 33), 3)

        field = cv2.bitwise_and(field, field, mask=v)
        hsv = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)

        red_players = extract(hsv, red1, red2)
        blue_players = extract(hsv, blue1, blue2)
        yellow_players = extract(hsv, yellow1, yellow2)

        draw(frame, red_players, (0,0,255))
        draw(frame, blue_players, (255,0,0))
        draw(frame, yellow_players, (255,255,0))

        #cv2.imwrite("testing.png", field)
        writer.write(frame)
    reader.release()
    writer.release()

def main():
    video = os.getcwd() + "/input.avi"
    bg = cv2.imread(os.getcwd() + "/bg.png")
    track(video, bg)

if __name__ == "__main__":
    main()
