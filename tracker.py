import cv2
import cv2.cv as cv
import numpy as np
import os
import homography as hm
from math import factorial

def moving_average(points, window_size=5):
    res = []
    half_window = window_size/2

    res += points[0:half_window]
    for i in range(half_window, len(points)-half_window):
        total_x = 0
        total_y = 0
        for j in range(-half_window, half_window):
            total_x += points[i+j][0]
            total_y += points[i+j][1]
        res.append( (int(total_x/window_size), int(total_y/window_size)) )
    res += points[len(points)-half_window:]
    return res

def moving_average2(points, window_size=11):
    res = []
    for i in range(0, len(points)):
        # get weighted sum of current and n-1 previous
        minimum = 0 if i - window_size < 0 else i - window_size + 1
        arr = points[minimum:i+1]
        total_x = 0
        total_y = 0
        for j in arr:
            total_x += j[0]
            total_y += j[1]
        res.append( [[int(total_x/len(arr)), int(total_y/len(arr))]] )
    return res

# returns a list of [list of points per player]
def homographed_points(pts):
    pts = np.array(pts)
    h, w, _ = pts.shape
    res = []
    # for each column(player),
    for i in range(0, w):
        player = pts[:,i]
        p = hm.coord_homography(player)
        res.append(moving_average2(p))
    return np.hstack(res)

# Performs closest pair matching between two sets of coordinates
# Returns a list of coordinates with length = len(pts1)
def estimate_points(pts, pts2, max_distance=80):
    pts1 = pts[-1]
    #if len(pts1) == len(pts2): return pts2
    candidates = [[] for i in pts1]
    picked = [False for i in pts2]
    result = [None for i in pts1]

    for i,p1 in enumerate(pts1):
        for j,p2 in enumerate(pts2):
            d = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            if d < max_distance:
                candidates[i].append((d, j))
    for i,point in enumerate(pts1):
        i = int(i)
        candidates[i].sort()
        for candidate in candidates[i]:
            j = candidate[1]
            #if j >= len(pts1): continue
            if picked[j]:
                continue
            else:
                picked[j] = True
                result[i] = pts2[j]
                break
        if result[i] == None:
            if len(candidates[i]) > 0:
                result[i] = pts2[candidates[i][0][1]]
            #else:
            #if len(pts) > 2:
            #    dx = 0.2 * (pts1[i][0] - pts[-2][i][0])
            #    dy = 0.2 * (pts1[i][1] - pts[-2][i][1])
            #    result[i] = [pts1[i][0] + dx, pts1[i][0] + dy]
            else:
                result[i] = pts1[i]
    return result

# Given an image (frame), find the centroid of contours within a certain color range
# returns the resulting list of coordinates and the mask of the selected range
def extract_points(frame, points, color_range, max_distance=80, size=150):
    mask = cv2.inRange(frame, color_range[0], color_range[1])
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    h,s,v = cv2.split(masked)

    blurred = cv2.GaussianBlur(h, (33,33), 3)
    kernel = np.ones([3,3], np.uint8)
    dilated = cv2.dilate(blurred, kernel, iterations=3)

    result = []
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < size:
            continue
        moment = cv2.moments(c)
        if moment['m00'] == 0:
            continue
        x = int(moment['m10'] / moment['m00'])
        y = int(moment['m01'] / moment['m00'])
        result.append((x,y))

    result = result if len(points) == 0 else estimate_points(points, result, max_distance)
    points.append(result)
    return result

def draw_bev(frame, points, color):
    for point in points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 10, color, -1)

def track(video, bg):
    # mask to remove everything outside the field
    mask = np.zeros(bg.shape, np.uint8)
    corners = np.array([[1870, 153], [3490, 144],[5750, 650], [100, 650]])
    cv2.fillPoly(mask, [corners.reshape((-1,1,2))], (255, 255, 255))

    _bg = cv2.imread("fakebg.png")
    h, w, _ = _bg.shape

    reader = cv2.VideoCapture(video)
    height, width, _ = bg.shape
    codec = cv.CV_FOURCC('M', 'P', '4', '2')
    writer2 = cv2.VideoWriter("testing2.avi", codec, 24, (width, height), True)
    writer = cv2.VideoWriter("testing.avi", codec, 24, (w, h), True)

    red_range = np.array([[0, 50, 120],[15,255,255]])
    blue_range = np.array([[90, 30, 30],[140,255,255]])
    yellow_range = np.array([[35, 130, 130],[40,255,255]])
    green_range = np.array([[45, 80, 80],[55,255,255]])
    white_range = np.array([[30, 10, 200],[40,30,255]])
    ball_range = np.array([[30, 30, 150],[40,90,185]])

    blue = []
    red = []
    yellow = []
    green = []
    white = []
    ball = [[(3174,213)]]

    fc = 2000
    for i in range(0, fc):
        print "Processing frame ", i, " of ", fc
        _, frame = reader.read()
        field = cv2.bitwise_and(mask, frame)

        subtracted = cv2.absdiff(frame, bg)
        subtracted_hsv = cv2.cvtColor(subtracted, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(subtracted_hsv)
        _, v = cv2.threshold(v, 50, 255, cv2.THRESH_BINARY)

        field = cv2.bitwise_and(field, field, mask=v)
        hsv = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)

        extract_points(hsv, blue, blue_range)
        extract_points(hsv, red, red_range)
        extract_points(hsv, green, green_range)
        extract_points(hsv, yellow, yellow_range)
        extract_points(hsv, white, white_range)
        #print len(blue[-1])
        draw_bev(frame, blue[-1], (255, 0, 0))
        draw_bev(frame, red[-1], (0, 0, 255))
        draw_bev(frame, green[-1], (0, 255, 0))
        draw_bev(frame, white[-1], (255, 255, 255))
        draw_bev(frame, yellow[-1], (140, 255, 100))
        #cv2.imwrite("test.png", frame)
        writer2.write(frame)

    blue = homographed_points(blue)
    red = homographed_points(red)
    green = homographed_points(green)
    white = homographed_points(white)
    yellow = homographed_points(yellow)

    for i in range(0, fc):
        bebg = _bg.copy()
        draw_bev(bebg, blue[i], (255, 0, 0))
        draw_bev(bebg, red[i], (0, 0, 255))
        draw_bev(bebg, green[i], (0, 255, 0))
        draw_bev(bebg, white[i], (255, 255, 255))
        draw_bev(bebg, yellow[i], (140, 255, 100))
        writer.write(bebg)

    reader.release()
    writer.release()
    writer2.release()

def main():
    video = os.getcwd() + "/input.avi"
    bg = cv2.imread(os.getcwd() + "/bg.png")
    track(video, bg)

if __name__ == "__main__":
    main()
