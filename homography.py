import cv2
import numpy as np

bev_field_corners = np.array([[0,0], [1112,0], [1113,745], [0,745]], np.float64)
field_corners = np.array([[1930,156], [3418, 146], [5770, 690], [0, 690]], np.float64)
Hmatrix, _ = cv2.findHomography(field_corners, bev_field_corners, 0)
IHmatrix, _ = cv2.findHomography(bev_field_corners, field_corners, 0)

def img_homography(img, inverse=False):
    h = IHmatrix if inverse else Hmatrix
    return cv2.warpPerspective(inputImage, Hmatrix, (1112, 745))

# takes in a list of coordinates
def coord_homography(coord_list, inverse=False):
    res = []
    matrix = IHmatrix if inverse else Hmatrix
    for p in coord_list:
        q = matrix * np.matrix([[p[0]],[p[1]],[1]])
        q /= q[2]
        res.append( (int(q.item(0)), int(q.item(1))) )
    return res